import AppKit
import CoreGraphics
import Monitoring

/// 상대커서 모드 — 진짜(네이티브) 커서를 그대로 포인터로 쓰고, 마우스 이벤트만 소스 좌표로
/// 재타깃해 뷰어 아래 소스가 호버·스크롤·클릭·드래그를 받게 한다.
///
/// **CGEventTap은 전용 고우선순위 스레드에서 돌린다.** 액티브(동기) 탭은 윈도우서버가 핸들러
/// 응답을 기다리는데, 메인 런루프에 붙이면 소스 활성화로 백그라운드가 된 MacFG의 메인 런루프가
/// 부하/저우선순위로 늦게 서비스될 때 탭이 굶주려 — 특히 소스 앱의 드래그(모달 선택) 루프가
/// 매 이벤트를 기다리다 수 초 멈춘다(실측 증상). 전용 스레드는 메인과 무관하게 즉시 서비스된다.
///
/// 탭 스레드는 AppKit을 만지지 않는다 — 필요한 지오메트리(창/레터박스/소스프레임/디스플레이)는
/// 메인이 락 보호 스냅샷으로 밀어주고, 매핑 수학은 순수 함수로 탭 스레드에서 수행한다.
final class RelativePointer {
    /// 지오메트리 스냅샷 (전부 값 타입 — 스레드 안전) — 메인이 push, 탭 스레드가 read.
    struct Geometry: Sendable {
        var displayID: CGDirectDisplayID = CGMainDisplayID()
        var windowFrame: CGRect = .zero        // 뷰어 창 프레임 (NS)
        var letterbox: CGRect = .zero          // 레터박스 (창 content 좌표, NS)
        var sourceFrameNS: CGRect = .zero      // 소스 창 프레임 (NS)
        var primaryHeight: CGFloat = 0         // 주 스크린 높이 (NS↔CG 반전용)
        var sourcePID: pid_t = 0               // 복귀 시 재활성 대상 (Sendable — 스레드 넘김 안전)
    }

    private let geoLock = NSLock()
    private var geo = Geometry()
    func setGeometry(_ g: Geometry) { geoLock.lock(); geo = g; geoLock.unlock() }

    private(set) var active = false
    private var tap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var thread: Thread?
    private var threadRunLoop: CFRunLoop?
    private let threadReady = DispatchSemaphore(value: 0)
    private let threadStopped = DispatchSemaphore(value: 0)
    private var wasOutside = false                   // 직전 이벤트가 옆 모니터였나 (탭 스레드 전용)
    private var dragFrameNS: CGRect?                 // 버튼 다운 중 고정된 소스 프레임 (탭 스레드 전용)

    // MARK: - 진입/이탈

    func enable() {
        guard !active else { return }
        wasOutside = false
        dragFrameNS = nil
        let t = Thread { [weak self] in self?.threadMain() }
        t.name = "MacFG.RelPointerTap"
        t.qualityOfService = .userInteractive
        thread = t
        t.start()
        threadReady.wait()                            // 탭 생성 완료 대기
        active = (tap != nil)
        DiagnosticLog.shared.log("[RELPTR] enabled (dedicated thread, tap=\(tap != nil))")
    }

    private func threadMain() {
        let rl = CFRunLoopGetCurrent()
        threadRunLoop = rl
        let mask: CGEventMask =
            (1 << CGEventType.mouseMoved.rawValue) |
            (1 << CGEventType.leftMouseDragged.rawValue) |
            (1 << CGEventType.rightMouseDragged.rawValue) |
            (1 << CGEventType.otherMouseDragged.rawValue) |
            (1 << CGEventType.leftMouseDown.rawValue) |
            (1 << CGEventType.leftMouseUp.rawValue) |
            (1 << CGEventType.rightMouseDown.rawValue) |
            (1 << CGEventType.rightMouseUp.rawValue) |
            (1 << CGEventType.otherMouseDown.rawValue) |
            (1 << CGEventType.otherMouseUp.rawValue) |
            (1 << CGEventType.scrollWheel.rawValue)
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
                                          options: .defaultTap, eventsOfInterest: mask,
                                          callback: relativePointerTapCallback, userInfo: refcon) else {
            DiagnosticLog.shared.log("[RELPTR] tap create FAILED (accessibility 권한 필요)")
            threadReady.signal()
            return
        }
        self.tap = tap
        let src = CFMachPortCreateRunLoopSource(nil, tap, 0)
        self.runLoopSource = src
        CFRunLoopAddSource(rl, src, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        threadReady.signal()
        CFRunLoopRun()                                // disable()의 CFRunLoopStop까지 실행
        threadStopped.signal()                        // 종료 확정 신호 (disable이 대기)
    }

    func disable() {
        guard active else { return }
        active = false
        guard let rl = threadRunLoop else { return }
        let tapRef = tap, srcRef = runLoopSource
        // 정리는 탭을 만든 그 스레드에서 (교차 스레드 레이스 방지) → 런루프 정지 → 스레드 종료
        CFRunLoopPerformBlock(rl, CFRunLoopMode.commonModes.rawValue) {
            if let tapRef { CGEvent.tapEnable(tap: tapRef, enable: false) }
            if let srcRef { CFRunLoopRemoveSource(rl, srcRef, .commonModes) }
            CFRunLoopStop(rl)
        }
        CFRunLoopWakeUp(rl)
        // **스레드 종료를 동기 대기** — 비동기로 두면 스레드가 아직 도는 중 소유자(OverlayWindow)가
        // dealloc돼 refcon(passUnretained self)이 dangling → 힙 손상(동시 해제되는 다른 객체에서
        // over-release로 표출). 종료 확정 후 정리한다(0.5s 타임아웃 — 최악에도 앱은 안 멈춤).
        _ = threadStopped.wait(timeout: .now() + 0.5)
        tap = nil; runLoopSource = nil; thread = nil; threadRunLoop = nil; dragFrameNS = nil
        DiagnosticLog.shared.log("[RELPTR] disabled")
    }

    fileprivate func reenableTap() {
        if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
    }

    // MARK: - 매핑 (순수 함수, 탭 스레드에서 스냅샷으로)

    /// 전역 CG 좌표를 레터박스(영상 표시 영역) 안으로 클램프 — 드래그 up이 블랙바에서
    /// 떨어져도 시퀀스를 닫을 수 있게 가장자리로 끌어온다.
    private func clampToLetterbox(_ p: CGPoint, _ g: Geometry) -> CGPoint {
        let lb = g.letterbox
        guard lb.width > 1, lb.height > 1 else { return p }
        // 창 content NS 기준으로 레터박스 경계를 전역 CG로 환산
        let originX = g.windowFrame.minX + lb.minX
        let originYTop = g.primaryHeight - (g.windowFrame.minY + lb.maxY)
        return CGPoint(x: min(max(p.x, originX + 0.5), originX + lb.width - 0.5),
                       y: min(max(p.y, originYTop + 0.5), originYTop + lb.height - 0.5))
    }

    /// 커서 전역 CG → 소스 전역 CG. frameOverride면 그 프레임 기준(드래그 고정), clickInset면
    /// 소스 프레임 8pt 안쪽(리사이즈 존 회피). 레터박스 밖이면 nil.
    private func mapToSource(_ p: CGPoint, frameOverride: CGRect?, clickInset: Bool, _ g: Geometry) -> CGPoint? {
        let lb = g.letterbox
        guard lb.width > 1, lb.height > 1 else { return nil }
        // 전역 CG → 창 content NS
        let winX = p.x - g.windowFrame.minX
        let winY = (g.primaryHeight - p.y) - g.windowFrame.minY
        let nx = (winX - lb.minX) / lb.width
        let ny = (winY - lb.minY) / lb.height          // NS: 0=하단
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return nil }
        let frame = frameOverride ?? g.sourceFrameNS
        guard frame.width > 1, frame.height > 1 else { return nil }
        var sxNS = frame.minX + nx * frame.width
        var syNS = frame.minY + ny * frame.height
        // 프레임 전체를 인셋하면 매핑이 중심으로 압축돼 커서와 클릭 위치가 어긋난다(업스케일 배율만큼
        // 증폭 — 오른쪽 클릭이 왼쪽으로 쏠림, 실측). 대신 **매핑된 위치만 가장자리 3pt 안으로 클램프** —
        // 내부는 커서와 1:1 정확하고, 창 테두리(리사이즈 존)만 회피.
        if clickInset, frame.width > 40, frame.height > 40 {
            sxNS = min(max(sxNS, frame.minX + 3), frame.maxX - 3)
            syNS = min(max(syNS, frame.minY + 3), frame.maxY - 3)
        }
        return CGPoint(x: sxNS, y: g.primaryHeight - syNS)
    }

    // MARK: - 탭 처리 (탭 스레드)

    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        geoLock.lock(); let g = geo; geoLock.unlock()
        let loc = event.location
        let screen = CGDisplayBounds(g.displayID)

        // 옆 모니터 — 손대지 않음 (평범한 다중모니터 조작)
        if !screen.contains(loc) {
            wasOutside = true
            dragFrameNS = nil
            return Unmanaged.passUnretained(event)
        }
        if wasOutside {
            wasOutside = false
            let pid = g.sourcePID                       // 옆 모니터 복귀 — 소스 재활성 (메인에서)
            if pid != 0 {
                DispatchQueue.main.async { NSRunningApplication(processIdentifier: pid)?.activate() }
            }
        }

        switch type {
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            dragFrameNS = g.sourceFrameNS               // 드래그 매핑 고정 (down 시점 프레임)
        default:
            break
        }
        let clickLike: Bool
        switch type {
        case .leftMouseDown, .leftMouseUp, .rightMouseDown, .rightMouseUp,
             .otherMouseDown, .otherMouseUp,
             .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            clickLike = true
        default:
            clickLike = false
        }
        // 드래그 시퀀스(down~up)는 반드시 닫아야 한다 — up이 레터박스 밖(블랙바)에서 떨어지면
        // 매핑이 nil이라 그대로 소비돼, down을 받은 소스 앱이 버튼이 눌린 상태로 갇힌다
        // (스크러버가 계속 따라오거나 텍스트 선택 유지, 리뷰 확정). up과 드래그 중 dragged는
        // 밖이어도 가장자리로 클램프해 배달한다.
        let inDragSequence = dragFrameNS != nil
        let mustDeliver: Bool
        switch type {
        case .leftMouseUp, .rightMouseUp, .otherMouseUp: mustDeliver = true
        case .leftMouseDragged, .rightMouseDragged, .otherMouseDragged: mustDeliver = inDragSequence
        default: mustDeliver = false
        }
        var mapped = mapToSource(loc, frameOverride: dragFrameNS, clickInset: clickLike, g)
        if mapped == nil, mustDeliver {
            mapped = mapToSource(clampToLetterbox(loc, g), frameOverride: dragFrameNS,
                                 clickInset: clickLike, g)
        }
        switch type {
        case .leftMouseUp, .rightMouseUp, .otherMouseUp: dragFrameNS = nil
        default: break
        }

        guard let mapped else { return nil }            // 레터박스 밖(블랙바) — 소비
        event.location = mapped
        event.setDoubleValueField(.mouseEventDeltaX, value: 0)
        event.setDoubleValueField(.mouseEventDeltaY, value: 0)
        return Unmanaged.passUnretained(event)
    }
}

/// CGEventTap C 콜백 — refcon으로 인스턴스 복원. 전용 탭 스레드에서 발화.
private func relativePointerTapCallback(proxy: CGEventTapProxy, type: CGEventType,
                                        event: CGEvent, refcon: UnsafeMutableRawPointer?) -> Unmanaged<CGEvent>? {
    guard let refcon else { return Unmanaged.passUnretained(event) }
    let instance = Unmanaged<RelativePointer>.fromOpaque(refcon).takeUnretainedValue()
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        instance.reenableTap()
        return Unmanaged.passUnretained(event)
    }
    return instance.handle(type: type, event: event)
}
