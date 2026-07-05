import AppKit
import CoreGraphics
import Monitoring

/// 상대커서 모드 — 진짜 커서를 소스 창 안에 상주시켜 호버·스크롤·클릭·드래그를 전부 "진짜"
/// 이벤트로 동작시킨다. (Lossless Scaling이 게임에서 쓰는 상대입력의 영상판.)
///
/// 원리:
/// 1. 뷰어를 클릭투과(ignoresMouseEvents=true)로 만든다 — 재게시 이벤트가 뷰어를 통과해
///    아래 소스 창으로 도달하도록.
/// 2. CGEventTap(세션)으로 하드웨어 마우스를 **시스템 레벨**에서 가로챈다 — 뷰어의 클릭투과와
///    무관하게 모든 마우스 이벤트를 먼저 잡는다.
/// 3. 하드웨어 delta를 누적해 가상 커서(뷰어 레터박스 내)를 옮기고, 그 지점을 소스 스크린
///    좌표로 매핑한다.
/// 4. 원본 이벤트를 복제(모든 필드 보존: clickState/스크롤 위상/모멘텀 등)해 소스 좌표로
///    재타깃, 매직 태그를 박아 재게시한다. 클릭투과 뷰어를 통과해 소스가 진짜 이벤트를 받는다.
/// 5. 원본은 소비(return nil). 뷰어엔 합성 커서를 그려 어디를 가리키는지 보여준다.
///
/// 안전장치: 진짜 커서는 디커플(CGAssociateMouseAndMouseCursorPosition=false)+숨김이라
/// 사용자가 통제 불능이 되면 위험 → disable/close/willTerminate에서 반드시 복구하고,
/// macOS는 프로세스 종료 시 디커플·커서숨김을 자동 복원하므로(크래시해도 앱만 끄면 마우스 정상)
/// 최후 안전망이 된다.
final class RelativePointer {
    /// 뷰어 contentView 내 레터박스(NS bottom-left) — 최신값 제공.
    var letterboxProvider: () -> CGRect = { .zero }
    /// 뷰어 contentView NS 점 → 소스 스크린 CG(top-left). 매핑 불가면 nil.
    var mapToSourceGlobal: (CGPoint) -> CGPoint? = { _ in nil }
    /// 합성 커서를 뷰어 NS 좌표에 배치.
    var onSyntheticCursor: (CGPoint) -> Void = { _ in }

    private(set) var active = false
    private var tap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var virtualViewer: CGPoint = .zero      // 가상 커서(뷰어 contentView NS 좌표)
    private var hiddenDisplay: CGDirectDisplayID = CGMainDisplayID()
    private var termObserver: NSObjectProtocol?
    private let postSource = CGEventSource(stateID: .hidSystemState)

    /// 재게시 이벤트 식별 태그 — 탭이 자기 이벤트를 되잡지 않게.
    private static let magic: Int64 = 0x6D_61_63_66   // "macf"

    // MARK: - 진입/이탈

    func enable(initialViewer: CGPoint, on screen: NSScreen?) {
        guard !active else { return }
        virtualViewer = initialViewer
        hiddenDisplay = screen.flatMap {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
        } ?? CGMainDisplayID()

        let mask: CGEventMask =
            (1 << CGEventType.mouseMoved.rawValue) |
            (1 << CGEventType.leftMouseDragged.rawValue) |
            (1 << CGEventType.rightMouseDragged.rawValue) |
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
            return
        }
        self.tap = tap
        let rls = CFMachPortCreateRunLoopSource(nil, tap, 0)
        self.runLoopSource = rls
        CFRunLoopAddSource(CFRunLoopGetMain(), rls, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        // 커서 숨김만. **디커플은 하지 않음** — 디커플하면 진짜 커서가 얼어붙어 그 위치의
        // hover가 윈도우서버에서 별도 생성돼 소스로 새어든다(실측: 정타깃 뒤 물리커서 위치로 2차 move).
        // 탭이 모든 move의 location을 소스 좌표로 재타깃하므로 진짜 커서가 소스 지점으로 실제 이동,
        // 물리 위치=재타깃 위치가 되어 충돌 후속이 없다.
        CGDisplayHideCursor(hiddenDisplay)

        // 최후 안전망 — 앱 종료 시 확실히 복구 (프로세스 종료 자동복원과 별개로 명시).
        // self 대신 display 값만 캡처(Sendable) — 종료 경로에서 인스턴스 상태 접근 없이 CG 직접 복원.
        let display = hiddenDisplay
        termObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { _ in
            CGAssociateMouseAndMouseCursorPosition(1)
            CGDisplayShowCursor(display)
        }

        active = true
        onSyntheticCursor(virtualViewer)
        DiagnosticLog.shared.log("[RELPTR] enabled viewer(\(Int(initialViewer.x)),\(Int(initialViewer.y))) display=\(hiddenDisplay)")
    }

    func disable() {
        guard active else { return }
        active = false
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let rls = runLoopSource { CFRunLoopRemoveSource(CFRunLoopGetMain(), rls, .commonModes) }
        tap = nil
        runLoopSource = nil
        if let termObserver { NotificationCenter.default.removeObserver(termObserver) }
        termObserver = nil
        restoreCursorState()
        DiagnosticLog.shared.log("[RELPTR] disabled")
    }

    private func restoreCursorState() {
        CGAssociateMouseAndMouseCursorPosition(1)   // 혹시 켜졌어도 확실히 (무해)
        CGDisplayShowCursor(hiddenDisplay)
    }

    /// 탭 타임아웃/사용자입력으로 비활성화되면 재활성 (콜백에서 호출)
    fileprivate func reenableTap() {
        guard let tap else { return }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    // MARK: - 탭 처리

    /// C 콜백에서 호출. **원본 이벤트의 location만 소스 좌표로 바꿔 그대로 통과**시킨다 —
    /// 새 이벤트를 만들지(copy+post) 않으므로 커서 재배치가 유발하는 2차 move·피드백 루프가 없다.
    /// 뷰어가 클릭투과라 재타깃된 이벤트는 아래 소스로 라우팅된다.
    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        let lb = letterboxProvider()
        guard lb.width > 1, lb.height > 1 else { return nil }   // 레터박스 미준비 — 소비

        // 이동/드래그: delta 누적 → 가상 커서 이동 (NS y는 위로, delta는 아래 양수라 반전)
        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            let dx = event.getDoubleValueField(.mouseEventDeltaX)
            let dy = event.getDoubleValueField(.mouseEventDeltaY)
            virtualViewer.x = min(max(virtualViewer.x + dx, lb.minX), lb.maxX)
            virtualViewer.y = min(max(virtualViewer.y - dy, lb.minY), lb.maxY)
            onSyntheticCursor(virtualViewer)
        default:
            break
        }

        guard let cg = mapToSourceGlobal(virtualViewer) else { return nil }
        event.location = cg   // 원본을 소스 좌표로 재타깃 (새 이벤트 없음 → 피드백 없음)
        // delta 제거 — 절대 location만 유효; 큰 delta는 윈도우서버가 중간 move를 합성해 다중 배달
        event.setDoubleValueField(.mouseEventDeltaX, value: 0)
        event.setDoubleValueField(.mouseEventDeltaY, value: 0)
        return Unmanaged.passUnretained(event)
    }
}

/// CGEventTap C 콜백 — refcon으로 인스턴스 복원. 탭은 메인 런루프에 붙어 메인 스레드에서 발화.
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
