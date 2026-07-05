import AppKit
import CoreGraphics
import Monitoring

/// 상대커서 모드 — 진짜(네이티브) 커서를 그대로 포인터로 쓰고, 마우스 이벤트만 소스 좌표로
/// 재타깃해 뷰어 아래 소스가 호버·스크롤·클릭·드래그를 받게 한다.
///
/// 핵심 통찰(실측): CGEventTap에서 event.location을 바꿔도 **진짜 커서는 움직이지 않는다** —
/// 배달 경로만 바뀐다. 따라서 진짜 커서는 늘 뷰어 위(가리키는 콘텐츠 위)에 있고, 그 위치를
/// 소스로 매핑해 재타깃하면 커서가 있는 콘텐츠에 정확히 이벤트가 간다. 링·숨김·디커플 불필요.
///
/// 동작:
/// - 커서가 뷰어 화면 안이면: 커서 위치(event.location)를 소스 좌표로 매핑해 재타깃, 통과.
///   뷰어는 클릭투과라 재타깃 이벤트가 아래 소스로 도달. 레터박스 밖(블랙바)이면 소비.
/// - 커서가 옆 모니터로 나가면: 무변조 통과(평범한 다중모니터 조작). 돌아오면 소스 재활성.
/// - 클릭/드래그는 소스 프레임 8pt 안쪽 인셋(가장자리 리사이즈 존 회피), 드래그는 down 시점
///   프레임을 up까지 고정(창이 제 드래그를 쫓는 폭주 방지).
///
/// 진짜 커서를 숨기거나 디커플하지 않으므로 크래시로도 마우스가 잠기지 않는다(안전).
final class RelativePointer {
    /// 뷰어가 있는 디스플레이 ID (CG 전역좌표계 화면 판정 기준)
    var viewerDisplayID: () -> CGDirectDisplayID = { CGMainDisplayID() }
    /// 현재 소스 창 프레임(NS) — 드래그 시작 시 스냅샷용
    var currentSourceFrameNS: () -> CGRect = { .zero }
    /// 커서 전역 CG → 소스 전역 CG. frameOverride 있으면 그 프레임 기준(드래그 고정),
    /// clickInset이면 리사이즈 존 회피 인셋. 레터박스 밖이면 nil.
    var mapVirtualToSource: (CGPoint, CGRect?, Bool) -> CGPoint? = { _, _, _ in nil }
    /// 옆 모니터에서 뷰어 화면으로 복귀 — 소스 재활성(호버 좌표 필수)
    var onReturnedToViewer: () -> Void = {}

    private(set) var active = false
    private var tap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var wasOutside = false                   // 직전 이벤트가 옆 모니터였나
    private var dragFrameNS: CGRect?                 // 버튼 다운 중 고정된 소스 프레임

    // MARK: - 진입/이탈

    func enable(on screen: NSScreen?) {
        guard !active else { return }
        wasOutside = false
        dragFrameNS = nil

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
            return
        }
        self.tap = tap
        let rls = CFMachPortCreateRunLoopSource(nil, tap, 0)
        self.runLoopSource = rls
        CFRunLoopAddSource(CFRunLoopGetMain(), rls, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        active = true
        DiagnosticLog.shared.log("[RELPTR] enabled (native cursor, no ring)")
    }

    func disable() {
        guard active else { return }
        active = false
        dragFrameNS = nil
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let rls = runLoopSource { CFRunLoopRemoveSource(CFRunLoopGetMain(), rls, .commonModes) }
        tap = nil
        runLoopSource = nil
        DiagnosticLog.shared.log("[RELPTR] disabled")
    }

    /// 탭 타임아웃/사용자입력으로 비활성화되면 재활성 (콜백에서 호출)
    fileprivate func reenableTap() {
        guard let tap else { return }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    // MARK: - 탭 처리

    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        let loc = event.location                      // 진짜 커서 위치 (비디커플)
        let screen = CGDisplayBounds(viewerDisplayID())

        // 옆 모니터 — 손대지 않음 (평범한 다중모니터 조작)
        if !screen.contains(loc) {
            wasOutside = true
            dragFrameNS = nil
            return Unmanaged.passUnretained(event)
        }
        // 뷰어 화면으로 복귀 — 소스 재활성 (옆에서 다른 앱을 활성화했을 수 있음)
        if wasOutside {
            wasOutside = false
            onReturnedToViewer()
        }

        // 드래그 매핑 고정 (down에서 스냅샷 → up까지 유지)
        switch type {
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            dragFrameNS = currentSourceFrameNS()
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
        let mapped = mapVirtualToSource(loc, dragFrameNS, clickLike)
        switch type {
        case .leftMouseUp, .rightMouseUp, .otherMouseUp: dragFrameNS = nil
        default: break
        }

        guard let mapped else { return nil }          // 레터박스 밖(블랙바) — 소비
        event.location = mapped                        // 재타깃 (커서는 안 움직임 — 배달만)
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
