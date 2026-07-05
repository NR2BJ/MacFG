import AppKit
import CoreGraphics
import Monitoring

/// 백그라운드 앱의 커서 숨김 허용 — CGDisplayHideCursor는 호출 앱이 활성일 때만 먹히는데,
/// 상대커서 모드는 소스 앱을 활성화하므로 MacFG는 백그라운드다. 비공개 SkyLight/CGS의
/// "SetsCursorInBackground" 연결 속성을 켜면 백그라운드에서도 숨김이 적용된다(BetterTouchTool 등
/// 표준 우회). 심볼을 dlsym으로 찾아 없으면 무해하게 건너뛴다.
private func allowBackgroundCursorControl() {
    typealias MainConnFn = @convention(c) () -> UInt32
    typealias SetPropFn = @convention(c) (UInt32, UInt32, CFString, CFTypeRef) -> Int32
    guard let handle = dlopen(nil, RTLD_LAZY) else { return }
    defer { dlclose(handle) }
    guard let mc = dlsym(handle, "CGSMainConnectionID"),
          let sp = dlsym(handle, "CGSSetConnectionProperty") else {
        DiagnosticLog.shared.log("[RELPTR] CGS 심볼 없음 — 백그라운드 커서 숨김 불가")
        return
    }
    let mainConn = unsafeBitCast(mc, to: MainConnFn.self)
    let setProp = unsafeBitCast(sp, to: SetPropFn.self)
    let cid = mainConn()
    _ = setProp(cid, cid, "SetsCursorInBackground" as CFString, kCFBooleanTrue)
}

/// 상대커서 모드 — 진짜 커서를 소스 창 안에 상주시켜 호버·스크롤·클릭·드래그를 전부 "진짜"
/// 이벤트로 동작시킨다. (Lossless Scaling이 게임에서 쓰는 상대입력 + 멀티모니터 모드의 영상판.)
///
/// 동작:
/// - CGEventTap(세션)이 마우스를 시스템 레벨에서 가로채고, delta를 누적해 **가상 커서**를
///   뷰어 화면(CG 전역좌표) 위에서 굴린다. 뷰어엔 합성 링 커서를 그린다(진짜 커서는 숨김).
/// - 가상 커서가 레터박스 안이면: 원본 이벤트의 location만 소스 좌표로 재타깃해 그대로 통과
///   (copy+post 아님 — 새 이벤트는 피드백 루프를 만든다, 실측). 클릭투과 뷰어를 지나
///   아래 소스 창에 진짜 이벤트로 배달된다. 레터박스 밖(블랙바)이면 소비.
/// - **클릭/드래그는 소스 창 프레임 8pt 안쪽으로 인셋** — 창 가장자리 리사이즈 존을 매핑에서
///   제외해 숨은 커서로 창이 리사이즈되는 사고 차단.
/// - **드래그 중엔 소스 프레임 스냅샷 고정** — 드래그가 창을 움직이면(타이틀바/PiP) 매핑
///   기준이 함께 움직여 창이 제 드래그를 쫓는 폭주가 생기므로, down~up 동안 down 시점
///   프레임으로 매핑한다.
/// - **멀티모니터 탈출**: 가상 커서가 뷰어 화면 밖 인접 모니터로 넘어가면 상대모드를 일시정지 —
///   진짜 커서를 그 지점에 보여주고 모든 이벤트를 무변조 통과(옆 화면 정상 조작). 커서가
///   뷰어 화면으로 돌아오면 재캡처(숨김+가상커서 재개+소스 재활성). cmd-tab 등 키보드는
///   애초에 안 건드린다.
///
/// 안전장치: 디커플은 쓰지 않으므로(재타깃이 커서를 실제로 움직임) 크래시로 복구를 못 해도
/// 커서가 잠기지 않는다. 숨김은 disable/close/willTerminate에서 복원 + 프로세스 종료 시 자동 복원.
final class RelativePointer {
    // MARK: 배선 (OverlayWindow 제공)
    /// 뷰어가 있는 디스플레이 ID (CG 전역좌표계의 화면 판정 기준)
    var viewerDisplayID: () -> CGDirectDisplayID = { CGMainDisplayID() }
    /// 현재 소스 창 프레임(NS) — 드래그 시작 시 스냅샷용
    var currentSourceFrameNS: () -> CGRect = { .zero }
    /// 가상 커서(전역 CG) → 소스 전역 CG. frameOverride 있으면 그 프레임 기준(드래그 고정).
    /// clickInset이면 리사이즈 존 회피 인셋 적용. 레터박스 밖이면 nil.
    var mapVirtualToSource: (CGPoint, CGRect?, Bool) -> CGPoint? = { _, _, _ in nil }
    /// 합성 커서 배치 (전역 CG — OverlayWindow가 창 좌표로 변환해 그림)
    var onSyntheticCursor: (CGPoint) -> Void = { _ in }
    /// 탈출(true)/복귀(false) — 링 숨김/표시 + 복귀 시 소스 재활성화
    var onSuspendChange: (Bool) -> Void = { _ in }

    private(set) var active = false
    private var suspended = false                    // 옆 모니터로 탈출 중 (무변조 통과)
    /// 진짜 커서 모드 — 소스가 화면에서 거의 풀스크린(≈1:1)이면 진짜 커서가 가리키는 위치와
    /// 거의 겹치므로 링을 끄고 진짜(네이티브) 커서를 그대로 보여준다. 진짜 업스케일이면 false
    /// (진짜 커서는 구석에 눌려 어긋나므로 숨기고 링을 그린다).
    private var useNativeCursor = false
    private var tap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var virtualCG: CGPoint = .zero           // 가상 커서 (전역 CG top-left)
    private var dragFrameNS: CGRect?                 // 버튼 다운 중 고정된 소스 프레임
    private var hiddenDisplay: CGDirectDisplayID = CGMainDisplayID()
    private var termObserver: NSObjectProtocol?

    // MARK: - 진입/이탈

    func enable(initialGlobalCG: CGPoint, on screen: NSScreen?, useNativeCursor: Bool) {
        guard !active else { return }
        virtualCG = initialGlobalCG
        suspended = false
        dragFrameNS = nil
        self.useNativeCursor = useNativeCursor
        hiddenDisplay = screen.flatMap {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
        } ?? CGMainDisplayID()

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

        // 링 모드에서만 진짜 커서를 숨긴다. 디커플은 안 함 — 재타깃이 진짜 커서를 소스 지점으로
        // 실제 이동시키므로 충돌 후속이 없다(디커플하면 얼어붙은 위치의 hover가 새어듦, 실측).
        // 소스가 활성이라 MacFG는 백그라운드 → 백그라운드 커서 제어를 먼저 켜야 숨김이 먹힌다.
        if !useNativeCursor {
            allowBackgroundCursorControl()
            CGDisplayHideCursor(hiddenDisplay)
        }

        // 최후 안전망 — 앱 종료 시 확실히 복구. self 대신 display 값만 캡처(Sendable).
        let display = hiddenDisplay
        termObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { _ in
            CGAssociateMouseAndMouseCursorPosition(1)
            CGDisplayShowCursor(display)
        }

        active = true
        if !useNativeCursor { onSyntheticCursor(virtualCG) }
        DiagnosticLog.shared.log("[RELPTR] enabled cg(\(Int(initialGlobalCG.x)),\(Int(initialGlobalCG.y))) display=\(hiddenDisplay) native=\(useNativeCursor)")
    }

    func disable() {
        guard active else { return }
        active = false
        suspended = false
        dragFrameNS = nil
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

    // MARK: - 탈출/복귀 (멀티모니터)

    /// p를 포함하는 디스플레이 (없으면 nil — 화면 밖 데드존)
    private func displayContaining(_ p: CGPoint) -> CGDirectDisplayID? {
        var id: CGDirectDisplayID = 0
        var count: UInt32 = 0
        guard CGGetDisplaysWithPoint(p, 1, &id, &count) == .success, count > 0 else { return nil }
        return id
    }

    /// 탈출 — 진짜 커서 표시(링 모드)만. 커서 이동은 호출부가 event.location으로 처리(워프 없이
    /// 연속). CGWarpMouseCursorPosition은 HID 재동기로 순간 입력 정지('턱')를 유발해 쓰지 않는다.
    private func suspendToNeighbor() {
        suspended = true
        dragFrameNS = nil
        if !useNativeCursor { CGDisplayShowCursor(hiddenDisplay) }
        onSuspendChange(true)
        DiagnosticLog.shared.log("[RELPTR] → 옆 모니터로 탈출 — 무변조 통과")
    }

    private func resumeFromNeighbor(at p: CGPoint) {
        suspended = false
        virtualCG = p
        if !useNativeCursor {
            CGDisplayHideCursor(hiddenDisplay)
            onSyntheticCursor(virtualCG)
        }
        onSuspendChange(false)                  // 링 표시(링 모드) + 소스 재활성
        DiagnosticLog.shared.log("[RELPTR] ← 뷰어 화면 복귀 cg(\(Int(p.x)),\(Int(p.y))) — 재캡처")
    }

    // MARK: - 탭 처리

    /// C 콜백에서 호출. 원본 이벤트의 location만 소스 좌표로 재타깃해 통과시키거나(레터박스 안),
    /// 소비하거나(블랙바), 무변조 통과시킨다(탈출 중).
    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        let screen = CGDisplayBounds(viewerDisplayID())

        // 탈출 중: 커서가 뷰어 화면으로 돌아오면 재캡처, 아니면 손대지 않고 통과
        if suspended {
            let loc = event.location
            if screen.insetBy(dx: 2, dy: 2).contains(loc) {
                resumeFromNeighbor(at: loc)
                return nil   // 복귀 이동 자체는 소비 (진짜 커서 잔상 방지)
            }
            return Unmanaged.passUnretained(event)
        }

        // 이동/드래그: delta 누적 (CG top-left: dy 아래 양수 = y 증가 방향 일치)
        switch type {
        case .mouseMoved, .leftMouseDragged, .rightMouseDragged, .otherMouseDragged:
            let dx = event.getDoubleValueField(.mouseEventDeltaX)
            let dy = event.getDoubleValueField(.mouseEventDeltaY)
            let raw = CGPoint(x: virtualCG.x + dx, y: virtualCG.y + dy)
            if !screen.contains(raw), dragFrameNS == nil,
               let other = displayContaining(raw), other != viewerDisplayID() {
                // 인접 모니터로 탈출 (드래그 중엔 유지). 이벤트 자체로 커서를 탈출점에 옮겨
                // 워프 스톨 없이 옆 화면으로 자연스럽게 이어지게 한다.
                suspendToNeighbor()
                event.location = raw
                return Unmanaged.passUnretained(event)
            }
            virtualCG.x = min(max(raw.x, screen.minX), screen.maxX - 1)
            virtualCG.y = min(max(raw.y, screen.minY), screen.maxY - 1)
            if !useNativeCursor { onSyntheticCursor(virtualCG) }
        case .leftMouseDown, .rightMouseDown, .otherMouseDown:
            dragFrameNS = currentSourceFrameNS()   // 드래그 매핑 고정 (창 폭주 방지)
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
        let mapped = mapVirtualToSource(virtualCG, dragFrameNS, clickLike)
        // up은 고정 프레임으로 매핑까지 마친 뒤 해제
        switch type {
        case .leftMouseUp, .rightMouseUp, .otherMouseUp: dragFrameNS = nil
        default: break
        }
        guard let mapped else { return nil }    // 레터박스 밖(블랙바) — 소비
        event.location = mapped                 // 재타깃 (새 이벤트 없음 → 피드백 없음)
        // delta 제거 — 큰 delta는 윈도우서버가 중간 move를 합성해 다중 배달(실측)
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
