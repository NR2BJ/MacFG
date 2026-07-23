import AppKit
import Metal
import CoreGraphics
import FramePacing
import os

public enum OverlayPlacement: String, CaseIterable, Identifiable, Sendable {
    case coverSource
    case viewerWindow

    public static let allCases: [OverlayPlacement] = [.coverSource, .viewerWindow]
    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .coverSource: "Cover Source"
        case .viewerWindow: "Separate Window"
        }
    }
}

/// 오버레이 출력 총괄 관리자
@MainActor
public final class OverlayManager {
    private let logger = Logger(subsystem: "com.macfg", category: "OverlayManager")

    private var overlayWindow: OverlayWindow?
    private let windowTracker = WindowTracker()
    private let device: any MTLDevice
    private var lastAppliedFrame: CGRect = .zero
    private var placement: OverlayPlacement = .coverSource
    private var upscaleMode: UpscaleMode = .off
    private var sharpness: Float = 0
    private var trackedWindowID: CGWindowID?
    private var captureColorSpace: CGColorSpace?
    /// 소스 창의 NS 좌표 프레임 (windowTracker.pollGeometry 결과)
    private var lastSourceFrame: CGRect = .zero

    /// 연속 창 추적 실패 횟수 — 대상 창 종료 감지용 (성공 시 0으로 리셋)
    public private(set) var trackingFailureCount: Int = 0

    /// 소스 창이 화면에 보이는지 — 최소화/다른 Space면 false (좀비 Cover 오버레이 방지용)
    public var sourceIsOnScreen: Bool { windowTracker.windowIsOnScreen }

    /// MacFG 자신이 띄운 창들의 CGWindowID — 디스플레이 캡처 제외 목록용.
    /// 오버레이/뷰어뿐 아니라 설정 창 등 이 앱의 모든 창을 포함해야 되먹임이 없다.
    public var ownWindowIDs: [CGWindowID] {
        // windowNumber는 Int이고 오프스크린/미실현 창은 음수일 수 있다 — UInt32 변환이
        // 트랩을 내므로 compactMap으로 안전하게 거른다 (음수를 filter로 막았어도 map에서
        // 다시 변환하면 경계 사례에서 터진다, 실측 크래시).
        var ids = NSApplication.shared.windows.compactMap { w -> CGWindowID? in
            let n = w.windowNumber
            guard n > 0, n <= Int(UInt32.max) else { return nil }
            return CGWindowID(n)
        }
        if let ow = overlayWindow?.cgWindowID, ow > 0, !ids.contains(ow) { ids.append(ow) }
        return ids
    }

    /// 소스 창의 현재 픽셀 크기 (points × 해당 화면 배율) — 캡처 해상도와 비교해 리사이즈 감지용
    public var sourcePixelSize: (width: Int, height: Int)? {
        guard lastSourceFrame.width > 1, lastSourceFrame.height > 1 else { return nil }
        let screen = NSScreen.screens.first {
            $0.frame.contains(CGPoint(x: lastSourceFrame.midX, y: lastSourceFrame.midY))
        } ?? NSScreen.main
        let scale = screen?.backingScaleFactor ?? 2.0
        return (Int(lastSourceFrame.width * scale), Int(lastSourceFrame.height * scale))
    }

    /// 소스 창이 현재 위치한 화면 (전체화면 자동 뷰어 판정용)
    public var sourceScreen: NSScreen? {
        NSScreen.screens.first { $0.frame.contains(CGPoint(x: lastSourceFrame.midX, y: lastSourceFrame.midY)) }
    }

    /// 소스 창이 자기 화면을 거의(≥95%) 채우면 true — 전체화면 자동 뷰어 전환 판정용.
    /// Cover는 소스 전체화면을 못 덮으므로(glass=0), 전체화면이면 뷰어(자기 창)로 가야 합성된다.
    /// 소스가 **진짜 전체화면**(자체 Space)인지 — Cover 오버레이가 합성되지 않는 조건.
    /// 최대화 창과 반드시 구분해야 한다: 최대화는 메뉴바 영역을 비워두므로 화면 높이보다
    /// 메뉴바만큼 작다(4K 2160pt에서 ~2135pt = 98.8%). 옛 "95% 이상" 임계는 둘을 못 갈라
    /// 최대화만 해도 뷰어로 자동 전환돼 Cover 모드를 쓸 수 없었다.
    public var sourceIsFullscreen: Bool {
        guard lastSourceFrame.width > 1, let scr = sourceScreen else { return false }
        return lastSourceFrame.width >= scr.frame.width - 2
            && lastSourceFrame.height >= scr.frame.height - 2
    }

    /// 뷰어 창을 사용자가 닫았을 때 (캡처 정지 트리거)
    public var onViewerClosed: (() -> Void)?
    /// 출력 창이 다른 화면으로 이동했을 때 (DisplayLink 재바인딩 트리거)
    public var onOutputScreenChanged: ((NSScreen?) -> Void)?
    private weak var lastOutputScreen: NSScreen?

    /// 출력 창이 현재 위치한 화면
    public var outputScreen: NSScreen? { overlayWindow?.currentScreen }

    public var trackingMethod: String { windowTracker.currentMethod.rawValue }

    public init(device: any MTLDevice) {
        self.device = device
    }

    /// 오버레이 시작 — 창 추적 + 출력 창 생성
    public func start(windowID: CGWindowID) throws {
        trackedWindowID = windowID
        windowTracker.startTracking(windowID: windowID)
        try createOutputWindow()
        logger.info("Overlay started for window \(windowID)")
    }

    /// 오버레이 중지
    public func stop() {
        windowTracker.stopTracking()
        overlayWindow?.close()
        overlayWindow = nil
        trackedWindowID = nil
        lastAppliedFrame = .zero
        lastSourceFrame = .zero
        trackingFailureCount = 0
        logger.info("Overlay stopped")
    }

    /// 소스 앱 PID — 뷰어 마우스 역매핑에서 CGEventPostToPid 대상. 창 재생성에도 유지.
    public var sourcePID: pid_t = 0 {
        didSet { overlayWindow?.sourcePID = sourcePID }
    }

    /// 업스케일 방식 — 현재 창에 즉시 적용 + 이후 창 재생성에도 유지
    public func setUpscaleMode(_ mode: UpscaleMode) {
        upscaleMode = mode
        overlayWindow?.upscaleMode = mode
        applyUpscaleGate()
    }

    /// 부하 거버너의 MetalFX 게이트 — false면 GPU를 쓰는 MetalFX 단계만 끈다.
    /// ANE 2x는 GPU-free(1.68ms 실측)라 유지해 체감 손실을 최소화한다.
    private var allowsMetalFX = true
    public func setUpscaleAllowsMetalFX(_ allow: Bool) {
        guard allowsMetalFX != allow else { return }
        allowsMetalFX = allow
        applyUpscaleGate()
    }

    private func applyUpscaleGate() {
        let effective: UpscaleMode
        if allowsMetalFX {
            effective = upscaleMode
        } else {
            switch upscaleMode {
            case .metalfx:     effective = .off   // MetalFX 단독 → 끔
            case .aneMetalfx:  effective = .ane   // 체인에서 MetalFX만 제거
            default:           effective = upscaleMode
            }
        }
        overlayWindow?.upscaleMode = effective
    }

    /// CAS 샤프닝 강도 (0=끔) — 업스케일과 독립, 창 재생성에도 유지
    public func setSharpness(_ value: Float) {
        sharpness = value
        overlayWindow?.sharpness = value
    }

    /// 업스케일 실동작 상태 (UI 표시용)
    public var scaleStatus: String? { overlayWindow?.scaleStatus }

    /// 정보 오버레이 (단축키 토글) 표시/갱신 — nil이면 숨김.
    public func setInfoOverlay(_ text: String?) { overlayWindow?.setInfoOverlay(text) }

    /// 현재 출력 창의 렌더 표면 (렌더 스레드가 직접 사용 — A2)
    public var currentRenderSurface: RenderSurface? { overlayWindow?.surface }

    /// 대상 창이 아직 살아있는지 (닫힘 감지)
    public var sourceWindowExists: Bool { windowTracker.windowExists }

    /// 소스 창을 target 픽셀 크기로 리사이즈 (해상도 프리셋 — 내부에서 point 변환). 성공 여부.
    @discardableResult
    public func resizeSourceWindow(toPixelWidth w: Int, height h: Int) -> Bool {
        guard lastSourceFrame.width > 1 else { return false }
        let screen = NSScreen.screens.first {
            $0.frame.contains(CGPoint(x: lastSourceFrame.midX, y: lastSourceFrame.midY))
        } ?? NSScreen.main
        let scale = screen?.backingScaleFactor ?? 2.0
        return windowTracker.resizeTrackedWindow(toPoints: CGSize(width: CGFloat(w) / scale, height: CGFloat(h) / scale))
    }

    public func setPlacement(_ newPlacement: OverlayPlacement) {
        let changed = placement != newPlacement
        placement = newPlacement
        lastAppliedFrame = .zero
        if changed, trackedWindowID != nil, overlayWindow != nil {
            // 캡처 중 배치 전환 — 출력 창 재생성
            overlayWindow?.setVisible(false)
            overlayWindow = nil
            try? createOutputWindow()
        } else {
            applyOcclusionPolicy()
            pollAndUpdateFrame()
        }
    }

    private func createOutputWindow() throws {
        let style: OverlayStyle = placement == .coverSource ? .overlay : .viewer
        let overlay = try OverlayWindow(device: device, style: style)
        overlay.onUserClose = { [weak self] in
            self?.onViewerClosed?()
        }
        overlay.upscaleMode = upscaleMode
        overlay.sharpness = sharpness
        overlay.sourcePID = sourcePID
        overlay.sourceWindowID = trackedWindowID ?? 0
        self.overlayWindow = overlay

        applyOcclusionPolicy()
        pollAndUpdateFrame()

        if style == .viewer {
            overlay.setInitialViewerFrame(initialViewerFrame(sourceFrame: lastSourceFrame))
        }
        applyColorPolicy()
        overlay.setVisible(true)
    }

    /// 캡처 색공간 기록 — 실제 적용은 화면 일치 여부와 함께 결정
    public func setCaptureColorSpace(_ colorSpace: CGColorSpace?) {
        captureColorSpace = colorSpace
        applyColorPolicy()
    }

    /// Cover든 전체화면 뷰어든 대상 창을 완전히 덮으므로 오클루전 우회(alpha 0.999) 필요 —
    /// 안 하면 소스가 occluded로 마킹돼 Firefox PiP 등 가려진 창이 렌더를 멈춰 화면이 정지한다(실측).
    private func applyOcclusionPolicy() {
        overlayWindow?.setOcclusionBypass(true)
    }

    /// 소스 창과 출력이 같은 디스플레이면 passthrough(무변환), 다르면 캡처 태그 적용
    private func applyColorPolicy() {
        guard let overlayWindow else { return }
        let sourceScreen = NSScreen.screens.first {
            $0.frame.contains(CGPoint(x: lastSourceFrame.midX, y: lastSourceFrame.midY))
        }
        let overlayScreen = overlayWindow.currentScreen
        let same = sourceScreen == nil || overlayScreen == nil || sourceScreen == overlayScreen
        overlayWindow.setColorSpace(captureColorSpace, sameDisplayAsSource: same)

        // 출력 화면이 바뀌면 페이싱도 그 화면 vsync로 재바인딩 필요 + 렌더 표면 배율 갱신
        if overlayScreen !== lastOutputScreen {
            lastOutputScreen = overlayScreen
            overlayWindow.refreshSurfaceParams()
            onOutputScreenChanged?(overlayScreen)
        }
    }

    /// 외부 CB에 렌더 인코딩. drawable 반환.
    /// 주의: 창 추적 폴링은 updateTracking(30Hz 타이머)에서만 — CGWindowList가 0.5-2ms라
    /// present 경로에 두면 vsync 틱을 놓친다 (틱 유실 ~5%/s 실측).
    public func encodeRenderFrame(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer) -> (any CAMetalDrawable)? {
        overlayWindow?.encodeRender(texture: texture, into: commandBuffer)
    }

    /// 렌더 없이 위치 추적만 갱신 (프레임 스킵 시 호출)
    public func updateTracking() {
        pollAndUpdateFrame()
        overlayWindow?.logViewerGeometryIfChanged()   // 창이 나중에 옮겨지는 경우 포착
    }

    /// Cover 오버레이 숨김/표시 (창은 유지 — 자동 숨김/수동 토글용).
    /// 표시 복귀 시 occlusion 우회·색 정책·위치를 다시 적용한다.
    /// 뷰어 배치는 사용자가 직접 제어하므로 무시.
    /// 화면 구성 변경 시 뷰어가 화면 밖에 남지 않도록 보정
    public func ensureViewerOnScreen() {
        overlayWindow?.ensureViewerOnScreen()
    }

    public func setOverlayHidden(_ hidden: Bool) {
        guard let overlayWindow, placement == .coverSource else { return }
        if hidden {
            overlayWindow.setVisible(false)
        } else {
            applyOcclusionPolicy()
            pollAndUpdateFrame()
            applyColorPolicy()
            overlayWindow.setVisible(true)
        }
    }

    /// CGWindowList에서 최신 위치를 읽고, cover 배치면 오버레이 위치 갱신
    private func pollAndUpdateFrame() {
        guard let geom = windowTracker.pollGeometry() else {
            trackingFailureCount += 1
            return
        }
        trackingFailureCount = 0
        overlayWindow?.sourceFrameNS = geom.frame   // 마우스 역매핑용 — 소스 위치 최신 유지

        if placement == .coverSource {
            // 위치/크기가 실제로 바뀌었을 때만 setFrame 호출 (윈도우 서버 부하 최소화)
            if geom.frame != lastAppliedFrame {
                lastAppliedFrame = geom.frame
                overlayWindow?.updateFrame(geom.frame)
            }
        }

        // 소스 창이 다른 화면으로 이동하면 색 정책 갱신
        if geom.frame != lastSourceFrame {
            lastSourceFrame = geom.frame
            applyColorPolicy()
        }
    }

    /// 뷰어 초기 배치: 소스 창이 있는 화면 전체(메뉴바·Dock 포함)를 덮는다 — 초록버튼 전체화면 느낌.
    /// visibleFrame이 아니라 frame이라 Dock/메뉴바 영역까지 채우고, 창은 shielding 레벨로 그 위에 뜬다.
    /// 영상은 뷰어 안에서 종횡비 유지 레터박스로 표시됨.
    private func initialViewerFrame(sourceFrame: CGRect) -> CGRect {
        let screens = NSScreen.screens
        let sourceScreen = screens.first(where: { $0.frame.contains(CGPoint(x: sourceFrame.midX, y: sourceFrame.midY)) })
            ?? NSScreen.main
        return sourceScreen?.frame ?? CGRect(x: 0, y: 0, width: 1600, height: 900)
    }

    private func aspectFitRect(aspect: CGFloat, inside rect: CGRect) -> CGRect {
        guard rect.width > 0, rect.height > 0 else { return .zero }
        var width = rect.width
        var height = width / aspect
        if height > rect.height {
            height = rect.height
            width = height * aspect
        }
        return CGRect(
            x: rect.midX - width / 2,
            y: rect.midY - height / 2,
            width: width,
            height: height
        )
    }
}
