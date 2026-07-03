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

    /// 소스 창의 현재 픽셀 크기 (points × 해당 화면 배율) — 캡처 해상도와 비교해 리사이즈 감지용
    public var sourcePixelSize: (width: Int, height: Int)? {
        guard lastSourceFrame.width > 1, lastSourceFrame.height > 1 else { return nil }
        let screen = NSScreen.screens.first {
            $0.frame.contains(CGPoint(x: lastSourceFrame.midX, y: lastSourceFrame.midY))
        } ?? NSScreen.main
        let scale = screen?.backingScaleFactor ?? 2.0
        return (Int(lastSourceFrame.width * scale), Int(lastSourceFrame.height * scale))
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

    /// 업스케일 방식 — 현재 창에 즉시 적용 + 이후 창 재생성에도 유지
    public func setUpscaleMode(_ mode: UpscaleMode) {
        upscaleMode = mode
        overlayWindow?.upscaleMode = mode
    }

    /// CAS 샤프닝 강도 (0=끔) — 업스케일과 독립, 창 재생성에도 유지
    public func setSharpness(_ value: Float) {
        sharpness = value
        overlayWindow?.sharpness = value
    }

    /// 업스케일 실동작 상태 (UI 표시용)
    public var scaleStatus: String? { overlayWindow?.scaleStatus }

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

        // 출력 화면이 바뀌면 페이싱도 그 화면 vsync로 재바인딩 필요
        if overlayScreen !== lastOutputScreen {
            lastOutputScreen = overlayScreen
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
    }

    /// Cover 오버레이 숨김/표시 (창은 유지 — 자동 숨김/수동 토글용).
    /// 표시 복귀 시 occlusion 우회·색 정책·위치를 다시 적용한다.
    /// 뷰어 배치는 사용자가 직접 제어하므로 무시.
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
