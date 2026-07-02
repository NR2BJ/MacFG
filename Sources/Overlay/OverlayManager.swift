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
    private var trackedWindowID: CGWindowID?
    private var captureColorSpace: CGColorSpace?
    /// 소스 창의 NS 좌표 프레임 (windowTracker.pollGeometry 결과)
    private var lastSourceFrame: CGRect = .zero

    /// 연속 창 추적 실패 횟수 — 대상 창 종료 감지용 (성공 시 0으로 리셋)
    public private(set) var trackingFailureCount: Int = 0

    /// 뷰어 창을 사용자가 닫았을 때 (캡처 정지 트리거)
    public var onViewerClosed: (() -> Void)?

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
        overlayWindow?.setVisible(false)
        overlayWindow = nil
        trackedWindowID = nil
        lastAppliedFrame = .zero
        lastSourceFrame = .zero
        trackingFailureCount = 0
        logger.info("Overlay stopped")
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

    /// Cover 배치는 대상 창을 완전히 덮으므로 오클루전 우회(alpha 0.999) 필요.
    /// 뷰어는 대상이 보이므로 불필요.
    private func applyOcclusionPolicy() {
        overlayWindow?.setOcclusionBypass(placement == .coverSource)
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
    }

    /// 외부 CB에 렌더 인코딩. drawable 반환.
    public func encodeRenderFrame(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer) -> (any CAMetalDrawable)? {
        pollAndUpdateFrame()
        return overlayWindow?.encodeRender(texture: texture, into: commandBuffer)
    }

    /// 렌더 없이 위치 추적만 갱신 (프레임 스킵 시 호출)
    public func updateTracking() {
        pollAndUpdateFrame()
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

    /// 뷰어 초기 배치: 다른 디스플레이가 있으면 거기에 크게, 없으면 소스 화면 우하단에 45% 크기
    private func initialViewerFrame(sourceFrame: CGRect) -> CGRect {
        let screens = NSScreen.screens
        let sourceScreen = screens.first(where: { $0.frame.contains(CGPoint(x: sourceFrame.midX, y: sourceFrame.midY)) })
            ?? NSScreen.main
        let aspect = sourceFrame.height > 0 ? max(sourceFrame.width / sourceFrame.height, 0.2) : 16.0 / 9.0

        if let sourceScreen, let other = screens.first(where: { $0 != sourceScreen }) {
            return aspectFitRect(aspect: aspect, inside: other.visibleFrame.insetBy(dx: 48, dy: 48))
        }

        let visible = (sourceScreen ?? NSScreen.main)?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1600, height: 900)
        let target = CGRect(
            x: visible.maxX - visible.width * 0.46 - 16,
            y: visible.minY + 16,
            width: visible.width * 0.46,
            height: visible.height * 0.46
        )
        return aspectFitRect(aspect: aspect, inside: target)
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
