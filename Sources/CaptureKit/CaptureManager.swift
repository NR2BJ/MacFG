import Metal
import CoreGraphics
import FramePacing
import os

/// ScreenCaptureKit → IOSurface 자동 전환 캡처 관리자
public final class CaptureManager: Sendable {
    private let ioSurfaceCapture = IOSurfaceCapture()
    private let sckCapture = SCKCapture()
    private let logger = Logger(subsystem: "com.macfg", category: "CaptureManager")

    // nonisolated(unsafe) for mutable state — protected by actor-like usage pattern
    nonisolated(unsafe) private var activeSource: (any FrameSource)?
    nonisolated(unsafe) private var _activeMethod: CaptureMethod = .screenCaptureKit

    public var activeMethod: CaptureMethod { _activeMethod }

    public init() {}

    /// 캡처 시작. SCK 우선, 실패 시 IOSurface 폴백.
    /// SCK는 frame status/fingerprint를 제공해 frame generation 타임라인이 더 안정적이다.
    public func startCapture(windowID: CGWindowID, device: any MTLDevice) async throws {
        // ScreenCaptureKit 먼저 시도
        do {
            try await sckCapture.startCapture(windowID: windowID, device: device)
            activeSource = sckCapture
            _activeMethod = .screenCaptureKit
            logger.info("Using ScreenCaptureKit capture")
            return
        } catch {
            logger.info("ScreenCaptureKit failed, falling back to IOSurface: \(error)")
        }

        // IOSurface 폴백
        do {
            try await ioSurfaceCapture.startCapture(windowID: windowID, device: device)
            activeSource = ioSurfaceCapture
            _activeMethod = .ioSurface
            logger.info("Using IOSurface capture")
        } catch {
            logger.error("All capture methods failed: \(error)")
            throw error
        }
    }

    /// 캡처 중지
    public func stopCapture() async {
        await activeSource?.stopCapture()
        activeSource = nil
    }

    /// 최신 프레임 가져오기
    public func latestFrame() -> FrameSlot? {
        activeSource?.latestFrame()
    }

    /// 마지막 drain 이후 도착한 프레임들 (타임스탬프 순)
    public func drainFrames() -> [FrameSlot] {
        activeSource?.drainFrames() ?? []
    }

    /// SCK로 수동 전환 (IOSurface 문제 감지 시)
    public func fallbackToSCK(windowID: CGWindowID, device: any MTLDevice) async throws {
        await ioSurfaceCapture.stopCapture()
        try await sckCapture.startCapture(windowID: windowID, device: device)
        activeSource = sckCapture
        _activeMethod = .screenCaptureKit
        logger.info("Manually switched to ScreenCaptureKit")
    }
}
