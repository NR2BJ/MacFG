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

    /// SCK 스트림 비정상 중단(대상 창 닫힘) 시 즉시 콜백 — AppState가 stopCapture 트리거.
    nonisolated(unsafe) public var onStreamStopped: (@Sendable () -> Void)?

    /// 새 프레임 도착 즉시 콜백 (캡처 스레드) — 소비자가 렌더 틱을 기다리지 않고 인제스트하도록.
    nonisolated(unsafe) public var onFrameAvailable: (@Sendable () -> Void)?

    public init() {}

    /// 캡처 시작. SCK 우선, 실패 시 IOSurface 폴백.
    /// SCK는 frame status/fingerprint를 제공해 frame generation 타임라인이 더 안정적이다.
    public func startCapture(windowID: CGWindowID, device: any MTLDevice, captureRect: CGRect? = nil) async throws {
        // ScreenCaptureKit 먼저 시도 (영역 캡처는 SCK 전용)
        do {
            sckCapture.onStreamStopped = onStreamStopped
            sckCapture.onFrameAvailable = onFrameAvailable
            try await sckCapture.startCapture(windowID: windowID, device: device, captureRect: captureRect)
            activeSource = sckCapture
            _activeMethod = .screenCaptureKit
            logger.info("Using ScreenCaptureKit capture")
            return
        } catch {
            logger.info("ScreenCaptureKit failed, falling back to IOSurface: \(error)")
        }

        // IOSurface 폴백 (영역 캡처 미지원 — 전체 창)
        do {
            try await ioSurfaceCapture.startCapture(windowID: windowID, device: device, captureRect: nil)
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

    /// 스트림 무중단 리사이즈 (SCK 전용). IOSurface 폴백 시엔 미지원 → throw (호출자가 전체 재시작).
    public func updateConfiguration(width: Int, height: Int) async throws {
        guard _activeMethod == .screenCaptureKit else { throw CaptureError.notCapturing }
        try await sckCapture.updateConfiguration(width: width, height: height)
    }

    /// 캡처 대상 창 무중단 교체 (SCK 전용). 전체화면/PiP 새 창 재타깃 (U2).
    public func updateTargetWindow(windowID: CGWindowID) async throws {
        guard _activeMethod == .screenCaptureKit else { throw CaptureError.notCapturing }
        try await sckCapture.updateTargetWindow(windowID: windowID)
    }

    /// 디스플레이 캡처로 무중단 전환 (SCK 전용) — 소스가 자체 Space 전체화면일 때.
    public func updateToDisplayCapture(displayID: CGDirectDisplayID, excludingWindowIDs: [CGWindowID]) async throws {
        guard _activeMethod == .screenCaptureKit else { throw CaptureError.notCapturing }
        try await sckCapture.updateToDisplayCapture(displayID: displayID, excludingWindowIDs: excludingWindowIDs)
    }

    /// 현재 디스플레이 캡처 중인지 (SCK 전용, 그 외엔 false)
    public var isDisplayCapture: Bool {
        _activeMethod == .screenCaptureKit && sckCapture.isDisplayCapture
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
