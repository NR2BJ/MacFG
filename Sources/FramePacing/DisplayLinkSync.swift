import AppKit
@preconcurrency import Metal
import QuartzCore
import os

/// CADisplayLink + MTLSharedEvent 기반 VSync 동기화
@MainActor
public final class DisplayLinkSync: @unchecked Sendable {
    private let device: any MTLDevice
    private let sharedEvent: any MTLSharedEvent
    private var displayLink: CADisplayLink?
    private var callback: (@Sendable (CFTimeInterval, CFTimeInterval) -> Void)?
    private var signalValue: UInt64 = 0
    private let logger = Logger(subsystem: "com.macfg", category: "DisplaySync")

    private weak var boundScreen: NSScreen?

    public var refreshRate: Double {
        // 디스플레이 모드의 고정 주사율 우선 — 링크 틱 간격은 히컵 시 절반/제3값으로 읽혀
        // "콘텐츠가 이미 빠름" 오판정으로 보간을 억제한 사례 있음 (2026-07-02)
        if let screen = boundScreen ?? NSScreen.main {
            let maxFPS = screen.maximumFramesPerSecond
            if maxFPS > 0 { return Double(maxFPS) }
        }
        if let link = displayLink, link.targetTimestamp > link.timestamp {
            let interval = link.targetTimestamp - link.timestamp
            if interval > 0 {
                return 1.0 / interval
            }
        }
        return 60.0
    }

    public init(device: any MTLDevice) {
        self.device = device
        self.sharedEvent = device.makeSharedEvent()!
    }

    /// screen: 출력 창이 위치한 화면 (nil이면 메인) — 페이싱은 출력 화면의 vsync를 따라야 한다
    public func start(screen: NSScreen? = nil, callback: @escaping @Sendable (CFTimeInterval, CFTimeInterval) -> Void) {
        self.callback = callback

        guard let target = screen ?? NSScreen.main else {
            logger.error("No screen available")
            return
        }
        self.boundScreen = target

        let link = target.displayLink(target: self, selector: #selector(displayLinkFired(_:)))
        // .common: 메뉴 추적/창 드래그 중에도 틱 유지 (.default만 쓰면 UI 조작 시 출력이 멈춘다)
        link.add(to: .main, forMode: .common)
        self.displayLink = link

        logger.info("DisplayLink started (\(target.localizedName), \(target.maximumFramesPerSecond)Hz)")
    }

    public func stop() {
        displayLink?.invalidate()
        displayLink = nil
        callback = nil
        logger.info("DisplayLink stopped")
    }

    @objc private func displayLinkFired(_ link: CADisplayLink) {
        callback?(link.timestamp, link.targetTimestamp)
    }

    /// GPU가 이전 프레임을 완료했는지 (non-blocking 폴링, 딜레이 0)
    /// MTLSharedEvent.signaledValue는 아토믹 읽기 — MainThread 블로킹 없음.
    public var gpuReady: Bool {
        sharedEvent.signaledValue >= signalValue
    }

    /// GPU 렌더링 완료 시그널을 커맨드 버퍼에 인코딩
    public func encodeSignal(to commandBuffer: any MTLCommandBuffer) {
        signalValue += 1
        commandBuffer.encodeSignalEvent(sharedEvent, value: signalValue)
    }

    deinit {
        // DisplayLink is invalidated in stop()
    }
}
