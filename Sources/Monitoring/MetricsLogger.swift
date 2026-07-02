import Foundation
import os

/// 주기적 성능 로깅
public final class MetricsLogger: @unchecked Sendable {
    private let monitor: PerformanceMonitor
    private let logger = Logger(subsystem: "com.macfg", category: "Metrics")
    private var timer: Timer?
    private let interval: TimeInterval

    public init(monitor: PerformanceMonitor, interval: TimeInterval = 2.0) {
        self.monitor = monitor
        self.interval = interval
    }

    @MainActor
    public func start() {
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.monitor.logStats()
        }
        logger.info("Metrics logging started (interval: \(self.interval)s)")
    }

    public func stop() {
        timer?.invalidate()
        timer = nil
    }
}
