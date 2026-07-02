import QuartzCore
import os

/// 성능 모니터링 — FPS 및 레이턴시 측정
public final class PerformanceMonitor: Sendable {
    private let logger = Logger(subsystem: "com.macfg", category: "Performance")

    private struct State: Sendable {
        var frameTimestamps: [CFTimeInterval] = []
        var renderTimestamps: [CFTimeInterval] = []
        var latencySamples: [Double] = [] // 프레임별 캡처→렌더 레이턴시 (ms)
        var currentFrameStart: CFTimeInterval = 0 // 현재 프레임의 시작 시점
        var maxSamples: Int = 120 // 최근 120프레임
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    public init() {}

    /// 프레임 도착 기록
    public func recordFrameArrival() {
        let now = CACurrentMediaTime()
        state.withLock { s in
            s.frameTimestamps.append(now)
            if s.frameTimestamps.count > s.maxSamples {
                s.frameTimestamps.removeFirst()
            }
        }
    }

    /// 프레임 처리 시작 시점 기록 (DisplayLink 틱 시작)
    public func recordCaptureTime() {
        let now = CACurrentMediaTime()
        state.withLock { s in
            s.currentFrameStart = now
        }
    }

    /// 렌더링 완료 시점 기록 + 레이턴시 계산
    public func recordRenderTime() {
        let now = CACurrentMediaTime()
        state.withLock { s in
            s.renderTimestamps.append(now)
            if s.renderTimestamps.count > s.maxSamples {
                s.renderTimestamps.removeFirst()
            }
            // 같은 프레임의 시작~완료 레이턴시
            if s.currentFrameStart > 0 {
                let latencyMs = (now - s.currentFrameStart) * 1000.0
                s.latencySamples.append(latencyMs)
                if s.latencySamples.count > s.maxSamples {
                    s.latencySamples.removeFirst()
                }
            }
        }
    }

    /// 입력 FPS 계산
    public var inputFPS: Double {
        state.withLock { s in
            calculateFPS(from: s.frameTimestamps)
        }
    }

    /// 출력 FPS 계산
    public var outputFPS: Double {
        state.withLock { s in
            calculateFPS(from: s.renderTimestamps)
        }
    }

    /// 캡처→출력 평균 레이턴시 (ms)
    public var averageLatencyMs: Double {
        state.withLock { s in
            guard !s.latencySamples.isEmpty else { return 0 }
            let recent = s.latencySamples.suffix(60)
            return recent.reduce(0, +) / Double(recent.count)
        }
    }

    /// 콘솔에 현재 상태 로깅
    public func logStats() {
        let inFPS = inputFPS
        let outFPS = outputFPS
        let latency = averageLatencyMs
        logger.info("Input: \(String(format: "%.1f", inFPS)) FPS | Output: \(String(format: "%.1f", outFPS)) FPS | Latency: \(String(format: "%.2f", latency)) ms")
    }

    // MARK: - Private

    private func calculateFPS(from timestamps: [CFTimeInterval]) -> Double {
        guard timestamps.count >= 2 else { return 0 }
        let recent = timestamps.suffix(60)
        guard recent.count >= 2,
              let first = recent.first,
              let last = recent.last,
              last > first else { return 0 }
        return Double(recent.count - 1) / (last - first)
    }
}
