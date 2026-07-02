import Foundation

/// 프레임 보간 배수 모드
public enum MultiplierMode: Sendable, Equatable {
    /// 고정 배수 (x2, x3, x4)
    case fixed(Int)
    /// 목표 FPS — 입력 FPS로부터 배수 자동 계산
    case targetFPS(Double)

    /// 현재 소스 프레임 간격에 삽입할 보간 프레임의 t 값들을 반환.
    /// 예: x2 → [0.5], x3 → [0.33, 0.67], x4 → [0.25, 0.5, 0.75]
    public func interpolationPoints(inputFPS: Double, displayRefreshRate: Double) -> [Float] {
        let multiplier: Int
        switch self {
        case .fixed(let m):
            multiplier = max(1, min(m, 4))
        case .targetFPS(let target):
            let raw = target / max(inputFPS, 1.0)
            multiplier = max(1, min(Int(raw.rounded()), 4))
        }

        guard multiplier > 1 else { return [] }
        return (1..<multiplier).map { Float($0) / Float(multiplier) }
    }

    /// 현재 실효 배수
    public func effectiveMultiplier(inputFPS: Double) -> Int {
        switch self {
        case .fixed(let m):
            return max(1, min(m, 4))
        case .targetFPS(let target):
            let raw = target / max(inputFPS, 1.0)
            return max(1, min(Int(raw.rounded()), 4))
        }
    }
}
