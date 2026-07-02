import Metal
import QuartzCore

/// 디스플레이 동기화 프로토콜
public protocol DisplaySync: AnyObject, Sendable {
    /// VSync 콜백 시작
    func start(callback: @escaping @Sendable (CFTimeInterval, CFTimeInterval) -> Void)
    /// VSync 콜백 중지
    func stop()
    /// 현재 디스플레이 주사율
    var refreshRate: Double { get }
}
