@preconcurrency import Metal

/// 프레임 보간 엔진 공통 프로토콜
/// 모든 메서드(prepare 제외)는 동기식 — CPU측 Metal 커맨드 인코딩만 수행.
/// GPU 실행은 commandBuffer.commit() 후 Metal이 처리.
public protocol FrameInterpolator: AnyObject, Sendable {
    var name: String { get }
    var isAvailable: Bool { get }

    /// 엔진 리소스 준비 (셰이더 컴파일 등). 1회 호출. async 유지.
    func prepare(device: any MTLDevice) async throws

    /// 프레임 A와 B 사이의 중간 프레임 생성 (동기, ~0.1ms CPU).
    /// commandBuffer에 compute 커맨드를 인코딩만 함. commit은 호출자가 담당.
    func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)?

    /// 여러 t값에 대해 배치 보간 (동기).
    /// 엔진이 배치 최적화(공유 downsample/blockmatch 등)를 지원하면 오버라이드.
    func batchInterpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        tValues: [Float],
        commandBuffer: any MTLCommandBuffer
    ) throws -> [any MTLTexture]

    func shutdown()
}

// 기본 구현: 개별 호출 폴백
extension FrameInterpolator {
    public func batchInterpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        tValues: [Float],
        commandBuffer: any MTLCommandBuffer
    ) throws -> [any MTLTexture] {
        var results: [any MTLTexture] = []
        for t in tValues {
            if let tex = try interpolate(frameA: frameA, frameB: frameB, t: t, commandBuffer: commandBuffer) {
                results.append(tex)
            }
        }
        return results
    }
}
