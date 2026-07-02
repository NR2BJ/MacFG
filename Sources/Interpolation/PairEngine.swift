@preconcurrency import Metal
import FramePacing

/// encodePair 결과 — 요청한 t별 보간 텍스처
public struct PairEncodeResult {
    /// (위상 t, 텍스처) — t 오름차순. 엔진이 일부 t만 지원하면 지원분만 반환.
    public let frames: [(t: Float, texture: any MTLTexture)]
    /// command buffer 완료 후 호출 가능 — true면 장면 전환 (모든 보간 프레임 폐기 권장). nil이면 판정 없음.
    public let sceneCutEvaluator: (@Sendable () -> Bool)?

    public init(frames: [(t: Float, texture: any MTLTexture)], sceneCutEvaluator: (@Sendable () -> Bool)? = nil) {
        self.frames = frames
        self.sceneCutEvaluator = sceneCutEvaluator
    }
}

/// 연속 프레임 쌍 (A→B) 에서 중간 프레임(들)을 생성하는 엔진.
///
/// FrameInterpolator(순수 텍스처 함수)와 달리 세션/타임스탬프 상태를 가질 수 있다
/// (VTLowLatencyFrameInterpolation은 세션 기반 + 단조 증가 타임스탬프 요구).
/// encodePair는 매 연속 쌍마다 호출되어야 하며, 불연속(캡처 재시작/모드 전환)시 reset() 필수.
///
/// tValues: 소스 프레임 드랍으로 갭이 크면 스케줄러가 여러 위상을 요청한다
/// (예: 33ms 갭 @120Hz → [0.25, 0.5, 0.75]). 임의 t 미지원 엔진은 지원분만 반환.
public protocol PairInterpolationEngine: AnyObject {
    var name: String { get }

    func prepare(device: any MTLDevice) async throws

    /// stableA/stableB: SCK 재활용에서 분리된 안정 텍스처 (BGRA, 동일 크기).
    /// 반환 프레임들은 GPU/ANE 작업이 commandBuffer 완료 후 유효.
    /// nil 또는 빈 frames = 이 쌍은 보간 불가 — 호출자는 소스만 출력.
    func encodePair(
        stableA: any MTLTexture,
        stableB: any MTLTexture,
        tsA: CFTimeInterval,
        tsB: CFTimeInterval,
        tValues: [Float],
        into commandBuffer: any MTLCommandBuffer
    ) -> PairEncodeResult?

    /// 프레임 연속성 단절 시 호출 (캡처 재시작 등)
    func reset()

    func shutdown()
}

/// 기존 FrameInterpolator(Blend 등)를 PairInterpolationEngine으로 감싸는 어댑터
public final class LegacyPairEngine: PairInterpolationEngine {
    private let inner: any FrameInterpolator
    public var name: String { inner.name }

    public init(_ inner: any FrameInterpolator) {
        self.inner = inner
    }

    public func prepare(device: any MTLDevice) async throws {
        try await inner.prepare(device: device)
    }

    public func encodePair(
        stableA: any MTLTexture,
        stableB: any MTLTexture,
        tsA: CFTimeInterval,
        tsB: CFTimeInterval,
        tValues: [Float],
        into commandBuffer: any MTLCommandBuffer
    ) -> PairEncodeResult? {
        guard let texes = try? inner.batchInterpolate(frameA: stableA, frameB: stableB, tValues: tValues, commandBuffer: commandBuffer),
              texes.count == tValues.count else {
            return nil
        }
        return PairEncodeResult(frames: Array(zip(tValues, texes)))
    }

    public func reset() {}

    public func shutdown() {
        inner.shutdown()
    }
}
