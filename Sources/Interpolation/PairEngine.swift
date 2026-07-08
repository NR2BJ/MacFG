@preconcurrency import Metal
import FramePacing

/// encodePair 결과 — 요청한 t별 보간 텍스처
public struct PairEncodeResult {
    /// (위상 t, 텍스처, stamp) — t 오름차순. 엔진이 일부 t만 지원하면 지원분만 반환.
    /// stamp: 이 텍스처가 엔진 출력 링에서 차지한 슬롯의 세대 도장 (0 = 링 재사용 없음/검증 불요).
    /// 표시 직전 isFrameLive(stamp)로 "후속 warp가 이 슬롯을 덮지 않았나" 확인 — burst에서
    /// 표시 대기 텍스처가 덮여도 잘못된 픽셀 대신 깨끗한 드롭(직전 프레임 유지)이 되게 한다.
    public let frames: [(t: Float, texture: any MTLTexture, stamp: UInt64)]
    /// command buffer 완료 후 호출 가능 — true면 장면 전환 (모든 보간 프레임 폐기 권장). nil이면 판정 없음.
    public let sceneCutEvaluator: (@Sendable () -> Bool)?

    public init(frames: [(t: Float, texture: any MTLTexture, stamp: UInt64)], sceneCutEvaluator: (@Sendable () -> Bool)? = nil) {
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

    /// stamp가 가리키는 출력 텍스처가 아직 유효한가 (후속 encodePair가 그 링 슬롯을 재사용해
    /// 덮지 않았는가). 표시 경로가 렌더 스레드에서 호출 — encodePair와 같은 스레드라 락 불요.
    /// 링을 재사용하지 않는 엔진(Legacy 등)은 기본 구현(항상 true)을 그대로 쓴다.
    func isFrameLive(_ stamp: UInt64) -> Bool
}

public extension PairInterpolationEngine {
    func isFrameLive(_ stamp: UInt64) -> Bool { true }
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
        // Blend는 매 호출 새 텍스처(링 재사용 없음) → stamp 0 = 검증 불요 (isFrameLive 기본 true)
        return PairEncodeResult(frames: zip(tValues, texes).map { (t: $0, texture: $1, stamp: UInt64(0)) })
    }

    public func reset() {}

    public func shutdown() {
        inner.shutdown()
    }
}
