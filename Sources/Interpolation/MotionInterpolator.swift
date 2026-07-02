@preconcurrency import Metal
import Monitoring
import os

/// GPU 블록 매칭 모션 추정 + 순방향 외삽 기반 프레임 생성 엔진
/// 전체 GPU 파이프라인 — 모션 추정(3 패스) + 외삽 워프(1 패스) = 4 compute passes
/// 외삽(extrapolation): frameA→B 모션으로 B 이후 시점을 예측 → 시간 역전 없음
public final class MotionInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "Motion (GPU Block Match)"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var motionEstimator: MotionEstimator?
    private var warpRenderer: WarpRenderer?
    private var sceneChangeDetector: SceneChangeDetector?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "MotionInterpolator")

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)

        let estimator = MotionEstimator(device: device)
        try await estimator.prepare()
        self.motionEstimator = estimator

        let renderer = WarpRenderer(device: device)
        try await renderer.prepare()
        self.warpRenderer = renderer

        let detector = SceneChangeDetector(device: device)
        try await detector.prepare()
        self.sceneChangeDetector = detector

        logger.info("MotionInterpolator prepared (block match + forward extrapolation)")
        DiagnosticLog.shared.log("[MotionInterp] Prepared OK (extrapolation mode)")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard let _ = device, let motionEstimator, let warpRenderer,
              let texturePool, let sceneChangeDetector else {
            throw InterpolationError.notPrepared
        }

        // 0. MAD readback 콜백 — 반드시 scene change 판정 전에 등록
        // ⚠️ 이전: success path에서만 등록 → scene change 시 readResult 미호출 → lastMAD 고정 → 전체 차단
        // 수정: 항상 등록하여 command buffer commit 후 반드시 lastMAD 갱신
        let detector = sceneChangeDetector
        let pixelCount = frameA.width * frameA.height
        commandBuffer.addCompletedHandler { _ in
            detector.readResult(pixelCount: pixelCount)
        }

        // 1. 장면 전환 감지 (이전 프레임 쌍 결과 기반, 1프레임 지연 허용)
        if sceneChangeDetector.isSceneChange(frameA: frameA, frameB: frameB, commandBuffer: commandBuffer) {
            logger.debug("Scene change detected, skipping interpolation")
            return nil
        }

        // 2. GPU 모션 추정 (3 compute passes → flow texture)
        let (flowTexture, flowScale) = try motionEstimator.estimateMotion(
            frameA: frameA,
            frameB: frameB,
            commandBuffer: commandBuffer
        )

        // 3. 출력 텍스처 할당
        guard let output = texturePool.acquire(
            width: frameB.width,
            height: frameB.height,
            pixelFormat: .bgra8Unorm,
            usage: [.shaderRead, .shaderWrite]
        ) else {
            throw InterpolationError.textureAllocationFailed
        }

        // 4. 순방향 외삽 워프 (1 compute pass)
        // frameB를 기준으로 t만큼 미래를 예측 (frameA는 모션 추정에만 사용)
        try warpRenderer.encode(
            frameA: frameA,  // 워프 셰이더에서 미사용 (API 호환용)
            frameB: frameB,
            flowTexture: flowTexture,
            output: output,
            t: t,
            flowScale: flowScale,
            commandBuffer: commandBuffer
        )

        return output
    }

    public func shutdown() {
        motionEstimator?.shutdown()
        motionEstimator = nil
        warpRenderer = nil
        sceneChangeDetector = nil
        texturePool?.drain()
        texturePool = nil
        logger.info("MotionInterpolator shut down")
    }
}
