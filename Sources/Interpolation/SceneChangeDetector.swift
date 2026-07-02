@preconcurrency import Metal
import os

/// GPU 기반 장면 전환 감지 — 두 프레임 간 MAD(Mean Absolute Difference) 계산
public final class SceneChangeDetector: @unchecked Sendable {
    private var pipeline: MTLComputePipelineState?
    private var resultBuffer: (any MTLBuffer)?
    private let device: any MTLDevice
    private let logger = Logger(subsystem: "com.macfg", category: "SceneChangeDetector")

    /// MAD 임계값 (0~255 스케일). 기본 30/255 ≈ 0.118
    public var threshold: Float = 0.118

    // 이전 프레임 쌍의 결과 (1프레임 지연 회피용)
    private var lastMAD: Float = 0

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void computeMAD(
        texture2d<float, access::read> frameA [[texture(0)]],
        texture2d<float, access::read> frameB [[texture(1)]],
        device atomic_uint *accumulator       [[buffer(0)]],
        constant uint2 &dimensions            [[buffer(1)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= dimensions.x || gid.y >= dimensions.y) return;
        float3 a = frameA.read(gid).rgb;
        float3 b = frameB.read(gid).rgb;
        float lumA = dot(a, float3(0.299, 0.587, 0.114));
        float lumB = dot(b, float3(0.299, 0.587, 0.114));
        uint absDiff = uint(abs(lumA - lumB) * 255.0);
        atomic_fetch_add_explicit(accumulator, absDiff, memory_order_relaxed);
    }
    """

    public init(device: any MTLDevice) {
        self.device = device
    }

    public func prepare() async throws {
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let function = library.makeFunction(name: "computeMAD") else {
            throw InterpolationError.shaderCompilationFailed
        }
        pipeline = try await device.makeComputePipelineState(function: function)

        // Accumulator 버퍼 (shared storage — CPU에서 읽기 가능)
        resultBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        logger.info("SceneChangeDetector prepared")
    }

    /// 두 프레임 간 장면 전환 여부 판단.
    /// GPU readback 지연 회피를 위해, 현재 호출 시 이전 결과를 반환하고
    /// 동시에 새 계산을 커맨드 버퍼에 인코딩.
    public func isSceneChange(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) -> Bool {
        let isChange = lastMAD > threshold

        // 새 계산 인코딩 (다음 호출 시 결과 사용)
        encodeMAD(frameA: frameA, frameB: frameB, commandBuffer: commandBuffer)

        return isChange
    }

    /// MAD 결과 읽기 (커맨드 버퍼 완료 후 호출)
    public func readResult(pixelCount: Int) {
        guard let buffer = resultBuffer else { return }
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        let totalDiff = Float(ptr.pointee)
        lastMAD = totalDiff / Float(max(pixelCount, 1)) / 255.0

        // 리셋
        ptr.pointee = 0
    }

    private func encodeMAD(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) {
        guard let pipeline, let resultBuffer else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        // Accumulator 초기화
        let ptr = resultBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
        ptr.pointee = 0

        var dims = SIMD2<UInt32>(UInt32(frameA.width), UInt32(frameA.height))

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frameA, index: 0)
        encoder.setTexture(frameB, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 0)
        encoder.setBytes(&dims, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (frameA.width + 15) / 16,
            height: (frameA.height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
