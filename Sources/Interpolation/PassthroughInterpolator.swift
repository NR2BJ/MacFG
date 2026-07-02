@preconcurrency import Metal
import os

/// 더미 보간 엔진 — 실제 보간 없이 frameA/B 블렌딩만 수행.
/// InterpolationPipeline 통합 테스트용.
public final class PassthroughInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "Passthrough (Test)"
    public let isAvailable = true

    private var blendPipeline: MTLComputePipelineState?
    private var device: (any MTLDevice)?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "PassthroughInterpolator")

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)

        // 간단한 블렌드 compute shader를 인라인으로 컴파일
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void blendFrames(
            texture2d<float, access::read>  frameA  [[texture(0)]],
            texture2d<float, access::read>  frameB  [[texture(1)]],
            texture2d<float, access::write> output  [[texture(2)]],
            constant float &t                       [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]])
        {
            if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
            float4 a = frameA.read(gid);
            float4 b = frameB.read(gid);
            output.write(mix(a, b, t), gid);
        }
        """

        let library = try await device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "blendFrames") else {
            throw InterpolationError.shaderCompilationFailed
        }
        blendPipeline = try await device.makeComputePipelineState(function: function)
        logger.info("PassthroughInterpolator prepared")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard let pipeline = blendPipeline,
              let pool = texturePool else {
            throw InterpolationError.notPrepared
        }

        let width = frameB.width
        let height = frameB.height

        guard let output = pool.acquire(width: width, height: height, pixelFormat: frameB.pixelFormat) else {
            throw InterpolationError.textureAllocationFailed
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        var tValue = t
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frameA, index: 0)
        encoder.setTexture(frameB, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBytes(&tValue, length: MemoryLayout<Float>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        return output
    }

    public func shutdown() {
        blendPipeline = nil
        texturePool?.drain()
        logger.info("PassthroughInterpolator shut down")
    }
}

// MARK: - Errors

public enum InterpolationError: Error, Sendable {
    case shaderCompilationFailed
    case notPrepared
    case textureAllocationFailed
    case encoderCreationFailed
    case opticalFlowFailed(String)
    case visionRequestFailed(String)
}
