@preconcurrency import Metal
import os

/// GPU 알파 블렌딩 기반 초고속 프레임 보간 엔진
/// Vision Optical Flow 대신 단순 mix(A, B, t) — <1ms GPU 처리
/// 고스팅은 있지만 시간적으로 정확하고 부드러움
public final class BlendInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "Blend (Fast)"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var blendPipeline: MTLComputePipelineState?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "BlendInterpolator")

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void blendFrames(
        texture2d<float, access::read>  frameA  [[texture(0)]],
        texture2d<float, access::read>  frameB  [[texture(1)]],
        texture2d<float, access::write> output  [[texture(2)]],
        constant float &t                       [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint w = output.get_width();
        uint h = output.get_height();
        if (gid.x >= w || gid.y >= h) return;

        float4 colorA = frameA.read(gid);
        float4 colorB = frameB.read(gid);
        float4 result = mix(colorA, colorB, t);
        result.a = 1.0;
        output.write(result, gid);
    }
    """

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)

        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let function = library.makeFunction(name: "blendFrames") else {
            throw InterpolationError.shaderCompilationFailed
        }
        blendPipeline = try await device.makeComputePipelineState(function: function)
        logger.info("BlendInterpolator prepared")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard device != nil, let blendPipeline, let texturePool else {
            throw InterpolationError.notPrepared
        }

        // 출력 텍스처 할당
        guard let output = texturePool.acquire(
            width: frameB.width,
            height: frameB.height,
            pixelFormat: frameB.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        ) else {
            throw InterpolationError.textureAllocationFailed
        }

        // 블렌딩 인코딩
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        var tValue = t
        encoder.setComputePipelineState(blendPipeline)
        encoder.setTexture(frameA, index: 0)
        encoder.setTexture(frameB, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBytes(&tValue, length: MemoryLayout<Float>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (output.width + 15) / 16,
            height: (output.height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        return output
    }

    public func shutdown() {
        texturePool?.drain()
        texturePool = nil
        blendPipeline = nil
        logger.info("BlendInterpolator shut down")
    }
}
