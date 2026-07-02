@preconcurrency import Metal
import os

/// MV 감상용 보간/후처리 엔진.
/// 복잡한 optical flow 대신 motion-adaptive temporal blend와 약한 unsharp mask를 한 패스에 적용한다.
public final class MVBoostInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "MV Boost"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var texturePool: TexturePool?
    private var pipeline: MTLComputePipelineState?
    private let logger = Logger(subsystem: "com.macfg", category: "MVBoost")

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float t;
        float sharpenStrength;
        float temporalStrength;
        uint2 size;
    };

    static float luma(float3 c) {
        return dot(c, float3(0.299, 0.587, 0.114));
    }

    kernel void mvBoost(
        texture2d<float, access::read> frameA [[texture(0)]],
        texture2d<float, access::read> frameB [[texture(1)]],
        texture2d<float, access::write> output [[texture(2)]],
        constant Params& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= p.size.x || gid.y >= p.size.y) return;

        uint2 left = uint2(gid.x > 0 ? gid.x - 1 : gid.x, gid.y);
        uint2 right = uint2(min(gid.x + 1, p.size.x - 1), gid.y);
        uint2 up = uint2(gid.x, gid.y > 0 ? gid.y - 1 : gid.y);
        uint2 down = uint2(gid.x, min(gid.y + 1, p.size.y - 1));

        float4 a = frameA.read(gid);
        float4 b = frameB.read(gid);

        float4 blurB = (
            frameB.read(left) +
            frameB.read(right) +
            frameB.read(up) +
            frameB.read(down) +
            b * 4.0
        ) * 0.125;

        float3 sharpenedB = clamp(b.rgb + (b.rgb - blurB.rgb) * p.sharpenStrength, 0.0, 1.0);

        float diff = abs(luma(a.rgb) - luma(b.rgb));
        float temporalWeight = p.temporalStrength * (1.0 - smoothstep(0.04, 0.22, diff));
        float3 temporal = mix(a.rgb, sharpenedB, p.t);
        float3 currentBiased = mix(temporal, sharpenedB, 1.0 - temporalWeight);

        output.write(float4(currentBiased, 1.0), gid);
    }
    """

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)

        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let function = library.makeFunction(name: "mvBoost") else {
            throw InterpolationError.shaderCompilationFailed
        }
        pipeline = try await device.makeComputePipelineState(function: function)
        logger.info("MVBoostInterpolator prepared")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard device != nil, let texturePool, let pipeline else {
            throw InterpolationError.notPrepared
        }

        guard let output = texturePool.acquire(
            width: frameB.width,
            height: frameB.height,
            pixelFormat: frameB.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        ), let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        struct Params {
            var t: Float
            var sharpenStrength: Float
            var temporalStrength: Float
            var size: SIMD2<UInt32>
        }
        var params = Params(
            t: t,
            sharpenStrength: 0.42,
            temporalStrength: 0.72,
            size: SIMD2(UInt32(frameB.width), UInt32(frameB.height))
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frameA, index: 0)
        encoder.setTexture(frameB, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: (frameB.width + 15) / 16, height: (frameB.height + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
        encoder.endEncoding()

        return output
    }

    public func shutdown() {
        texturePool?.drain()
        texturePool = nil
        pipeline = nil
        logger.info("MVBoostInterpolator shut down")
    }
}
