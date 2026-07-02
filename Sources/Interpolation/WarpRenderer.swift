@preconcurrency import Metal
import os

/// Metal 워프 셰이더 파이프라인 관리 및 디스패치
/// 양방향 시간 블렌딩 + bilinear flow sampling
public final class WarpRenderer: @unchecked Sendable {
    private var warpPipeline: MTLComputePipelineState?
    private let device: any MTLDevice
    private let logger = Logger(subsystem: "com.macfg", category: "WarpRenderer")

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    static float4 bilinearSample(texture2d<float, access::read> tex, float2 coord) {
        float2 texSize = float2(tex.get_width(), tex.get_height());
        float2 clampedCoord = clamp(coord, float2(0.0), texSize - 1.0);
        float2 base = floor(clampedCoord);
        float2 frac = clampedCoord - base;
        uint2 p00 = uint2(base);
        uint2 p10 = uint2(min(base.x + 1, texSize.x - 1), base.y);
        uint2 p01 = uint2(base.x, min(base.y + 1, texSize.y - 1));
        uint2 p11 = uint2(min(base.x + 1, texSize.x - 1), min(base.y + 1, texSize.y - 1));
        float4 top = mix(tex.read(p00), tex.read(p10), frac.x);
        float4 bot = mix(tex.read(p01), tex.read(p11), frac.x);
        return mix(top, bot, frac.y);
    }

    struct WarpParams {
        float t;
        float flowScale; // flow 텍스처 크기 / 원본 크기 (예: 0.25)
    };

    // Flow 텍스처를 bilinear 보간으로 읽기 (point sample 대신)
    // 원본 좌표 → flow 좌표 변환 + flow 벡터를 원본 좌표계로 스케일업
    static float2 sampleFlowBilinear(texture2d<half, access::read> flowTex, float2 outputCoord, float flowScale) {
        float2 flowCoord = outputCoord * flowScale;
        float2 flowSize = float2(flowTex.get_width(), flowTex.get_height());
        flowCoord = clamp(flowCoord, float2(0.5), flowSize - 0.5);

        float2 base = floor(flowCoord - 0.5);
        float2 frac = flowCoord - 0.5 - base;

        int2 p00 = int2(base);
        int2 p10 = p00 + int2(1, 0);
        int2 p01 = p00 + int2(0, 1);
        int2 p11 = p00 + int2(1, 1);

        int2 maxP = int2(flowSize) - 1;
        p00 = clamp(p00, int2(0), maxP);
        p10 = clamp(p10, int2(0), maxP);
        p01 = clamp(p01, int2(0), maxP);
        p11 = clamp(p11, int2(0), maxP);

        float2 f00 = float2(flowTex.read(uint2(p00)).rg);
        float2 f10 = float2(flowTex.read(uint2(p10)).rg);
        float2 f01 = float2(flowTex.read(uint2(p01)).rg);
        float2 f11 = float2(flowTex.read(uint2(p11)).rg);

        float2 top = mix(f00, f10, frac.x);
        float2 bot = mix(f01, f11, frac.x);
        float2 flow = mix(top, bot, frac.y);

        // flow 벡터를 원본 좌표계로 스케일업
        return flow / flowScale;
    }

    kernel void motionWarp(
        texture2d<float, access::read>  frameA    [[texture(0)]],
        texture2d<float, access::read>  frameB    [[texture(1)]],
        texture2d<half, access::read>   flowAtoB  [[texture(2)]],
        texture2d<float, access::write> output    [[texture(3)]],
        constant WarpParams &params               [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint w = output.get_width();
        uint h = output.get_height();
        if (gid.x >= w || gid.y >= h) return;

        float t = params.t;

        // ── 진단 모드: 단순 프레임 블렌딩 (flow 무시) ──
        // 고스팅은 생기지만 flow 문제와 타이밍 문제를 분리하기 위한 테스트
        float4 colorA = frameA.read(gid);
        float4 colorB = frameB.read(gid);
        float4 blended = mix(colorA, colorB, t);
        blended.a = 1.0;
        output.write(blended, gid);
    }
    """

    public init(device: any MTLDevice) {
        self.device = device
    }

    /// 셰이더 컴파일 (1회)
    public func prepare() async throws {
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let function = library.makeFunction(name: "motionWarp") else {
            throw InterpolationError.shaderCompilationFailed
        }
        warpPipeline = try await device.makeComputePipelineState(function: function)
        logger.info("WarpRenderer prepared (motionWarp + bilinear flow)")
    }

    /// 워프 셰이더를 커맨드 버퍼에 인코딩
    public func encode(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        flowTexture: any MTLTexture,
        output: any MTLTexture,
        t: Float,
        flowScale: Float = 1.0,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        guard let pipeline = warpPipeline else {
            throw InterpolationError.notPrepared
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        var params = (t, flowScale)
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(frameA, index: 0)
        encoder.setTexture(frameB, index: 1)
        encoder.setTexture(flowTexture, index: 2)
        encoder.setTexture(output, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<(Float, Float)>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (output.width + 15) / 16,
            height: (output.height + 15) / 16,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}
