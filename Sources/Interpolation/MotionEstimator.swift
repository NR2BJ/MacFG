@preconcurrency import Metal
import Monitoring
import os

/// Horn-Schunck variational optical flow 기반 모션 추정 엔진
/// 3-level coarse-to-fine pyramid: L0(1/8) → L1(1/4) → L2(1/2)
/// 1/2 해상도 출력 → 1px 오차 = 2px full-res (vs 이전 1/4 → 4px 오차)
/// Warm-start + Jacobi 반복 → 시간적 일관성 + GPU 완벽 병렬화
public final class MotionEstimator: @unchecked Sendable {
    private let device: any MTLDevice
    private var downsamplePipeline: MTLComputePipelineState?
    private var computeGradPipeline: MTLComputePipelineState?
    private var hsIteratePipeline: MTLComputePipelineState?
    private var upscaleFlowPipeline: MTLComputePipelineState?
    private var warpGrayPipeline: MTLComputePipelineState?
    private var combineFlowPipeline: MTLComputePipelineState?
    private var clearFlowPipeline: MTLComputePipelineState?
    private let logger = Logger(subsystem: "com.macfg", category: "MotionEstimator")

    // Per-level textures
    private var grayA: [any MTLTexture] = []
    private var grayB: [any MTLTexture] = []
    private var gradTextures: [any MTLTexture] = []
    private var flowPing: [any MTLTexture] = []
    private var flowPong: [any MTLTexture] = []
    // Inter-level textures (levels 1+)
    private var upscaledFlowTex: [any MTLTexture] = []   // upscaled from prev level
    private var warpedATex: [any MTLTexture] = []         // A warped by coarse flow
    private var combinedFlowTex: [any MTLTexture] = []    // upscaled + refinement

    private var lastWidth: Int = 0
    private var lastHeight: Int = 0
    private var frameCount: Int = 0

    // ── Configuration ──
    private let scales: [Int] = [8, 4, 2]              // 3-level pyramid
    private let iterationsPerLevel: [Int] = [30, 20, 16] // fewer iterations at finer levels (warm-start helps)
    private let smoothnessAlpha: Float = 0.3            // Horn-Schunck smoothness

    // MARK: - Metal Shaders

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct DownsampleParams {
        uint srcWidth; uint srcHeight;
        uint dstWidth; uint dstHeight;
        uint scale;
        uint _p1; uint _p2; uint _p3;
    };

    struct GradParams {
        uint width; uint height;
        uint _p1; uint _p2;
    };

    struct HSParams {
        uint width; uint height;
        float alpha2;
        uint _p1;
    };

    struct UpscaleParams {
        uint srcWidth; uint srcHeight;
        uint dstWidth; uint dstHeight;
        float scaleRatio;
        uint _p1; uint _p2; uint _p3;
    };

    struct WarpParams {
        uint width; uint height;
        uint _p1; uint _p2;
    };

    struct CombineParams {
        uint width; uint height;
        uint _p1; uint _p2;
    };

    kernel void clearFlow(
        texture2d<float, access::write> tex [[texture(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x < tex.get_width() && gid.y < tex.get_height())
            tex.write(float4(0, 0, 0, 0), gid);
    }

    kernel void downsampleGrayscale(
        texture2d<float, access::read> input   [[texture(0)]],
        texture2d<float, access::write> output [[texture(1)]],
        constant DownsampleParams &params      [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.dstWidth || gid.y >= params.dstHeight) return;
        uint scale = params.scale;
        uint2 base = gid * scale;
        float sum = 0.0;
        uint count = 0;
        for (uint dy = 0; dy < scale && (base.y + dy) < params.srcHeight; dy++)
            for (uint dx = 0; dx < scale && (base.x + dx) < params.srcWidth; dx++) {
                float4 c = input.read(base + uint2(dx, dy));
                sum += dot(c.rgb, float3(0.299, 0.587, 0.114));
                count++;
            }
        float lum = count > 0 ? sum / float(count) : 0.0;
        output.write(float4(lum, 0, 0, 1), gid);
    }

    kernel void computeGradients(
        texture2d<float, access::read> imgA  [[texture(0)]],
        texture2d<float, access::read> imgB  [[texture(1)]],
        texture2d<float, access::write> grad [[texture(2)]],
        constant GradParams &params          [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.width || gid.y >= params.height) return;
        int w = int(params.width);
        int h = int(params.height);
        int2 pos = int2(gid);

        int2 pL  = clamp(pos + int2(-1, 0), int2(0), int2(w-1, h-1));
        int2 pR  = clamp(pos + int2( 1, 0), int2(0), int2(w-1, h-1));
        int2 pU  = clamp(pos + int2( 0,-1), int2(0), int2(w-1, h-1));
        int2 pD  = clamp(pos + int2( 0, 1), int2(0), int2(w-1, h-1));
        int2 pLU = clamp(pos + int2(-1,-1), int2(0), int2(w-1, h-1));
        int2 pRU = clamp(pos + int2( 1,-1), int2(0), int2(w-1, h-1));
        int2 pLD = clamp(pos + int2(-1, 1), int2(0), int2(w-1, h-1));
        int2 pRD = clamp(pos + int2( 1, 1), int2(0), int2(w-1, h-1));

        float aLU = 0.5 * (imgA.read(uint2(pLU)).r + imgB.read(uint2(pLU)).r);
        float aRU = 0.5 * (imgA.read(uint2(pRU)).r + imgB.read(uint2(pRU)).r);
        float aL  = 0.5 * (imgA.read(uint2(pL)).r  + imgB.read(uint2(pL)).r);
        float aR  = 0.5 * (imgA.read(uint2(pR)).r  + imgB.read(uint2(pR)).r);
        float aLD = 0.5 * (imgA.read(uint2(pLD)).r + imgB.read(uint2(pLD)).r);
        float aRD = 0.5 * (imgA.read(uint2(pRD)).r + imgB.read(uint2(pRD)).r);
        float aU  = 0.5 * (imgA.read(uint2(pU)).r  + imgB.read(uint2(pU)).r);
        float aD  = 0.5 * (imgA.read(uint2(pD)).r  + imgB.read(uint2(pD)).r);

        float Ix = (-3*aLU + 3*aRU - 10*aL + 10*aR - 3*aLD + 3*aRD) / 32.0;
        float Iy = (-3*aLU - 10*aU - 3*aRU + 3*aLD + 10*aD + 3*aRD) / 32.0;
        float It = imgB.read(gid).r - imgA.read(gid).r;

        grad.write(float4(Ix, Iy, It, 0), gid);
    }

    kernel void hornSchunckIterate(
        texture2d<float, access::read> flowIn   [[texture(0)]],
        texture2d<float, access::read> gradTex  [[texture(1)]],
        texture2d<float, access::write> flowOut [[texture(2)]],
        constant HSParams &params              [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.width || gid.y >= params.height) return;

        int w = int(params.width);
        int h = int(params.height);
        int2 pos = int2(gid);

        float2 sum = float2(0);
        float count = 0;
        if (pos.x > 0)   { sum += flowIn.read(uint2(pos + int2(-1, 0))).rg; count += 1; }
        if (pos.x < w-1) { sum += flowIn.read(uint2(pos + int2( 1, 0))).rg; count += 1; }
        if (pos.y > 0)   { sum += flowIn.read(uint2(pos + int2( 0,-1))).rg; count += 1; }
        if (pos.y < h-1) { sum += flowIn.read(uint2(pos + int2( 0, 1))).rg; count += 1; }
        float2 avg = sum / max(count, 1.0);

        float3 g = gradTex.read(gid).rgb;
        float Ix = g.x, Iy = g.y, It = g.z;

        float denom = params.alpha2 + Ix * Ix + Iy * Iy;
        float term = (Ix * avg.x + Iy * avg.y + It) / denom;

        float2 flow;
        flow.x = avg.x - Ix * term;
        flow.y = avg.y - Iy * term;

        flowOut.write(float4(flow, 0, 0), gid);
    }

    kernel void upscaleFlow(
        texture2d<float, access::read> srcFlow   [[texture(0)]],
        texture2d<float, access::write> dstFlow  [[texture(1)]],
        constant UpscaleParams &params           [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.dstWidth || gid.y >= params.dstHeight) return;

        float2 srcSize = float2(params.srcWidth, params.srcHeight);
        float2 dstSize = float2(params.dstWidth, params.dstHeight);
        float2 srcCoord = (float2(gid) + 0.5) * srcSize / dstSize;
        srcCoord = clamp(srcCoord, float2(0.5), srcSize - 0.5);

        float2 base = floor(srcCoord - 0.5);
        float2 frac = srcCoord - 0.5 - base;
        int2 maxP = int2(srcSize) - 1;
        int2 p00 = clamp(int2(base), int2(0), maxP);
        int2 p10 = clamp(int2(base) + int2(1,0), int2(0), maxP);
        int2 p01 = clamp(int2(base) + int2(0,1), int2(0), maxP);
        int2 p11 = clamp(int2(base) + int2(1,1), int2(0), maxP);

        float2 f00 = srcFlow.read(uint2(p00)).rg;
        float2 f10 = srcFlow.read(uint2(p10)).rg;
        float2 f01 = srcFlow.read(uint2(p01)).rg;
        float2 f11 = srcFlow.read(uint2(p11)).rg;

        float2 top = mix(f00, f10, frac.x);
        float2 bot = mix(f01, f11, frac.x);
        float2 flow = mix(top, bot, frac.y) * params.scaleRatio;

        dstFlow.write(float4(flow, 0, 0), gid);
    }

    kernel void warpGrayscale(
        texture2d<float, access::read> input   [[texture(0)]],
        texture2d<float, access::read> flow    [[texture(1)]],
        texture2d<float, access::write> output [[texture(2)]],
        constant WarpParams &params            [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.width || gid.y >= params.height) return;
        float2 f = flow.read(gid).rg;
        float w = float(params.width), h = float(params.height);
        float2 srcPos = clamp(float2(gid) - f, float2(0), float2(w-1, h-1));

        float2 base = floor(srcPos);
        float2 frac = srcPos - base;
        uint2 p00 = uint2(base);
        uint2 p10 = uint2(min(base.x+1, w-1), base.y);
        uint2 p01 = uint2(base.x, min(base.y+1, h-1));
        uint2 p11 = uint2(min(base.x+1, w-1), min(base.y+1, h-1));

        float top = mix(input.read(p00).r, input.read(p10).r, frac.x);
        float bot = mix(input.read(p01).r, input.read(p11).r, frac.x);
        float val = mix(top, bot, frac.y);
        output.write(float4(val, 0, 0, 1), gid);
    }

    kernel void combineFlows(
        texture2d<float, access::read> flowA   [[texture(0)]],
        texture2d<float, access::read> flowB   [[texture(1)]],
        texture2d<float, access::write> output [[texture(2)]],
        constant CombineParams &params         [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= params.width || gid.y >= params.height) return;
        float2 a = flowA.read(gid).rg;
        float2 b = flowB.read(gid).rg;
        float2 combined = a + b;
        if (length(combined) < 0.3) combined = float2(0);
        output.write(float4(combined, 0, 0), gid);
    }
    """

    // MARK: - Init / Prepare

    public init(device: any MTLDevice) {
        self.device = device
    }

    public func prepare() async throws {
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)

        guard let clearFn = library.makeFunction(name: "clearFlow"),
              let dsFn = library.makeFunction(name: "downsampleGrayscale"),
              let gradFn = library.makeFunction(name: "computeGradients"),
              let hsFn = library.makeFunction(name: "hornSchunckIterate"),
              let upFn = library.makeFunction(name: "upscaleFlow"),
              let warpFn = library.makeFunction(name: "warpGrayscale"),
              let combineFn = library.makeFunction(name: "combineFlows") else {
            throw InterpolationError.shaderCompilationFailed
        }

        clearFlowPipeline = try await device.makeComputePipelineState(function: clearFn)
        downsamplePipeline = try await device.makeComputePipelineState(function: dsFn)
        computeGradPipeline = try await device.makeComputePipelineState(function: gradFn)
        hsIteratePipeline = try await device.makeComputePipelineState(function: hsFn)
        upscaleFlowPipeline = try await device.makeComputePipelineState(function: upFn)
        warpGrayPipeline = try await device.makeComputePipelineState(function: warpFn)
        combineFlowPipeline = try await device.makeComputePipelineState(function: combineFn)

        logger.info("MotionEstimator prepared (Horn-Schunck 3-level)")
        DiagnosticLog.shared.log("[MotionEst] HS 3-level: scales=\(self.scales), iters=\(self.iterationsPerLevel), α=\(self.smoothnessAlpha)")
    }

    // MARK: - Motion Estimation

    public func estimateMotion(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (flowTexture: any MTLTexture, flowScale: Float) {
        guard let downsamplePipeline, let computeGradPipeline, let hsIteratePipeline,
              let upscaleFlowPipeline, let warpGrayPipeline, let combineFlowPipeline,
              let clearFlowPipeline else {
            throw InterpolationError.notPrepared
        }

        let fullW = frameA.width
        let fullH = frameA.height
        let numLevels = scales.count

        let needsInit = (fullW != lastWidth || fullH != lastHeight)
        if needsInit {
            allocateTextures(fullW: fullW, fullH: fullH)
            lastWidth = fullW
            lastHeight = fullH
            frameCount = 0
        }

        frameCount += 1
        let verbose = frameCount <= 5 || frameCount % 300 == 0
        let tg16 = MTLSize(width: 16, height: 16, depth: 1)
        let alpha2 = smoothnessAlpha * smoothnessAlpha

        // Clear flow on first frame
        if needsInit {
            guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                throw InterpolationError.encoderCreationFailed
            }
            enc.setComputePipelineState(clearFlowPipeline)
            for level in 0..<numLevels {
                let w = max(1, fullW / scales[level])
                let h = max(1, fullH / scales[level])
                let groups = MTLSize(width: (w+15)/16, height: (h+15)/16, depth: 1)
                enc.setTexture(flowPing[level], index: 0)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                enc.setTexture(flowPong[level], index: 0)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
            }
            enc.endEncoding()
        }

        // Process each level
        for level in 0..<numLevels {
            let lW = max(1, fullW / scales[level])
            let lH = max(1, fullH / scales[level])
            let groups = MTLSize(width: (lW+15)/16, height: (lH+15)/16, depth: 1)

            // (a) Downsample A,B → grayscale
            do {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                    throw InterpolationError.encoderCreationFailed
                }
                enc.setComputePipelineState(downsamplePipeline)
                var dsParams = DownsampleParams(
                    srcWidth: UInt32(fullW), srcHeight: UInt32(fullH),
                    dstWidth: UInt32(lW), dstHeight: UInt32(lH),
                    scale: UInt32(scales[level])
                )
                enc.setTexture(frameA, index: 0)
                enc.setTexture(grayA[level], index: 1)
                enc.setBytes(&dsParams, length: MemoryLayout<DownsampleParams>.size, index: 0)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)

                enc.setTexture(frameB, index: 0)
                enc.setTexture(grayB[level], index: 1)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                enc.endEncoding()
            }

            if level == 0 {
                // Level 0: compute gradients directly from grayA, grayB
                do {
                    guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                        throw InterpolationError.encoderCreationFailed
                    }
                    enc.setComputePipelineState(computeGradPipeline)
                    enc.setTexture(grayA[0], index: 0)
                    enc.setTexture(grayB[0], index: 1)
                    enc.setTexture(gradTextures[0], index: 2)
                    var params = GradParams(width: UInt32(lW), height: UInt32(lH))
                    enc.setBytes(&params, length: MemoryLayout<GradParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    enc.endEncoding()
                }
            } else {
                // Levels 1+: upscale previous flow → warp A → compute residual gradients
                let prevLevel = level - 1
                let prevW = max(1, fullW / scales[prevLevel])
                let prevH = max(1, fullH / scales[prevLevel])
                let scaleRatio = Float(lW) / Float(prevW)
                let interIdx = level - 1  // index into inter-level arrays

                // Previous level result: combinedFlowTex[prevLevel-1] for level≥2, flowPing[0] for level 1
                let prevFlowResult: any MTLTexture = (level == 1) ? flowPing[0] : combinedFlowTex[interIdx - 1]

                // Upscale flow
                do {
                    guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                        throw InterpolationError.encoderCreationFailed
                    }
                    enc.setComputePipelineState(upscaleFlowPipeline)
                    enc.setTexture(prevFlowResult, index: 0)
                    enc.setTexture(upscaledFlowTex[interIdx], index: 1)
                    var params = UpscaleParams(
                        srcWidth: UInt32(prevW), srcHeight: UInt32(prevH),
                        dstWidth: UInt32(lW), dstHeight: UInt32(lH),
                        scaleRatio: scaleRatio
                    )
                    enc.setBytes(&params, length: MemoryLayout<UpscaleParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    enc.endEncoding()
                }

                // Warp grayA using upscaled flow
                do {
                    guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                        throw InterpolationError.encoderCreationFailed
                    }
                    enc.setComputePipelineState(warpGrayPipeline)
                    enc.setTexture(grayA[level], index: 0)
                    enc.setTexture(upscaledFlowTex[interIdx], index: 1)
                    enc.setTexture(warpedATex[interIdx], index: 2)
                    var params = WarpParams(width: UInt32(lW), height: UInt32(lH))
                    enc.setBytes(&params, length: MemoryLayout<WarpParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    enc.endEncoding()
                }

                // Compute residual gradients (warpedA vs grayB)
                do {
                    guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                        throw InterpolationError.encoderCreationFailed
                    }
                    enc.setComputePipelineState(computeGradPipeline)
                    enc.setTexture(warpedATex[interIdx], index: 0)
                    enc.setTexture(grayB[level], index: 1)
                    enc.setTexture(gradTextures[level], index: 2)
                    var params = GradParams(width: UInt32(lW), height: UInt32(lH))
                    enc.setBytes(&params, length: MemoryLayout<GradParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    enc.endEncoding()
                }
            }

            // (b) Horn-Schunck iterations (warm-start from flowPing[level])
            do {
                guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                    throw InterpolationError.encoderCreationFailed
                }
                enc.setComputePipelineState(hsIteratePipeline)
                var params = HSParams(width: UInt32(lW), height: UInt32(lH), alpha2: alpha2)

                let iters = iterationsPerLevel[level]
                for i in 0..<iters {
                    let src = (i % 2 == 0) ? flowPing[level] : flowPong[level]
                    let dst = (i % 2 == 0) ? flowPong[level] : flowPing[level]
                    enc.setTexture(src, index: 0)
                    enc.setTexture(gradTextures[level], index: 1)
                    enc.setTexture(dst, index: 2)
                    enc.setBytes(&params, length: MemoryLayout<HSParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    if i < iters - 1 {
                        enc.memoryBarrier(scope: .textures)
                    }
                }
                enc.endEncoding()
            }

            // Result in flowPing[level] (even iteration counts)

            // (c) For levels 1+: combine upscaled + refinement
            if level > 0 {
                let interIdx = level - 1
                do {
                    guard let enc = commandBuffer.makeComputeCommandEncoder() else {
                        throw InterpolationError.encoderCreationFailed
                    }
                    enc.setComputePipelineState(combineFlowPipeline)
                    enc.setTexture(upscaledFlowTex[interIdx], index: 0)
                    enc.setTexture(flowPing[level], index: 1)
                    enc.setTexture(combinedFlowTex[interIdx], index: 2)
                    var params = CombineParams(width: UInt32(lW), height: UInt32(lH))
                    enc.setBytes(&params, length: MemoryLayout<CombineParams>.size, index: 0)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg16)
                    enc.endEncoding()
                }
            }
        }

        // Final output: combinedFlowTex for last level, or flowPing[0] if single level
        let lastLevel = numLevels - 1
        let finalFlow: any MTLTexture = (lastLevel > 0) ? combinedFlowTex[lastLevel - 1] : flowPing[0]
        let finalW = max(1, fullW / scales[lastLevel])
        let finalScale = Float(finalW) / Float(fullW)

        if verbose {
            let finalH = max(1, fullH / scales[lastLevel])
            DiagnosticLog.shared.log("[MotionEst] HS frame #\(frameCount): final \(finalW)x\(finalH) (1/\(scales[lastLevel])), α²=\(alpha2)")
        }

        return (finalFlow, finalScale)
    }

    // MARK: - Texture Allocation

    private func allocateTextures(fullW: Int, fullH: Int) {
        grayA.removeAll()
        grayB.removeAll()
        gradTextures.removeAll()
        flowPing.removeAll()
        flowPong.removeAll()
        upscaledFlowTex.removeAll()
        warpedATex.removeAll()
        combinedFlowTex.removeAll()

        for scale in scales {
            let w = max(1, fullW / scale)
            let h = max(1, fullH / scale)
            grayA.append(makeTexture(width: w, height: h, format: .r8Unorm)!)
            grayB.append(makeTexture(width: w, height: h, format: .r8Unorm)!)
            gradTextures.append(makeTexture(width: w, height: h, format: .rgba16Float)!)
            flowPing.append(makeTexture(width: w, height: h, format: .rg16Float)!)
            flowPong.append(makeTexture(width: w, height: h, format: .rg16Float)!)
        }

        // Inter-level textures (for levels 1+)
        for i in 1..<scales.count {
            let w = max(1, fullW / scales[i])
            let h = max(1, fullH / scales[i])
            upscaledFlowTex.append(makeTexture(width: w, height: h, format: .rg16Float)!)
            warpedATex.append(makeTexture(width: w, height: h, format: .r8Unorm)!)
            combinedFlowTex.append(makeTexture(width: w, height: h, format: .rg16Float)!)
        }

        let l0W = max(1, fullW / scales[0])
        let l0H = max(1, fullH / scales[0])
        let finalW = max(1, fullW / scales[scales.count - 1])
        let finalH = max(1, fullH / scales[scales.count - 1])
        DiagnosticLog.shared.log("[MotionEst] 3-level: L0 \(l0W)x\(l0H), L1 \(max(1,fullW/scales[1]))x\(max(1,fullH/scales[1])), L2 \(finalW)x\(finalH)")
    }

    private func makeTexture(width: Int, height: Int, format: MTLPixelFormat) -> (any MTLTexture)? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: max(1, width), height: max(1, height), mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    // MARK: - Parameter Structs

    private struct DownsampleParams {
        var srcWidth: UInt32; var srcHeight: UInt32
        var dstWidth: UInt32; var dstHeight: UInt32
        var scale: UInt32
        var _p1: UInt32 = 0; var _p2: UInt32 = 0; var _p3: UInt32 = 0
    }

    private struct GradParams {
        var width: UInt32; var height: UInt32
        var _p1: UInt32 = 0; var _p2: UInt32 = 0
    }

    private struct HSParams {
        var width: UInt32; var height: UInt32
        var alpha2: Float
        var _p1: UInt32 = 0
    }

    private struct UpscaleParams {
        var srcWidth: UInt32; var srcHeight: UInt32
        var dstWidth: UInt32; var dstHeight: UInt32
        var scaleRatio: Float
        var _p1: UInt32 = 0; var _p2: UInt32 = 0; var _p3: UInt32 = 0
    }

    private struct WarpParams {
        var width: UInt32; var height: UInt32
        var _p1: UInt32 = 0; var _p2: UInt32 = 0
    }

    private struct CombineParams {
        var width: UInt32; var height: UInt32
        var _p1: UInt32 = 0; var _p2: UInt32 = 0
    }

    public func shutdown() {
        grayA.removeAll()
        grayB.removeAll()
        gradTextures.removeAll()
        flowPing.removeAll()
        flowPong.removeAll()
        upscaledFlowTex.removeAll()
        warpedATex.removeAll()
        combinedFlowTex.removeAll()
        downsamplePipeline = nil
        computeGradPipeline = nil
        hsIteratePipeline = nil
        upscaleFlowPipeline = nil
        warpGrayPipeline = nil
        combineFlowPipeline = nil
        clearFlowPipeline = nil
    }
}
