import Metal
import VideoToolbox
import CoreVideo
import CoreMedia
import os

/// Apple VideoToolbox 저지연 신경망 SR(VTLowLatencySuperResolutionScaler) 2x 래퍼.
/// ANE에서 실행 → GPU 미사용, 실측 ~1.3ms(960×540→1080p). 입력 ≤960px, 배율 2배 고정.
/// 신경망이라 MetalFX Spatial보다 화질↑. 초과 배율은 호출자가 MetalFX로 체이닝.
///
/// 파이프라인: BGRA → 420v(CSC) → VT SR 2x → 420v → BGRA. 전부 호출자 커맨드버퍼에 인코딩.
/// 핵심: VT config가 요구하는 sourcePixelBufferAttributes(ExtendedPixels 패딩 포함)로 버퍼 생성 —
/// 손수 만든 420v 버퍼는 -19730 실패 (LLFI와 동일 함정, 2026-07-03 실측).
final class NeuralUpscaler {
    static let maxInput = 960

    private let device: any MTLDevice
    private let logger = Logger(subsystem: "com.macfg", category: "NeuralUpscaler")

    private var toYUV: (any MTLComputePipelineState)?
    private var toRGB: (any MTLComputePipelineState)?

    private var proc: VTFrameProcessor?
    private var srcFrame: VTFrameProcessorFrame?
    private var dstFrame: VTFrameProcessorFrame?
    private var srcLuma, srcChroma, dstLuma, dstChroma: (any MTLTexture)?
    private var outBGRA: (any MTLTexture)?
    private var inW = 0, inH = 0
    private var unsupported = false

    init(device: any MTLDevice) {
        self.device = device
        compileShaders()
        if !VTLowLatencySuperResolutionScalerConfiguration.isSupported {
            unsupported = true
            logger.info("VT low-latency SR unsupported on this device")
        }
    }

    /// BGRA 텍스처를 2배로 신경망 업스케일. 입력이 >960이거나 비지원/실패면 nil(→ 호출자 MetalFX 폴백).
    func upscale(_ bgra: any MTLTexture, into cb: any MTLCommandBuffer) -> (any MTLTexture)? {
        guard !unsupported, let toYUV, let toRGB else { return nil }
        let w = bgra.width, h = bgra.height
        guard w >= 96, h >= 96, w <= Self.maxInput, h <= Self.maxInput else { return nil }

        if proc == nil || inW != w || inH != h {
            guard rebuild(w, h) else { unsupported = true; return nil }
        }
        guard let srcLuma, let srcChroma, let dstLuma, let dstChroma,
              let outBGRA, let proc, let srcFrame, let dstFrame else { return nil }

        // 1) BGRA → src 420v (다운샘플 CSC)
        guard let e1 = cb.makeComputeCommandEncoder() else { return nil }
        e1.setComputePipelineState(toYUV)
        e1.setTexture(bgra, index: 0)
        e1.setTexture(srcLuma, index: 1)
        e1.setTexture(srcChroma, index: 2)
        dispatch(e1, toYUV, srcChroma.width, srcChroma.height)
        e1.endEncoding()

        // 2) VT SR 2x — 호출자 커맨드버퍼에 인코딩 (ANE). 실패는 cb 완료 시 error로 표면화.
        let params = VTLowLatencySuperResolutionScalerParameters(sourceFrame: srcFrame, destinationFrame: dstFrame)
        proc.process(with: cb, parameters: params)

        // 3) dst 420v → BGRA
        guard let e2 = cb.makeComputeCommandEncoder() else { return nil }
        e2.setComputePipelineState(toRGB)
        e2.setTexture(dstLuma, index: 0)
        e2.setTexture(dstChroma, index: 1)
        e2.setTexture(outBGRA, index: 2)
        dispatch(e2, toRGB, outBGRA.width, outBGRA.height)
        e2.endEncoding()

        return outBGRA
    }

    private func dispatch(_ enc: any MTLComputeCommandEncoder, _ pso: any MTLComputePipelineState, _ w: Int, _ h: Int) {
        let tw = min(pso.threadExecutionWidth, 16)
        let th = min(max(pso.maxTotalThreadsPerThreadgroup / tw, 1), 16)
        enc.dispatchThreadgroups(
            MTLSize(width: (w + tw - 1) / tw, height: (h + th - 1) / th, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }

    private func rebuild(_ w: Int, _ h: Int) -> Bool {
        teardown()
        let config = VTLowLatencySuperResolutionScalerConfiguration(frameWidth: w, frameHeight: h, scaleFactor: 2.0)
        guard let srcAttrs = config.value(forKey: "sourcePixelBufferAttributes") as? [String: Any],
              let dstAttrs = config.value(forKey: "destinationPixelBufferAttributes") as? [String: Any] else { return false }
        let p = VTFrameProcessor()
        do { try p.startSession(configuration: config) } catch { logger.error("startSession: \(error.localizedDescription)"); return false }

        guard let sBuf = makeBuffer(srcAttrs), let dBuf = makeBuffer(dstAttrs),
              let sSurf = CVPixelBufferGetIOSurface(sBuf)?.takeUnretainedValue(),
              let dSurf = CVPixelBufferGetIOSurface(dBuf)?.takeUnretainedValue() else { return false }

        // 평면 텍스처 뷰 (luma r8 = 실제 크기, chroma rg8 = 절반)
        func plane(_ surf: IOSurfaceRef, _ pw: Int, _ ph: Int, _ fmt: MTLPixelFormat, _ idx: Int, write: Bool) -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: pw, height: ph, mipmapped: false)
            d.usage = write ? [.shaderWrite, .shaderRead] : [.shaderRead]
            d.storageMode = .shared
            return device.makeTexture(descriptor: d, iosurface: surf, plane: idx)
        }
        guard let sl = plane(sSurf, w, h, .r8Unorm, 0, write: true),
              let sc = plane(sSurf, w/2, h/2, .rg8Unorm, 1, write: true),
              let dl = plane(dSurf, w*2, h*2, .r8Unorm, 0, write: false),
              let dc = plane(dSurf, w, h, .rg8Unorm, 1, write: false) else { return false }

        let od = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w*2, height: h*2, mipmapped: false)
        od.usage = [.shaderRead, .shaderWrite]
        od.storageMode = .private
        guard let ob = device.makeTexture(descriptor: od) else { return false }

        proc = p; srcLuma = sl; srcChroma = sc; dstLuma = dl; dstChroma = dc; outBGRA = ob
        srcFrame = VTFrameProcessorFrame(buffer: sBuf, presentationTimeStamp: .zero)
        dstFrame = VTFrameProcessorFrame(buffer: dBuf, presentationTimeStamp: .zero)
        inW = w; inH = h
        guard srcFrame != nil, dstFrame != nil else { return false }
        logger.info("NeuralUpscaler ready \(w)x\(h) → \(w*2)x\(h*2)")
        return true
    }

    private func makeBuffer(_ attrs: [String: Any]) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        guard let w = attrs["Width"] as? Int, let h = attrs["Height"] as? Int,
              let f = attrs["PixelFormatType"] as? Int else { return nil }
        return CVPixelBufferCreate(nil, w, h, OSType(f), attrs as CFDictionary, &pb) == kCVReturnSuccess ? pb : nil
    }

    private func teardown() {
        if proc != nil { proc?.endSession() }
        proc = nil; srcFrame = nil; dstFrame = nil
        srcLuma = nil; srcChroma = nil; dstLuma = nil; dstChroma = nil; outBGRA = nil
    }

    private func compileShaders() {
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        constant float3 kY = float3(0.2126, 0.7152, 0.0722);
        inline float3 rgbToYCbCr(float3 rgb) {
            float y = dot(rgb, kY);
            float cb = (rgb.b - y) / 1.8556, cr = (rgb.r - y) / 1.5748;
            return float3((16.0 + 219.0*y)/255.0, (128.0 + 224.0*cb)/255.0, (128.0 + 224.0*cr)/255.0);
        }
        inline float3 yCbCrToRGB(float y8, float cb8, float cr8) {
            float y = (y8*255.0 - 16.0)/219.0, cb = (cb8*255.0 - 128.0)/224.0, cr = (cr8*255.0 - 128.0)/224.0;
            float r = y + 1.5748*cr, b = y + 1.8556*cb;
            float g = (y - kY.r*r - kY.b*b) / kY.g;
            return saturate(float3(r,g,b));
        }
        kernel void bgraToP420v(texture2d<float, access::sample> src [[texture(0)]],
                                texture2d<float, access::write> luma [[texture(1)]],
                                texture2d<float, access::write> chroma [[texture(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
            uint lw = luma.get_width(), lh = luma.get_height();
            if (gid.x*2 >= lw || gid.y*2 >= lh) return;
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 dsz = float2(lw, lh);
            float3 rgb[4];
            for (uint i=0;i<4;i++){ uint2 lp = uint2(gid.x*2+(i&1), gid.y*2+(i>>1));
                float2 uv=(float2(lp)+0.5)/dsz; rgb[i]=src.sample(s,uv).rgb;
                luma.write(float4(rgbToYCbCr(rgb[i]).x), lp); }
            float3 avg=(rgb[0]+rgb[1]+rgb[2]+rgb[3])*0.25; float3 c=rgbToYCbCr(avg);
            chroma.write(float4(c.y, c.z, 0, 0), gid);
        }
        kernel void p420vToBGRAup(texture2d<float, access::read> luma [[texture(0)]],
                                  texture2d<float, access::sample> chroma [[texture(1)]],
                                  texture2d<float, access::write> dst [[texture(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
            uint w = dst.get_width(), h = dst.get_height();
            if (gid.x >= w || gid.y >= h) return;
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float y = luma.read(gid).r;
            float2 cc = chroma.sample(s, (float2(gid)+0.5)/float2(w,h)).rg;
            dst.write(float4(yCbCrToRGB(y, cc.x, cc.y), 1.0), gid);
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: nil)
            if let f1 = lib.makeFunction(name: "bgraToP420v") { toYUV = try device.makeComputePipelineState(function: f1) }
            if let f2 = lib.makeFunction(name: "p420vToBGRAup") { toRGB = try device.makeComputePipelineState(function: f2) }
        } catch { logger.error("shader compile: \(error.localizedDescription)"); unsupported = true }
    }
}
