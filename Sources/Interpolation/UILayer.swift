import Foundation
import Metal
import Monitoring

/// **정적 UI 층** — 엔진 밖에서 정적 UI를 떼어내고(추출) 보간 프레임 위에 다시 얹는다(합성).
/// 계획: MacFG_UILayer_Plan.md. 엔진은 UI가 지워진 `clean` 프레임만 받는다.
///
/// - extract: α = mask × still(prev, cur) (강한 마스크는 정지 여부 무관), clean = push-pull 인페인팅(α 자리를 주변 배경으로).
/// - composite: out = mix(interp, layerSource(B 원본), α_B).
public final class UILayer {
    public nonisolated(unsafe) static var enabled = Knob.isSet("MACFG_UILAYER")
    /// 정지 판정 문턱(루마): |cur−prev| < lo면 정지, > hi면 움직임.
    public nonisolated(unsafe) static var stillLo: Float = Float(Knob.string("MACFG_UILSTILLLO") ?? "") ?? 0.02
    public nonisolated(unsafe) static var stillHi: Float = Float(Knob.string("MACFG_UILSTILLHI") ?? "") ?? 0.06
    /// push-pull 단수 (레벨 1 = 소스/2 부터). 6이면 4K에서 1/128까지 — 자막 폭 600px 구멍도 닫힌다.
    public nonisolated(unsafe) static var levels: Int = Int(Knob.string("MACFG_UILLEVELS") ?? "") ?? 6

    private let device: any MTLDevice
    private var alphaPSO: (any MTLComputePipelineState)?
    private var pushPSO: (any MTLComputePipelineState)?
    private var pullPSO: (any MTLComputePipelineState)?
    private var pullFinalPSO: (any MTLComputePipelineState)?
    private var compositePSO: (any MTLComputePipelineState)?
    // 피라미드(rgba16f: rgb=가중 색, a=가중치), 레벨 1..L
    private var pyr: [any MTLTexture] = []
    private var pyrW = 0, pyrH = 0
    // 출력 링: (alpha r8, clean bgra8) × 3 — A/B 두 장 + 다음 추출용 여유
    private var alphaRing: [any MTLTexture] = []
    private var cleanRing: [any MTLTexture] = []
    private var ringIdx = 0

    public init(device: any MTLDevice) { self.device = device }

    public func prepare() async throws {
        let lib = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ n: String) throws -> any MTLComputePipelineState {
            guard let f = lib.makeFunction(name: n) else { throw NSError(domain: "UILayer", code: 1, userInfo: [NSLocalizedDescriptionKey: n]) }
            return try device.makeComputePipelineState(function: f)
        }
        alphaPSO = try pso("uilAlpha"); pushPSO = try pso("uilPush"); pullPSO = try pso("uilPull")
        pullFinalPSO = try pso("uilPullFinal"); compositePSO = try pso("uilComposite")
        DiagnosticLog.shared.log("[UILAYER] prepared still=\(Self.stillLo)/\(Self.stillHi) levels=\(Self.levels)")
    }

    private var cleanFormat: MTLPixelFormat = .bgra8Unorm
    private func ensure(w: Int, h: Int, format: MTLPixelFormat) {
        guard w != pyrW || h != pyrH || alphaRing.isEmpty || format != cleanFormat else { return }
        cleanFormat = format
        pyrW = w; pyrH = h
        pyr = []
        var lw = max(1, w / 2), lh = max(1, h / 2)
        for _ in 0..<max(1, Self.levels) {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: lw, height: lh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
            if let t = device.makeTexture(descriptor: d) { pyr.append(t) }
            lw = max(1, lw / 2); lh = max(1, lh / 2)
        }
        alphaRing = []; cleanRing = []
        for _ in 0..<3 {
            let da = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
            da.usage = [.shaderRead, .shaderWrite]; da.storageMode = .private
            // clean은 소스와 같은 포맷 — sRGB 소스(벤치 PNG)를 non-sRGB에 쓰면 바이트 의미가 달라져 바이트를 읽는 엔진(RIFE/VT)이 다른 밝기를 본다.
            let dc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            dc.usage = [.shaderRead, .shaderWrite]; dc.storageMode = .private
            if let a = device.makeTexture(descriptor: da), let c = device.makeTexture(descriptor: dc) { alphaRing.append(a); cleanRing.append(c) }
        }
    }

    private func dispatch(_ enc: any MTLComputeCommandEncoder, _ w: Int, _ h: Int, _ pso: any MTLComputePipelineState) {
        enc.setComputePipelineState(pso)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
    }

    /// 소스 한 장에서 (α, clean)을 만든다. prev가 nil이면 still=1(첫 프레임). mask는 디텍터 출력(소스 좌표 UV, 아무 해상도).
    /// 반환 텍스처는 링(3장)이라 다음 두 번의 extract까지 유효.
    public func extract(source: any MTLTexture, prev: (any MTLTexture)?, mask: any MTLTexture,
                        into cb: any MTLCommandBuffer) -> (alpha: any MTLTexture, clean: any MTLTexture)? {
        guard let alphaPSO, let pushPSO, let pullPSO, let pullFinalPSO else { return nil }
        ensure(w: source.width, h: source.height, format: source.pixelFormat)
        guard !pyr.isEmpty, alphaRing.count == 3, let enc = cb.makeComputeCommandEncoder() else { return nil }
        let alpha = alphaRing[ringIdx], clean = cleanRing[ringIdx]
        ringIdx = (ringIdx + 1) % 3
        // 1) α
        enc.setTexture(source, index: 0); enc.setTexture(prev ?? source, index: 1); enc.setTexture(mask, index: 2); enc.setTexture(alpha, index: 3)
        var p = SIMD4<Float>(Self.stillLo, Self.stillHi, prev == nil ? 0 : 1, 0)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        dispatch(enc, source.width, source.height, alphaPSO)
        // 2) push: 레벨 1은 소스+α에서, 그 아래는 상위 레벨에서
        enc.setTexture(source, index: 0); enc.setTexture(alpha, index: 1); enc.setTexture(pyr[0], index: 2)
        var fromSource: Int32 = 1
        enc.setBytes(&fromSource, length: 4, index: 0)
        dispatch(enc, pyr[0].width, pyr[0].height, pushPSO)
        fromSource = 0
        for l in 1..<pyr.count {
            enc.setTexture(pyr[l - 1], index: 0); enc.setTexture(alpha, index: 1); enc.setTexture(pyr[l], index: 2)
            enc.setBytes(&fromSource, length: 4, index: 0)
            dispatch(enc, pyr[l].width, pyr[l].height, pushPSO)
        }
        // 3) pull: 거친 레벨의 색으로 가중치 부족한 픽셀을 채우며 올라온다 (제자리 갱신: 레벨 l을 l+1로 보완)
        if pyr.count >= 2 {
            for l in stride(from: pyr.count - 2, through: 0, by: -1) {
                enc.setTexture(pyr[l + 1], index: 0); enc.setTexture(pyr[l], index: 1)
                dispatch(enc, pyr[l].width, pyr[l].height, pullPSO)
            }
        }
        // 4) 최종: clean = α>0 ? 레벨1 채움 : 소스
        enc.setTexture(source, index: 0); enc.setTexture(alpha, index: 1); enc.setTexture(pyr[0], index: 2); enc.setTexture(clean, index: 3)
        dispatch(enc, source.width, source.height, pullFinalPSO)
        enc.endEncoding()
        return (alpha, clean)
    }

    /// out = mix(interp, mix(A, B, t), hole(α)). UI 층은 A·B의 t 블렌드 — 정지 UI면 동일, 게이지·타이머처럼 천천히
    /// 변하는 UI면 중간 프레임에 더 가깝다(엔진 내부 nearestPix/srcBlend와 같은 규약; B만 붙이면 삼중항 PSNR −1dB).
    public func composite(interp: any MTLTexture, sourceA: any MTLTexture, sourceB: any MTLTexture, t: Float, alpha: any MTLTexture,
                          into cb: any MTLCommandBuffer, dst: any MTLTexture) {
        guard let compositePSO, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setTexture(interp, index: 0); enc.setTexture(sourceA, index: 1); enc.setTexture(sourceB, index: 2)
        enc.setTexture(alpha, index: 3); enc.setTexture(dst, index: 4)
        var tt = t
        enc.setBytes(&tt, length: MemoryLayout<Float>.size, index: 0)
        dispatch(enc, dst.width, dst.height, compositePSO)
        enc.endEncoding()
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    constant half3 kLum = half3(0.299h, 0.587h, 0.114h);
    // 구멍/합성 공용 판정: 픽셀 자신의 α에 경계값. 팽창은 쓰지 않는다 — 글자 옆 움직이는 배경 2px가 구멍에 들어가
    // 얼어붙는다(실측: 띠 30→54%). 글자 안티앨리어스 가장자리는 자체 마스크가 높고 정지라 이미 구멍에 든다.
    static inline half uilHole(texture2d<half, access::sample> alpha, sampler s, float2 uv, uint w, uint h) {
        return smoothstep(0.25h, 0.35h, alpha.sample(s, uv).r);
    }

    // α = mask × mix(still, 1, smoothstep(0.5, 0.9, mask)); still = 1 − smoothstep(lo, hi, |luma(cur) − luma(prev)|)
    kernel void uilAlpha(texture2d<half, access::sample> cur [[texture(0)]],
                         texture2d<half, access::sample> prev [[texture(1)]],
                         texture2d<float, access::sample> mask [[texture(2)]],
                         texture2d<half, access::write> alpha [[texture(3)]],
                         constant float4& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
        uint w = alpha.get_width(), h = alpha.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float m = clamp(mask.sample(s, uv).r, 0.0, 1.0);
        float still = 1.0;
        if (p.z > 0.5) {
            float d = fabs(float(dot(cur.sample(s, uv).rgb, kLum) - dot(prev.sample(s, uv).rgb, kLum)));
            still = 1.0 - smoothstep(p.x, p.y, d);
        }
        float a = m * mix(still, 1.0, smoothstep(0.5, 0.9, m));
        alpha.write(half4(half(a)), gid);
    }

    // push: 2×2 가중 평균. fromSource=1이면 소스(bgra8)+α에서 (w = 1−α), 아니면 상위 피라미드(rgba16f: rgb 가중색, a 가중치)에서.
    kernel void uilPush(texture2d<half, access::sample> src [[texture(0)]],
                        texture2d<half, access::sample> alpha [[texture(1)]],
                        texture2d<half, access::write> dst [[texture(2)]],
                        constant int& fromSource [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::nearest, address::clamp_to_edge);
        float2 uvc = (float2(gid) + 0.5) / float2(w, h);
        float2 e = 0.25 / float2(w, h);
        half3 acc = half3(0.0h); half wsum = 0.0h;
        for (int j = 0; j < 4; j++) {
            float2 uv = uvc + float2((j & 1) ? e.x : -e.x, (j & 2) ? e.y : -e.y);
            if (fromSource != 0) {
                // 구멍 = 팽창된 경계 α (uilHole). 부분 가중치로 넣으면 글자색이 채움에 새어 안개가 되고(분홍 haze),
                // 낮은 문턱을 쓰면 정적 장면의 마스크 바닥까지 구멍이 돼 화면 대부분이 흐려진다(실측 PSNR −2.5dB).
                half wt = 1.0h - uilHole(alpha, s, uv, alpha.get_width(), alpha.get_height());
                acc += src.sample(s, uv).rgb * wt; wsum += wt;
            } else {
                half4 v = src.sample(s, uv);   // rgb = 정규화 색, a = 가중치
                acc += v.rgb * v.a; wsum += v.a;
            }
        }
        // 저장은 (정규화 색, 가중치) — 미정규화 저장은 거친 레벨의 미세 가중치에서 half 나눗셈이 터져 흰색이 됐다(실측).
        // 가중치가 너무 작으면(0.05 미만) 구멍으로 취급해 잡음 전파를 막는다.
        half wn = (wsum < 0.05h) ? 0.0h : min(wsum / 4.0h, 1.0h);
        half3 c = (wsum > 0.0h) ? (acc / wsum) : half3(0.0h);
        dst.write(half4(c, wn), gid);
    }

    // pull: 가중치 부족한 픽셀을 거친 레벨 색으로 보완 (제자리). out = c_f + (1−w_f)·c_coarse(정규화)
    kernel void uilPull(texture2d<half, access::sample> coarse [[texture(0)]],
                        texture2d<half, access::read_write> fine [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
        uint w = fine.get_width(), h = fine.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        half4 f = fine.read(gid);
        half4 c = coarse.sample(s, uv);      // 정규화 색 (pull을 거친 레벨은 a=1)
        half3 outc = (c.a > 0.0h) ? mix(c.rgb, f.rgb, f.a) : f.rgb;
        half outw = max(f.a, min(c.a, 1.0h)) > 0.0h ? 1.0h : 0.0h;
        fine.write(half4(outc, outw), gid);
    }

    // 최종: clean = mix(source, fill, α) — fill은 레벨1의 정규화 색
    kernel void uilPullFinal(texture2d<half, access::sample> src [[texture(0)]],
                             texture2d<half, access::sample> alpha [[texture(1)]],
                             texture2d<half, access::sample> lvl1 [[texture(2)]],
                             texture2d<half, access::write> clean [[texture(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint w = clean.get_width(), h = clean.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        half4 sv = src.sample(s, uv);
        half4 l = lvl1.sample(s, uv);
        half3 fill = (l.a > 0.0h) ? l.rgb : sv.rgb;
        half k = uilHole(alpha, s, uv, alpha.get_width(), alpha.get_height());
        clean.write(half4(mix(sv.rgb, fill, k), 1.0h), gid);
    }

    kernel void uilComposite(texture2d<half, access::sample> interp [[texture(0)]],
                             texture2d<half, access::sample> srcA [[texture(1)]],
                             texture2d<half, access::sample> srcB [[texture(2)]],
                             texture2d<half, access::sample> alpha [[texture(3)]],
                             texture2d<half, access::write> dst [[texture(4)]],
                             constant float& t [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        half a = uilHole(alpha, s, uv, alpha.get_width(), alpha.get_height());   // 채운 픽셀은 전부 UI 층으로 덮는다
        half3 layer = mix(srcA.sample(s, uv).rgb, srcB.sample(s, uv).rgb, half(t));
        half3 o = mix(interp.sample(s, uv).rgb, layer, a);
        dst.write(half4(o, 1.0h), gid);
    }
    """
}
