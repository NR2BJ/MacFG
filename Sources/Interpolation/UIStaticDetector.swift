@preconcurrency import Metal
import Monitoring

/// 시간축 화면정지 UI 검출 (채팅/HUD/오버레이). 오프라인 스파이크(temporal_ui_spike.py)에서
/// 검증된 '일관성 신호'를 실시간 구현: 화면고정 위치에 지속되는 구조(고주파의 시간적 일관성이
/// 높음)는 UI → 마스크로 표시해 엔진 워프가 소스로 프리즈하게 한다. 이동 배경은 고주파가
/// 시간적으로 비일관이라 낮게 남아 제외. 흐린(반투명) 텍스트도 정지면 잡힌다.
///
/// cons(p) = |EMA(hp)| / sqrt(EMA(hp²) - EMA(hp)² + eps),  hp = luma - 3x3box(luma)
/// mask = smoothstep(clo, chi, cons).  누적은 EMA(창 ~1/alpha 프레임).
/// 소스 좌표계 누적 — 캡처 창 콘텐츠는 화면정렬이라 UI는 고정, 콘텐츠만 이동. reset은
/// 캡처 시작/해상도 변경/장면 전환 시.
public final class UIStaticDetector {
    private let device: any MTLDevice
    private var pso: (any MTLComputePipelineState)?
    private var meanTex: [any MTLTexture] = []   // ping-pong EMA(hp)
    private var sqTex: [any MTLTexture] = []      // ping-pong EMA(hp²)
    private var maskTex: (any MTLTexture)?
    private var cur = 0
    private var w = 0, h = 0
    private var needsReset = true
    private var frames = 0

    /// 튜닝(오프라인 clo0.8 chi2.0 시작점). alpha=EMA율(~1/창길이). enabled=off면 no-op.
    public nonisolated(unsafe) static var enabled = true
    public nonisolated(unsafe) static var alpha: Float = 0.04   // ~25프레임 창 (스윕 최적)
    public nonisolated(unsafe) static var clo: Float = 0.5      // 스윕 최적 — 흐린 UI까지 잡되 무회귀
    public nonisolated(unsafe) static var chi: Float = 1.7
    public nonisolated(unsafe) static var strength: Float = 1.0  // 마스크 최대 프리즈 강도

    /// 워프가 샘플할 마스크 (없으면 nil). 워밍업 전(<8프레임)엔 nil 반환해 초기 노이즈 회피.
    public var mask: (any MTLTexture)? { (Self.enabled && frames >= 8) ? maskTex : nil }

    public init(device: any MTLDevice) { self.device = device }

    public func prepare() async throws {
        let lib = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let fn = lib.makeFunction(name: "uiStaticUpdate") else { throw InterpolationError.shaderCompilationFailed }
        pso = try await device.makeComputePipelineState(function: fn)
    }

    /// 해상도 세팅/재세팅 — 소스 크기의 1/2(텍스트 보존 + 저비용).
    private func ensure(srcW: Int, srcH: Int) {
        let mw = max(64, srcW / 2), mh = max(64, srcH / 2)
        guard mw != w || mh != h || maskTex == nil else { return }
        w = mw; h = mh
        func tex(_ fmt: MTLPixelFormat, _ usage: MTLTextureUsage) -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: mw, height: mh, mipmapped: false)
            d.usage = usage; d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        meanTex = [tex(.r16Float, [.shaderRead, .shaderWrite]), tex(.r16Float, [.shaderRead, .shaderWrite])].compactMap { $0 }
        sqTex = [tex(.r16Float, [.shaderRead, .shaderWrite]), tex(.r16Float, [.shaderRead, .shaderWrite])].compactMap { $0 }
        maskTex = tex(.r16Float, [.shaderRead, .shaderWrite])
        needsReset = true
    }

    /// 불연속(캡처 시작/장면 전환) — 다음 업데이트에서 누적 리셋.
    public func reset() { needsReset = true; frames = 0 }

    /// 현재 소스 프레임으로 누적 갱신 + 마스크 산출. cb에 인코딩 (blit 직후 = 소스 준비됨).
    public func update(source: any MTLTexture, into cb: any MTLCommandBuffer) {
        guard Self.enabled, let pso else { return }
        ensure(srcW: source.width, srcH: source.height)
        guard meanTex.count == 2, sqTex.count == 2, let maskTex,
              let enc = cb.makeComputeCommandEncoder() else { return }
        let prev = cur, next = 1 - cur
        enc.setComputePipelineState(pso)
        enc.setTexture(source, index: 0)
        enc.setTexture(meanTex[prev], index: 1)
        enc.setTexture(sqTex[prev], index: 2)
        enc.setTexture(meanTex[next], index: 3)
        enc.setTexture(sqTex[next], index: 4)
        enc.setTexture(maskTex, index: 5)
        var p = SIMD4<Float>(Self.alpha, Self.clo, Self.chi, needsReset ? 1 : 0)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        var strength = Self.strength
        enc.setBytes(&strength, length: MemoryLayout<Float>.size, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
        cur = next
        needsReset = false
        frames += 1
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void uiStaticUpdate(
        texture2d<half, access::sample> src   [[texture(0)]],
        texture2d<float, access::read>  meanIn [[texture(1)]],
        texture2d<float, access::read>  sqIn   [[texture(2)]],
        texture2d<float, access::write> meanOut [[texture(3)]],
        texture2d<float, access::write> sqOut   [[texture(4)]],
        texture2d<float, access::write> maskOut [[texture(5)]],
        constant float4& p [[buffer(0)]],       // alpha, clo, chi, reset
        constant float&  strength [[buffer(1)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint w = maskOut.get_width(), h = maskOut.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 sz = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / sz;
        float2 e = 1.0 / sz;
        const half3 kL = half3(0.299h, 0.587h, 0.114h);
        // 3x3 박스 대비 고주파 (마스크 해상도 = 소스 1/2, bilinear 자동 다운샘플)
        half lc = dot(src.sample(s, uv).rgb, kL);
        half acc = 0.0h;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++)
            acc += dot(src.sample(s, uv + float2(float(dx), float(dy)) * e).rgb, kL);
        float hp = float(lc - acc / 9.0h);       // 고주파 (구조=큼, 평탄=0)
        float a = p.x;
        float m0, s0;
        if (p.w > 0.5) { m0 = hp; s0 = hp * hp; }           // reset
        else {
            m0 = mix(meanIn.read(gid).r, hp, a);
            s0 = mix(sqIn.read(gid).r, hp * hp, a);
        }
        meanOut.write(float4(m0), gid);
        sqOut.write(float4(s0), gid);
        float var0 = max(s0 - m0 * m0, 0.0);
        float cons = fabs(m0) / (sqrt(var0) + 0.004);        // 시간적 일관성 (정지구조=큼)
        // 구조 게이트: 고주파 크기가 너무 작으면(평탄 배경) UI 아님 — 오검출 방지
        float structured = smoothstep(0.004, 0.02, fabs(m0));
        float m = smoothstep(p.y, p.z, cons) * structured * strength;
        maskOut.write(float4(m), gid);
    }
    """
}
