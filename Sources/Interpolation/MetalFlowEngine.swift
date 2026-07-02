@preconcurrency import Metal
import Monitoring
import os

/// LSFG(Lossless Scaling) 방식의 순수 GPU 보간 엔진 — NN 없음.
///
/// 핵심 통찰: flow 해상도 ≠ 출력 선명도. flow는 저해상도 피라미드에서 추정하고,
/// 워프는 풀해상도 원본 픽셀을 이동시키므로 출력은 원본 선명도를 유지한다.
/// (RIFE류는 flow 해상도에서 이미지를 "합성"하므로 해상도가 화질을 결정 — 그래서 무거움)
///
/// 파이프라인 (전부 Metal 컴퓨트, 단일 CB, FidelityFX Optical Flow 계열):
///  1. 루마 피라미드: base(≤1920 긴변, 소스 1/2) → 반씩 다운 ~7레벨
///  2. coarse→fine 매칭: 최상위 ±3, 이하 ±1 국소 탐색 (5x5 SAD), 양방향(A→B, B→A)
///     + 레벨마다 3x3 flow 스무딩 (노이즈/부들거림 억제)
///  3. finalize: forward-backward 일관성 → 신뢰도 마스크, |A-B| → 정적 마스크,
///     루마 히스토그램(장면컷 판정, AppleFI와 동일 방식)
///  4. 풀해상도 워프: w0=A(p-t·F_ab), w1=B(p+(1-t)·F_ba) → 신뢰도 기반 합성.
///     정적 영역=B 원본(선명도 유지), 저신뢰 영역=크로스페이드(아티팩트 대신 소프트).
///
/// 임의 t 지원: flow 벡터 t배 스케일이므로 어떤 위상이든 동일 비용 (144Hz 대응 기반).
public final class MetalFlowEngine: PairInterpolationEngine {
    public let name = "Metal Flow (GPU)"

    // 장면 전환 판정 (AppleFI와 동일)
    private let sceneCutIntersectionThreshold = 0.5

    /// flow 밀도 (긴 변 목표). 앱 시작 시 1회 설정(--flow-base) 후 읽기 전용 — 락 불필요.
    /// 4K 실측 (M4, 2026-07-02): 480→work 8ms / 960→12-13ms / 1280→16-18ms / 1920→40ms(붕괴).
    /// 960 = 60fps 예산(16.7ms) 내 최대 밀도 — 기본값. 1280은 30fps 이하 콘텐츠용 여지.
    public nonisolated(unsafe) static var flowBaseLongSide: Double = 960

    private var device: (any MTLDevice)?
    private let logger = Logger(subsystem: "com.macfg", category: "MetalFlow")

    // PSO
    private var downLumaPSO: (any MTLComputePipelineState)?
    private var downHalfPSO: (any MTLComputePipelineState)?
    private var matchPSO: (any MTLComputePipelineState)?
    private var smoothPSO: (any MTLComputePipelineState)?
    private var finalizePSO: (any MTLComputePipelineState)?
    private var warpPSO: (any MTLComputePipelineState)?

    // 피라미드 리소스 (소스 크기 의존)
    private var srcWidth = 0
    private var srcHeight = 0
    private var levels: [(w: Int, h: Int)] = []
    private var lumaA: [any MTLTexture] = []
    private var lumaB: [any MTLTexture] = []
    private var flowF: [any MTLTexture] = []   // A→B
    private var flowB: [any MTLTexture] = []   // B→A
    private var flowTmp: [any MTLTexture] = [] // 스무딩 핑퐁
    private var maskTex: (any MTLTexture)?     // base res: r=신뢰도, g=정적
    // 출력 링: 갭 채움으로 쌍당 최대 4장 → 넉넉히 12 (타임라인 cap 12와 정합)
    private var outputPool: [any MTLTexture] = []
    private var outputIndex = 0
    // 히스토그램은 쌍당 1개 — 별도 링
    private var statsBuffers: [any MTLBuffer] = []
    private var statsIndex = 0

    private var pairCount = 0

    public init() {}

    // MARK: - Prepare

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ n: String) async throws -> any MTLComputePipelineState {
            guard let f = library.makeFunction(name: n) else { throw InterpolationError.notPrepared }
            return try await device.makeComputePipelineState(function: f)
        }
        downLumaPSO = try await pso("mfDownLuma")
        downHalfPSO = try await pso("mfDownHalf")
        matchPSO = try await pso("mfMatch")
        smoothPSO = try await pso("mfSmooth")
        finalizePSO = try await pso("mfFinalize")
        warpPSO = try await pso("mfWarp")
        DiagnosticLog.shared.log("[MetalFlow] prepared (pyramid flow + full-res warp)")
    }

    // MARK: - Encode

    public func encodePair(
        stableA: any MTLTexture,
        stableB: any MTLTexture,
        tsA: CFTimeInterval,
        tsB: CFTimeInterval,
        tValues: [Float],
        into commandBuffer: any MTLCommandBuffer
    ) -> PairEncodeResult? {
        guard let downLumaPSO, let downHalfPSO, let matchPSO, let smoothPSO,
              let finalizePSO, let warpPSO,
              tsB > tsA, !tValues.isEmpty else { return nil }

        ensureResources(width: stableB.width, height: stableB.height)
        guard !levels.isEmpty, outputPool.count >= tValues.count, let maskTex,
              !statsBuffers.isEmpty else { return nil }

        let statsBuffer = statsBuffers[statsIndex]
        statsIndex = (statsIndex + 1) % statsBuffers.count
        let L = levels.count

        // 0) 히스토그램 클리어
        if let fill = commandBuffer.makeBlitCommandEncoder() {
            fill.fill(buffer: statsBuffer, range: 0..<(64 * 4), value: 0)
            fill.endEncoding()
        }

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return nil }

        // 1) 루마 피라미드
        enc.setComputePipelineState(downLumaPSO)
        enc.setTexture(stableA, index: 0); enc.setTexture(lumaA[0], index: 1)
        dispatch(enc, levels[0].w, levels[0].h, downLumaPSO)
        enc.setTexture(stableB, index: 0); enc.setTexture(lumaB[0], index: 1)
        dispatch(enc, levels[0].w, levels[0].h, downLumaPSO)
        enc.setComputePipelineState(downHalfPSO)
        for l in 1..<L {
            enc.setTexture(lumaA[l - 1], index: 0); enc.setTexture(lumaA[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, downHalfPSO)
            enc.setTexture(lumaB[l - 1], index: 0); enc.setTexture(lumaB[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, downHalfPSO)
        }

        // 2) coarse→fine 매칭 (양방향) + 스무딩
        for l in stride(from: L - 1, through: 0, by: -1) {
            var params = MatchParams(
                searchRadius: l == L - 1 ? 3 : 1,
                hasPrior: l == L - 1 ? 0 : 1,
                refine: l == 0 ? 1 : 0  // 서브픽셀 refine은 최종 레벨만 (비용 40%↓)
            )
            // forward: A→B
            enc.setComputePipelineState(matchPSO)
            enc.setTexture(lumaA[l], index: 0)
            enc.setTexture(lumaB[l], index: 1)
            enc.setTexture(l == L - 1 ? flowF[l] : flowF[l + 1], index: 2) // prior (coarser)
            enc.setTexture(flowTmp[l], index: 3)
            enc.setBytes(&params, length: MemoryLayout<MatchParams>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, matchPSO)
            enc.setComputePipelineState(smoothPSO)
            enc.setTexture(flowTmp[l], index: 0); enc.setTexture(flowF[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, smoothPSO)
            // backward: B→A
            enc.setComputePipelineState(matchPSO)
            enc.setTexture(lumaB[l], index: 0)
            enc.setTexture(lumaA[l], index: 1)
            enc.setTexture(l == L - 1 ? flowB[l] : flowB[l + 1], index: 2)
            enc.setTexture(flowTmp[l], index: 3)
            enc.setBytes(&params, length: MemoryLayout<MatchParams>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, matchPSO)
            enc.setComputePipelineState(smoothPSO)
            enc.setTexture(flowTmp[l], index: 0); enc.setTexture(flowB[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, smoothPSO)
        }

        // 3) finalize: 신뢰도/정적 마스크 + 히스토그램
        enc.setComputePipelineState(finalizePSO)
        enc.setTexture(lumaA[0], index: 0)
        enc.setTexture(lumaB[0], index: 1)
        enc.setTexture(flowF[0], index: 2)
        enc.setTexture(flowB[0], index: 3)
        enc.setTexture(maskTex, index: 4)
        enc.setBuffer(statsBuffer, offset: 0, index: 0)
        dispatch(enc, levels[0].w, levels[0].h, finalizePSO)

        // 4) 풀해상도 워프 + 합성 — flow는 한 번 계산, t별로 워프만 반복 (장당 ~1ms)
        // 이게 갭 채움(드랍된 소스 프레임 자리 메꾸기)을 싸게 만드는 핵심.
        var frames: [(t: Float, texture: any MTLTexture)] = []
        enc.setComputePipelineState(warpPSO)
        enc.setTexture(stableA, index: 0)
        enc.setTexture(stableB, index: 1)
        enc.setTexture(flowF[0], index: 2)
        enc.setTexture(flowB[0], index: 3)
        enc.setTexture(maskTex, index: 4)
        for t in tValues.sorted() {
            let output = outputPool[outputIndex]
            outputIndex = (outputIndex + 1) % outputPool.count
            var wp = WarpParams(t: t)
            enc.setTexture(output, index: 5)
            enc.setBytes(&wp, length: MemoryLayout<WarpParams>.stride, index: 0)
            dispatch(enc, output.width, output.height, warpPSO)
            frames.append((t, output))
        }
        enc.endEncoding()

        pairCount += 1
        if pairCount <= 3 || pairCount % 600 == 0 {
            DiagnosticLog.shared.log("[MetalFlow] pair #\(pairCount) (\(stableB.width)x\(stableB.height), base=\(levels[0].w)x\(levels[0].h), levels=\(L), tCount=\(tValues.count))")
        }

        let threshold = sceneCutIntersectionThreshold
        let evaluator: @Sendable () -> Bool = {
            let ptr = statsBuffer.contents().bindMemory(to: UInt32.self, capacity: 64)
            var totalA: UInt64 = 0
            var intersect: UInt64 = 0
            for i in 0..<32 {
                let a = UInt64(ptr[i]); let b = UInt64(ptr[32 + i])
                totalA += a
                intersect += min(a, b)
            }
            guard totalA > 1000 else { return false }
            return Double(intersect) / Double(totalA) < threshold
        }
        return PairEncodeResult(frames: frames, sceneCutEvaluator: evaluator)
    }

    private struct MatchParams {
        var searchRadius: Int32
        var hasPrior: Int32
        var refine: Int32
    }
    private struct WarpParams {
        var t: Float
    }

    private func dispatch(_ enc: any MTLComputeCommandEncoder, _ w: Int, _ h: Int, _ pso: any MTLComputePipelineState) {
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg
        )
    }

    // MARK: - Resources

    private func ensureResources(width: Int, height: Int) {
        guard let device else { return }
        guard width != srcWidth || height != srcHeight || levels.isEmpty else { return }
        srcWidth = width
        srcHeight = height

        // flow base: 긴 변 기준 flow 밀도 (이미지가 아니라 모션 지도의 해상도 —
        // 워프는 항상 풀해상도 원본 픽셀이므로 출력 선명도와 무관, 모션 경계 정밀도만 좌우).
        // 픽셀당 5x5 SAD 매칭이라 base²가 비용 지배: 1900 실측 77ms / 480 실측 ~3ms.
        let longSide = max(width, height)
        let s = min(1.0, Self.flowBaseLongSide / Double(longSide))
        var bw = Int(Double(width) * s), bh = Int(Double(height) * s)
        bw = max(bw & ~1, 64); bh = max(bh & ~1, 64)

        levels = []
        var w = bw, h = bh
        while w >= 15 && h >= 15 && levels.count < 7 {
            levels.append((w, h))
            w /= 2; h /= 2
        }

        func tex(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat, usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }

        lumaA = []; lumaB = []; flowF = []; flowB = []; flowTmp = []
        for (w, h) in levels {
            guard let la = tex(w, h, .r16Float), let lb = tex(w, h, .r16Float),
                  let ff = tex(w, h, .rg16Float), let fb = tex(w, h, .rg16Float),
                  let ft = tex(w, h, .rg16Float) else { levels = []; return }
            lumaA.append(la); lumaB.append(lb)
            flowF.append(ff); flowB.append(fb); flowTmp.append(ft)
        }
        maskTex = tex(levels[0].w, levels[0].h, .rg8Unorm)

        outputPool = []; statsBuffers = []; outputIndex = 0; statsIndex = 0
        // 출력 풀 크기를 바이트 예산으로 스케일 — 16MP(5120x3140) 캡처에서 12×64MB=770MB
        // 메모리 압박 실측. 예산 ~420MB: 4K→12장, 16MP→6장 (갭 채움 최소 요구 충족)
        let bytesPerTexture = max(width * height * 4, 1)
        let poolCount = min(12, max(6, (420 << 20) / bytesPerTexture))
        for _ in 0..<poolCount {
            if let t = tex(width, height, .bgra8Unorm) {
                outputPool.append(t)
            }
        }
        for _ in 0..<4 {
            if let b = device.makeBuffer(length: 64 * 4, options: .storageModeShared) {
                statsBuffers.append(b)
            }
        }
        DiagnosticLog.shared.log("[MetalFlow] resources: src=\(width)x\(height) base=\(bw)x\(bh) levels=\(levels.count)")
    }

    public func reset() {}

    public func shutdown() {
        levels = []
        lumaA = []; lumaB = []; flowF = []; flowB = []; flowTmp = []
        maskTex = nil
        outputPool = []; statsBuffers = []
    }

    // MARK: - Shaders

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct MatchParams { int searchRadius; int hasPrior; int refine; };
    struct WarpParams { float t; };

    constant float3 kLuma = float3(0.2126, 0.7152, 0.0722);

    // BGRA 소스 → base 루마 (박스 다운샘플)
    kernel void mfDownLuma(
        texture2d<float, access::sample> src [[texture(0)]],
        texture2d<float, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float2 o = 0.25 / float2(w, h);
        float l = dot(src.sample(s, uv + float2(-o.x,-o.y)).rgb, kLuma)
                + dot(src.sample(s, uv + float2( o.x,-o.y)).rgb, kLuma)
                + dot(src.sample(s, uv + float2(-o.x, o.y)).rgb, kLuma)
                + dot(src.sample(s, uv + float2( o.x, o.y)).rgb, kLuma);
        dst.write(float4(l * 0.25), gid);
    }

    // 루마 반 다운
    kernel void mfDownHalf(
        texture2d<float, access::sample> src [[texture(0)]],
        texture2d<float, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        dst.write(float4(src.sample(s, uv).r), gid);
    }

    // 피라미드 매칭: prior(코스 레벨) flow*2를 시작점으로 ±searchRadius 국소 탐색 (5x5 SAD)
    // flow 단위: 이 레벨의 픽셀
    kernel void mfMatch(
        texture2d<float, access::sample> src [[texture(0)]],   // 기준 프레임 루마
        texture2d<float, access::sample> ref [[texture(1)]],   // 대상 프레임 루마
        texture2d<float, access::sample> prior [[texture(2)]], // 코스 flow (없으면 미사용)
        texture2d<float, access::write> flowOut [[texture(3)]],
        constant MatchParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = flowOut.get_width(), h = flowOut.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 size = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / size;

        float2 base = float2(0.0);
        if (p.hasPrior != 0) {
            base = prior.sample(s, uv).rg * 2.0;  // 코스 레벨 픽셀 단위 → 이 레벨 단위
        }

        // 5x5 창 소스 패치 캐시
        float patch[25];
        int k = 0;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                float2 q = (float2(gid) + 0.5 + float2(dx, dy)) / size;
                patch[k++] = src.sample(s, q).r;
            }

        float bestSAD = 1e9;
        float2 bestOff = float2(0.0);
        for (int oy = -p.searchRadius; oy <= p.searchRadius; oy++) {
            for (int ox = -p.searchRadius; ox <= p.searchRadius; ox++) {
                float2 cand = base + float2(ox, oy);
                float sad = 0.0;
                int j = 0;
                for (int dy = -2; dy <= 2; dy++)
                    for (int dx = -2; dx <= 2; dx++) {
                        float2 q = (float2(gid) + 0.5 + cand + float2(dx, dy)) / size;
                        sad += fabs(ref.sample(s, q).r - patch[j++]);
                    }
                if (sad < bestSAD) { bestSAD = sad; bestOff = cand; }
            }
        }

        // 서브픽셀 refine (x/y 독립 3점 포물선) — 최종 레벨만
        float2 refined = bestOff;
        if (p.refine != 0) {
            float sadC = bestSAD;
            // x
            float sadL = 0.0, sadR = 0.0;
            int j = 0;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    float2 qL = (float2(gid) + 0.5 + bestOff + float2(dx - 1, dy)) / size;
                    float2 qR = (float2(gid) + 0.5 + bestOff + float2(dx + 1, dy)) / size;
                    float pv = patch[j++];
                    sadL += fabs(ref.sample(s, qL).r - pv);
                    sadR += fabs(ref.sample(s, qR).r - pv);
                }
            float denomX = sadL - 2.0 * sadC + sadR;
            if (denomX > 1e-5) refined.x += clamp(0.5 * (sadL - sadR) / denomX, -0.5, 0.5);
            // y
            float sadU = 0.0, sadD = 0.0;
            j = 0;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    float2 qU = (float2(gid) + 0.5 + bestOff + float2(dx, dy - 1)) / size;
                    float2 qD = (float2(gid) + 0.5 + bestOff + float2(dx, dy + 1)) / size;
                    float pv = patch[j++];
                    sadU += fabs(ref.sample(s, qU).r - pv);
                    sadD += fabs(ref.sample(s, qD).r - pv);
                }
            float denomY = sadU - 2.0 * sadC + sadD;
            if (denomY > 1e-5) refined.y += clamp(0.5 * (sadU - sadD) / denomY, -0.5, 0.5);
        }

        flowOut.write(float4(refined, 0, 0), gid);
    }

    // 3x3 flow 스무딩 (노이즈로 인한 정적 영역 부들거림 억제)
    kernel void mfSmooth(
        texture2d<float, access::sample> src [[texture(0)]],
        texture2d<float, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 size = float2(w, h);
        float2 acc = float2(0.0);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                float2 uv = (float2(gid) + 0.5 + float2(dx, dy)) / size;
                acc += src.sample(s, uv).rg;
            }
        dst.write(float4(acc / 9.0, 0, 0), gid);
    }

    // 신뢰도(순환 일관성) + 정적(|A-B|) 마스크 + 루마 히스토그램 (장면컷)
    kernel void mfFinalize(
        texture2d<float, access::read> lumA [[texture(0)]],
        texture2d<float, access::read> lumB [[texture(1)]],
        texture2d<float, access::sample> flowF [[texture(2)]],
        texture2d<float, access::sample> flowB [[texture(3)]],
        texture2d<float, access::write> mask [[texture(4)]],
        device atomic_uint* hist [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = mask.get_width(), h = mask.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 size = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / size;

        float2 f = flowF.sample(s, uv).rg;
        // 순환 검사: x + F(x) 위치의 backward flow와 합이 0이어야 일관
        float2 uv2 = (float2(gid) + 0.5 + f) / size;
        float2 b = flowB.sample(s, uv2).rg;
        float cyc = length(f + b);
        float conf = 1.0 - smoothstep(1.0, 4.0, cyc);  // 픽셀 단위 오차 1~4 사이 감쇠

        float la = lumA.read(gid).r;
        float lb = lumB.read(gid).r;
        float d = fabs(la - lb);
        float staticness = 1.0 - smoothstep(0.008, 0.04, d);

        mask.write(float4(conf, staticness, 0, 0), gid);

        if ((gid.x & 3) == 0 && (gid.y & 3) == 0) {
            uint binA = uint(clamp(la, 0.0, 0.999) * 32.0);
            uint binB = uint(clamp(lb, 0.0, 0.999) * 32.0);
            atomic_fetch_add_explicit(&hist[binA], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&hist[32u + binB], 1u, memory_order_relaxed);
        }
    }

    // 풀해상도 양방향 워프 + 합성.
    // flow는 base-res 픽셀 단위 → 풀해상도 UV 오프셋으로 변환해 샘플.
    kernel void mfWarp(
        texture2d<float, access::sample> imgA [[texture(0)]],
        texture2d<float, access::sample> imgB [[texture(1)]],
        texture2d<float, access::sample> flowF [[texture(2)]],
        texture2d<float, access::sample> flowB [[texture(3)]],
        texture2d<float, access::sample> mask [[texture(4)]],
        texture2d<float, access::write> dst [[texture(5)]],
        constant WarpParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float t = p.t;

        float2 baseSize = float2(flowF.get_width(), flowF.get_height());
        // flow(base 픽셀 단위) → 정규화 UV 오프셋
        float2 f = flowF.sample(s, uv).rg / baseSize;
        float2 b = flowB.sample(s, uv).rg / baseSize;

        // backward 매핑: 출력 p의 물체는 A에서 p - t·F_ab 에, B에서 p - (1-t)·F_ba 에 있었다
        // (flow를 p에서 샘플하는 소모션 근사 — 표준 기법)
        float3 w0 = imgA.sample(s, uv - f * t).rgb;
        float3 w1 = imgB.sample(s, uv - b * (1.0 - t)).rgb;
        float3 interp = mix(w0, w1, t);

        float4 m = mask.sample(s, uv);
        float conf = m.r;
        float staticness = m.g;

        float3 crossfade = mix(imgA.sample(s, uv).rgb, imgB.sample(s, uv).rgb, t);
        float3 moving = mix(crossfade, interp, conf);   // 저신뢰 → 소프트 크로스페이드
        float3 outc = mix(moving, imgB.sample(s, uv).rgb, staticness); // 정적 → B 원본 (선명)
        dst.write(float4(outc, 1.0), gid);
    }
    """
}
