import Foundation
import Metal
import Monitoring

/// **정적 UI 층** — 엔진 밖에서 정적 UI를 떼어내고(추출) 보간 프레임 위에 다시 얹는다(합성).
/// 계획: MacFG_UILayer_Plan.md. 엔진은 UI가 지워진 `clean` 프레임만 받는다.
///
/// 소스 프레임 하나당 (cb1, 안정 복사·디텍터 갱신 직후):
///   1. blit: 소스 → clean (전체 복사, 구멍 밖은 소스 그대로)
///   2. uilAlphaPush (소스/2): α = mask × mix(still, 1, smoothstep(0.5, 0.9, mask)), still은 직전 프레임의
///      반해상도 루마와 비교. hole = smoothstep(0.25, 0.35, α)를 기록하고, 피라미드 0단(색, 가중치 1−hole)과
///      다음 프레임용 루마를 같이 쓴다. 소스는 반해상도 픽셀 중심에서 bilinear 1회 = 2×2 평균이라 읽기가 한 번.
///   3. uilTiles: 32×32(소스 픽셀) 타일 중 구멍이 있는 것만 목록으로 (3×3 팽창 — bilinear 경계 이음매 방지)
///   4. push-pull: 구멍을 주변 배경으로 채움 (정규화 색 + 가중치 저장)
///   5. uilFill: 타일 목록에만 간접 디스패치 — clean의 구멍을 채움
/// 쌍 하나당 (cb2, encodePair 직후): uilComposite — 엔진 출력 텍스처에 **제자리로** 타일만
///   out = mix(out, mix(A, B, t), hole_B). 새 텍스처·새 수명 관리가 필요 없다.
///
/// 프레임 슬롯(clean·hole·luma·타일)은 소유 소스 텍스처(ObjectIdentifier)로 표시하고, 호출측이 넘기는
/// `busy`(아직 엔진이 읽을 수 있는 소스 텍스처들)에 속하지 않은 슬롯만 재사용한다. 즉 clean의 수명은 앱의
/// 기존 stable 텍스처 임대 규칙을 그대로 따른다 — 새 동기화가 없다.
public final class UILayer {
    /// P2 기본 ON (`MACFG_UILAYER=0`으로 끔).
    public nonisolated(unsafe) static var enabled = Knob.string("MACFG_UILAYER") != "0"
    /// 정지 판정 문턱(루마, 반해상도 2×2 평균): |cur−prev| < lo면 정지, > hi면 움직임.
    public nonisolated(unsafe) static var stillLo: Float = Float(Knob.string("MACFG_UILSTILLLO") ?? "") ?? 0.02
    public nonisolated(unsafe) static var stillHi: Float = Float(Knob.string("MACFG_UILSTILLHI") ?? "") ?? 0.06
    /// 구멍 문턱(α). 합성이 구멍 픽셀을 전부 원본 블렌드로 되돌리므로 낮춰도 정지 영역은 정확하다 —
    /// 낮추는 이유는 막 뜬 자막(마스크 형성 중, 0.1~0.3)의 글자 조각이 clean에 남아 엔진에 워프되는 것을 막기 위해서다.
    public nonisolated(unsafe) static var holeLo: Float = Float(Knob.string("MACFG_UILHOLELO") ?? "") ?? 0.12
    public nonisolated(unsafe) static var holeHi: Float = Float(Knob.string("MACFG_UILHOLEHI") ?? "") ?? 0.22
    /// 강한 마스크 근처(±nearR 마스크 텍셀)의 **정지** 픽셀도 구멍 — 글자 외곽선·획 사이 틈. 움직이는 배경은 제외.
    public nonisolated(unsafe) static var nearR: Float = Float(Knob.string("MACFG_UILNEAR") ?? "") ?? 0
    /// 프로파일용: 추출을 k단계에서 멈춘다 (1=복사 2=+α 3=+타일 4=+push 5=+pull 6=+채움). 0=전부.
    public nonisolated(unsafe) static var profileStop: Int = 0
    /// push-pull 단수 상한 (0단 = 소스/2). 1×1까지 내려가야 화면 폭만 한 구멍도 반드시 닫힌다 —
    /// 최하단이 구멍으로 남으면 채움이 검정이 된다. 4K는 12단에서 1×1.
    public nonisolated(unsafe) static var levels: Int = Int(Knob.string("MACFG_UILLEVELS") ?? "") ?? 12
    /// 프레임 슬롯 상한. 평시 3장(직전·현재·GPU 대기 1) — 백로그가 더 깊으면 이 프레임은 층 없이 간다.
    public nonisolated(unsafe) static var maxSlots: Int = 6
    static let tile = 32   // 소스 픽셀

    /// 소스 한 장의 층 산출물. `clean`은 엔진 입력, 나머지는 합성·다음 프레임 정지 판정용.
    public final class Frame {
        public let clean: any MTLTexture
        let hole: any MTLTexture       // 소스/2, r8 — 구멍 강도(합성 가중치)
        let luma: any MTLTexture       // 소스/2, r16f — 다음 프레임의 정지 판정용
        let tiles: any MTLBuffer       // uint 타일 인덱스
        let flags: any MTLBuffer       // uint 타일별 구멍 플래그 (α 패스가 세우고 압축 패스가 목록화)
        let args: any MTLBuffer        // 간접 디스패치 인자 (x = 타일 수, 1, 1)
        public let tilesX: UInt32
        public fileprivate(set) var owner: ObjectIdentifier?
        let tileCountTotal: Int
        init(clean: any MTLTexture, hole: any MTLTexture, luma: any MTLTexture, tiles: any MTLBuffer, flags: any MTLBuffer, args: any MTLBuffer, tilesX: UInt32, tileCountTotal: Int) {
            self.clean = clean; self.hole = hole; self.luma = luma; self.tiles = tiles; self.flags = flags; self.args = args
            self.tilesX = tilesX; self.tileCountTotal = tileCountTotal
        }
        /// 직전 완료된 추출의 구멍 타일 수 (CPU 판독, 진단용 — GPU가 쓰는 중이면 옛 값).
        public var tileCount: Int { Int(args.contents().load(as: UInt32.self)) }
    }

    public private(set) var available = false
    private let device: any MTLDevice
    private var alphaPushPSO: (any MTLComputePipelineState)?
    private var tilesPSO: (any MTLComputePipelineState)?
    private var pushPSO: (any MTLComputePipelineState)?
    private var pullPSO: (any MTLComputePipelineState)?
    private var fillPSO: (any MTLComputePipelineState)?
    private var compositePSO: (any MTLComputePipelineState)?
    private var initArgs: (any MTLBuffer)?
    private var pyr: [any MTLTexture] = []      // 공유 스크래치 (추출 한 번 안에서만 유효)
    private var slots: [Frame] = []
    private var slotW = 0, slotH = 0, slotFmt: MTLPixelFormat = .invalid
    private var extractCount = 0, skipCount = 0, compositeCount = 0
    private var coverageAcc = 0.0, coverageN = 0

    public init(device: any MTLDevice) { self.device = device }

    public func prepare() async throws {
        let lib = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ n: String) throws -> any MTLComputePipelineState {
            guard let f = lib.makeFunction(name: n) else { throw NSError(domain: "UILayer", code: 1, userInfo: [NSLocalizedDescriptionKey: n]) }
            return try device.makeComputePipelineState(function: f)
        }
        alphaPushPSO = try pso("uilAlphaPush"); tilesPSO = try pso("uilTiles"); pushPSO = try pso("uilPush")
        pullPSO = try pso("uilPull"); fillPSO = try pso("uilFill"); compositePSO = try pso("uilComposite")
        var ia: [UInt32] = [0, 1, 1]
        initArgs = device.makeBuffer(bytes: &ia, length: 12, options: .storageModeShared)
        available = selfTest(lib: lib)
        DiagnosticLog.shared.log("[UILAYER] prepared available=\(available) enabled=\(Self.enabled) still=\(Self.stillLo)/\(Self.stillHi) levels=\(Self.levels)")
    }

    /// 합성은 엔진 출력(bgra8Unorm)을 read_write로 제자리 갱신한다. 기기가 이 포맷의 read_write를 못 하면
    /// 층을 끈다(잘못된 픽셀보다 UI 처리 없는 쪽이 낫다). M4에서 검증 레이어 통과·결과 정확 확인(2026-09-30).
    private func selfTest(lib: any MTLLibrary) -> Bool {
        guard let f = lib.makeFunction(name: "uilRWTest"), let p = try? device.makeComputePipelineState(function: f) else { return false }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 4, height: 4, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d), let q = device.makeCommandQueue(),
              let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        var px = [UInt8](repeating: 0, count: 64)
        for i in 0..<16 { px[i * 4] = 200; px[i * 4 + 1] = 100; px[i * 4 + 2] = 50; px[i * 4 + 3] = 255 }
        t.replace(region: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0, withBytes: px, bytesPerRow: 16)
        enc.setComputePipelineState(p); enc.setTexture(t, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 4, height: 4, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var out = [UInt8](repeating: 0, count: 64)
        t.getBytes(&out, bytesPerRow: 16, from: MTLRegionMake2D(0, 0, 4, 4), mipmapLevel: 0)
        // 메모리 BGRA (200,100,50) = RGB (50,100,200) → 커널이 R만 절반 → RGB (25,100,200) → 메모리 (200,100,25)
        let ok = cb.status == .completed && abs(Int(out[0]) - 200) <= 1 && abs(Int(out[1]) - 100) <= 1 && abs(Int(out[2]) - 25) <= 1
        if !ok { DiagnosticLog.shared.log("[UILAYER] ⚠︎ bgra8 read_write 자가진단 실패 → 층 비활성 (out=\(Array(out[0..<4])))") }
        return ok
    }

    private func tex(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat) -> (any MTLTexture)? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: max(1, w), height: max(1, h), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
        return device.makeTexture(descriptor: d)
    }

    private func ensure(w: Int, h: Int, fmt: MTLPixelFormat) -> Bool {
        if w == slotW && h == slotH && fmt == slotFmt && !pyr.isEmpty { return true }
        slots = []; pyr = []
        slotW = w; slotH = h; slotFmt = fmt
        // 0단 = 소스/4 (α 패스가 스레드그룹 안에서 2×2 축약해 바로 쓴다). 채움은 원래 흐려서 반해상도 0단이 이득이 없었다.
        var lw = max(1, (w + 3) / 4), lh = max(1, (h + 3) / 4)
        for _ in 0..<max(2, Self.levels) {
            guard let t = tex(lw, lh, .rgba16Float) else { pyr = []; return false }
            pyr.append(t)
            if lw == 1 && lh == 1 { break }
            lw = max(1, (lw + 1) / 2); lh = max(1, (lh + 1) / 2)
        }
        DiagnosticLog.shared.log("[UILAYER] 자원 \(w)x\(h) 피라미드 \(pyr.count)단 (최하 \(pyr.last?.width ?? 0)x\(pyr.last?.height ?? 0))")
        return true
    }

    private func makeSlot() -> Frame? {
        let hw = (slotW + 1) / 2, hh = (slotH + 1) / 2
        let tx = (slotW + Self.tile - 1) / Self.tile, ty = (slotH + Self.tile - 1) / Self.tile
        guard let clean = tex(slotW, slotH, slotFmt), let hole = tex(hw, hh, .r8Unorm), let luma = tex(hw, hh, .r16Float),
              let tiles = device.makeBuffer(length: max(4, tx * ty * 4), options: .storageModePrivate),
              let flags = device.makeBuffer(length: max(4, tx * ty * 4), options: .storageModePrivate),
              let args = device.makeBuffer(length: 12, options: .storageModeShared) else { return nil }
        return Frame(clean: clean, hole: hole, luma: luma, tiles: tiles, flags: flags, args: args, tilesX: UInt32(tx), tileCountTotal: tx * ty)
    }

    /// 소스 한 장에서 층을 추출한다. `prev`는 직전 프레임의 추출 결과(정지 판정용, 없으면 still=1).
    /// `busy`: 아직 엔진이 읽을 수 있는 소스 텍스처들 — 그 소유의 슬롯은 재사용하지 않는다.
    /// 슬롯이 없으면(백로그가 깊음) nil — 호출측은 이 프레임을 층 없이 처리한다.
    public func extract(source: any MTLTexture, prev: Frame?, mask: any MTLTexture, owner: ObjectIdentifier,
                        busy: Set<ObjectIdentifier>, into cb: any MTLCommandBuffer) -> Frame? {
        guard available, let alphaPushPSO, let tilesPSO, let pushPSO, let pullPSO, let fillPSO, let initArgs,
              ensure(w: source.width, h: source.height, fmt: source.pixelFormat) else { return nil }
        var slot = slots.first { s in s !== prev && (s.owner == nil || !busy.contains(s.owner!)) }
        if slot == nil, slots.count < Self.maxSlots, let s = makeSlot() { slots.append(s); slot = s }
        guard let f = slot else { skipCount += 1; return nil }
        f.owner = owner
        let usePrev = prev != nil && prev!.luma.width == f.luma.width && prev!.luma.height == f.luma.height

        guard let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
                  to: f.clean, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.copy(from: initArgs, sourceOffset: 0, to: f.args, destinationOffset: 0, size: 12)
        blit.fill(buffer: f.flags, range: 0..<f.flags.length, value: 0)
        blit.endEncoding()
        let stop = Self.profileStop
        if stop == 1 { return f }

        guard let enc = cb.makeComputeCommandEncoder() else { return nil }
        let hw = f.hole.width, hh = f.hole.height
        // 1) α·hole·루마·피라미드 0단
        enc.setComputePipelineState(alphaPushPSO)
        // prev 없으면 더미로 mask를 묶는다 (같은 디스패치에서 쓰는 텍스처를 읽기로 겹쳐 묶지 않는다)
        enc.setTexture(source, index: 0); enc.setTexture(usePrev ? prev!.luma : mask, index: 1)
        enc.setTexture(mask, index: 2); enc.setTexture(f.hole, index: 3); enc.setTexture(f.luma, index: 4); enc.setTexture(pyr[0], index: 5)
        var p = SIMD4<Float>(Self.stillLo, Self.stillHi, usePrev ? 1 : 0, Self.nearR)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        var p2 = SIMD2<Float>(Self.holeLo, Self.holeHi)
        enc.setBytes(&p2, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        var tx = f.tilesX
        var tyN = UInt32((hh + 15) / 16)
        enc.setBytes(&tx, length: 4, index: 2); enc.setBytes(&tyN, length: 4, index: 3)
        enc.setBuffer(f.flags, offset: 0, index: 4)
        grid(enc, hw, hh)   // 스레드그룹 16×16 반해상도 = 소스 32×32 타일 하나 (타일 플래그의 전제)
        if stop == 2 { enc.endEncoding(); return f }
        // 2) 타일 목록 압축 — 타일당 스레드 하나
        enc.setComputePipelineState(tilesPSO)
        enc.setBuffer(f.flags, offset: 0, index: 0); enc.setBuffer(f.args, offset: 0, index: 1); enc.setBuffer(f.tiles, offset: 0, index: 2)
        var nt = UInt32(f.tileCountTotal)
        enc.setBytes(&nt, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (f.tileCountTotal + 255) / 256, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if stop == 3 { enc.endEncoding(); return f }
        // 3) push (정규화 색 + 가중치)
        enc.setComputePipelineState(pushPSO)
        for l in 1..<pyr.count {
            enc.setTexture(pyr[l - 1], index: 0); enc.setTexture(pyr[l], index: 1)
            grid(enc, pyr[l].width, pyr[l].height)
        }
        if stop == 4 { enc.endEncoding(); return f }
        // 4) pull — 거친 단의 색으로 가중치 부족분을 채우며 올라온다 (제자리)
        enc.setComputePipelineState(pullPSO)
        for l in stride(from: pyr.count - 2, through: 0, by: -1) {
            enc.setTexture(pyr[l + 1], index: 0); enc.setTexture(pyr[l], index: 1)
            grid(enc, pyr[l].width, pyr[l].height)
        }
        if stop == 5 { enc.endEncoding(); return f }
        // 5) 채움 — 구멍 타일에만
        enc.setComputePipelineState(fillPSO)
        enc.setTexture(source, index: 0); enc.setTexture(f.hole, index: 1); enc.setTexture(pyr[0], index: 2); enc.setTexture(f.clean, index: 3)
        enc.setBuffer(f.tiles, offset: 0, index: 0)
        enc.setBytes(&tx, length: 4, index: 1)
        enc.dispatchThreadgroups(indirectBuffer: f.args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()

        extractCount += 1
        if let pv = prev {   // 직전 추출은 대개 완료 — 커버리지 진단
            let total = Double(Int(f.tilesX) * ((slotH + Self.tile - 1) / Self.tile))
            if total > 0 { coverageAcc += Double(pv.tileCount) / total; coverageN += 1 }
        }
        if extractCount % 600 == 0 {
            let cov = coverageN > 0 ? coverageAcc / Double(coverageN) * 100 : 0
            DiagnosticLog.shared.log(String(format: "[UILAYER] 추출 %d 슬롯부족 %d 합성 %d 구멍타일 %.1f%% 슬롯 %d장",
                                            extractCount, skipCount, compositeCount, cov, slots.count))
            coverageAcc = 0; coverageN = 0
        }
        return f
    }

    /// 엔진 출력(`output`, bgra8)에 제자리 합성: 구멍 타일에서 out = mix(out, mix(A, B, t), hole_B).
    /// UI 층은 A·B의 t 블렌드 — 정지 UI면 동일, 게이지·타이머처럼 천천히 변하는 UI면 중간 프레임에 더 가깝다.
    public func composite(output: any MTLTexture, sourceA: any MTLTexture, sourceB: any MTLTexture, t: Float,
                          frameB: Frame, into cb: any MTLCommandBuffer) {
        guard available, let compositePSO, output.width == frameB.clean.width, output.height == frameB.clean.height,
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(compositePSO)
        enc.setTexture(output, index: 0); enc.setTexture(sourceA, index: 1); enc.setTexture(sourceB, index: 2); enc.setTexture(frameB.hole, index: 3)
        enc.setBuffer(frameB.tiles, offset: 0, index: 0)
        var tx = frameB.tilesX, tt = t
        enc.setBytes(&tx, length: 4, index: 1); enc.setBytes(&tt, length: 4, index: 2)
        enc.dispatchThreadgroups(indirectBuffer: frameB.args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        compositeCount += 1
    }

    /// 불연속(캡처 재시작·크기 변경) — 슬롯을 통째로 버린다. 소유 표시만 지우면 아직 GPU에서 읽히는
    /// 슬롯을 다음 추출이 덮을 수 있다(특히 RIFE의 분리 큐). 버린 텍스처는 커맨드 버퍼가 붙잡고 있다가 놓는다.
    public func reset() { slots = [] }

    private func grid(_ enc: any MTLComputeCommandEncoder, _ w: Int, _ h: Int) {
        enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    constant half3 kLum = half3(0.299h, 0.587h, 0.114h);

    kernel void uilRWTest(texture2d<half, access::read_write> t [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= t.get_width() || gid.y >= t.get_height()) return;
        half4 v = t.read(gid);
        t.write(half4(v.r * 0.5h, v.g, v.b, 1.0h), gid);
    }

    // 반해상도 격자, 스레드그룹 16×16 = 소스 32×32 타일. 소스는 반해상도 픽셀 중심에서 bilinear = 소스 2×2 평균.
    // 산출: hole(r8)·luma(r16f) 반해상도, 피라미드 0단(소스/4)은 그룹 안 2×2 축약, 타일 플래그(가장자리면 이웃 타일도 —
    // 채움·합성이 hole을 bilinear로 읽어 이웃 타일 구멍이 가장자리 픽셀에 번지기 때문).
    kernel void uilAlphaPush(texture2d<half, access::sample> cur [[texture(0)]],
                             texture2d<half, access::sample> prevLuma [[texture(1)]],
                             texture2d<float, access::sample> mask [[texture(2)]],
                             texture2d<half, access::write> hole [[texture(3)]],
                             texture2d<half, access::write> lumaOut [[texture(4)]],
                             texture2d<half, access::write> pyr0 [[texture(5)]],
                             constant float4& p [[buffer(0)]],        // stillLo, stillHi, hasPrev, nearR(마스크 텍셀)
                             constant float2& hp [[buffer(1)]],       // holeLo, holeHi
                             constant uint& tilesX [[buffer(2)]],
                             constant uint& tilesY [[buffer(3)]],
                             device uint* flags [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]]) {
        threadgroup half4 red[16][16];
        uint w = hole.get_width(), h = hole.get_height();
        bool inside = gid.x < w && gid.y < h;
        half4 contrib = half4(0.0h);
        if (inside) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 uv = (float2(gid) + 0.5) / float2(w, h);
            half3 c = cur.sample(s, uv).rgb;
            half l = dot(c, kLum);
            lumaOut.write(half4(l), gid);
            float m = clamp(mask.sample(s, uv).r, 0.0, 1.0);
            float still = 1.0;
            if (p.z > 0.5) still = 1.0 - smoothstep(p.x, p.y, fabs(float(l - prevLuma.sample(s, uv).r)));
            // 강한 마스크(글자 획·패널 텍스트)는 정지 여부 무관 — 스크롤 중 채팅은 A≠B인데 마스크가 높다.
            float a = m * mix(still, 1.0, smoothstep(0.5, 0.9, m));
            float ms = m;
            if (p.w > 0.0) {
                float2 em = p.w / float2(mask.get_width(), mask.get_height());
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        ms = max(ms, mask.sample(s, uv + float2(dx, dy) * em).r);
            }
            half hv = half(max(smoothstep(hp.x, hp.y, a), p.w > 0.0 ? smoothstep(0.5, 0.9, ms) * still : 0.0));
            hole.write(half4(hv), gid);
            half wgt = 1.0h - hv;
            contrib = half4(c * wgt, wgt);
            if (hv > 0.004h) {
                int tx = int(tg.x), ty = int(tg.y);
                int x0 = (lid.x == 0) ? -1 : 0, x1 = (lid.x == 15) ? 1 : 0;
                int y0 = (lid.y == 0) ? -1 : 0, y1 = (lid.y == 15) ? 1 : 0;
                for (int dy = y0; dy <= y1; dy++)
                    for (int dx = x0; dx <= x1; dx++) {
                        int nx = tx + dx, ny = ty + dy;
                        if (nx >= 0 && ny >= 0 && nx < int(tilesX) && ny < int(tilesY)) flags[ny * int(tilesX) + nx] = 1u;
                    }
            }
        }
        red[lid.y][lid.x] = contrib;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((lid.x & 1) == 0 && (lid.y & 1) == 0) {
            half4 acc = red[lid.y][lid.x] + red[lid.y][lid.x + 1] + red[lid.y + 1][lid.x] + red[lid.y + 1][lid.x + 1];
            uint2 q = gid / 2;
            if (q.x < pyr0.get_width() && q.y < pyr0.get_height()) {
                half wn = (acc.a < 0.05h) ? 0.0h : min(acc.a / 4.0h, 1.0h);
                half3 cc = (acc.a > 0.0h) ? (acc.rgb / acc.a) : half3(0.0h);
                pyr0.write(half4(cc, wn), q);
            }
        }
    }

    // 타일 목록 압축: 플래그가 선 타일만 간접 디스패치 목록에.
    kernel void uilTiles(device const uint* flags [[buffer(0)]],
                         device atomic_uint* args [[buffer(1)]],
                         device uint* tiles [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint t [[thread_position_in_grid]]) {
        if (t >= n || flags[t] == 0u) return;
        uint idx = atomic_fetch_add_explicit(&args[0], 1u, memory_order_relaxed);
        tiles[idx] = t;
    }

    // push: 2×2 가중 평균. 저장은 (정규화 색, 가중치) — 미정규화 저장은 거친 단의 미세 가중치에서
    // half 나눗셈이 터져 흰색이 됐다(2026-09-28 실측). 가중치 합이 너무 작으면 구멍으로 둔다.
    kernel void uilPush(texture2d<half, access::read> src [[texture(0)]],
                        texture2d<half, access::write> dst [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        uint sw = src.get_width(), sh = src.get_height();
        half3 acc = half3(0.0h); half wsum = 0.0h;
        for (uint j = 0; j < 4; j++) {
            uint2 q = uint2(min(gid.x * 2 + (j & 1), sw - 1), min(gid.y * 2 + (j >> 1), sh - 1));
            half4 v = src.read(q);
            acc += v.rgb * v.a; wsum += v.a;
        }
        half wn = (wsum < 0.05h) ? 0.0h : min(wsum / 4.0h, 1.0h);
        half3 c = (wsum > 0.0h) ? (acc / wsum) : half3(0.0h);
        dst.write(half4(c, wn), gid);
    }

    // pull: 가중치 부족분을 거친 단 색으로 (제자리). 거친 단은 이미 pull을 거쳐 a=1(최하단은 push 결과).
    kernel void uilPull(texture2d<half, access::sample> coarse [[texture(0)]],
                        texture2d<half, access::read_write> fine [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
        uint w = fine.get_width(), h = fine.get_height();
        if (gid.x >= w || gid.y >= h) return;
        half4 f = fine.read(gid);
        if (f.a >= 1.0h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        half4 c = coarse.sample(s, uv);
        half3 outc = (c.a > 0.0h) ? mix(c.rgb, f.rgb, f.a) : f.rgb;
        fine.write(half4(outc, (c.a > 0.0h || f.a > 0.0h) ? 1.0h : 0.0h), gid);
    }

    // 채움: 타일당 16×16 스레드가 소스 2×2씩. clean은 이미 소스 복사본이라 구멍 픽셀만 쓴다.
    kernel void uilFill(texture2d<half, access::read> src [[texture(0)]],
                        texture2d<half, access::sample> hole [[texture(1)]],
                        texture2d<half, access::sample> pyr0 [[texture(2)]],
                        texture2d<half, access::write> clean [[texture(3)]],
                        device const uint* tiles [[buffer(0)]],
                        constant uint& tilesX [[buffer(1)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]]) {
        uint t = tiles[tg.x];
        uint2 base = uint2((t % tilesX) * 32u, (t / tilesX) * 32u) + lid * 2u;
        uint w = clean.get_width(), h = clean.get_height();
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        for (uint j = 0; j < 4; j++) {
            uint2 g = base + uint2(j & 1, j >> 1);
            if (g.x >= w || g.y >= h) continue;
            float2 uv = (float2(g) + 0.5) / float2(w, h);
            half k = hole.sample(s, uv).r;
            if (k <= 0.004h) continue;
            half3 fill = pyr0.sample(s, uv).rgb;
            clean.write(half4(mix(src.read(g).rgb, fill, k), 1.0h), g);
        }
    }

    // 제자리 합성 (엔진 출력 read_write).
    kernel void uilComposite(texture2d<half, access::read_write> out [[texture(0)]],
                             texture2d<half, access::read> srcA [[texture(1)]],
                             texture2d<half, access::read> srcB [[texture(2)]],
                             texture2d<half, access::sample> hole [[texture(3)]],
                             device const uint* tiles [[buffer(0)]],
                             constant uint& tilesX [[buffer(1)]],
                             constant float& t [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]]) {
        uint ti = tiles[tg.x];
        uint2 base = uint2((ti % tilesX) * 32u, (ti / tilesX) * 32u) + lid * 2u;
        uint w = out.get_width(), h = out.get_height();
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        for (uint j = 0; j < 4; j++) {
            uint2 g = base + uint2(j & 1, j >> 1);
            if (g.x >= w || g.y >= h) continue;
            float2 uv = (float2(g) + 0.5) / float2(w, h);
            half k = hole.sample(s, uv).r;
            if (k <= 0.004h) continue;
            half3 layer = mix(srcA.read(g).rgb, srcB.read(g).rgb, half(t));
            out.write(half4(mix(out.read(g).rgb, layer, k), 1.0h), g);
        }
    }
    """
}
