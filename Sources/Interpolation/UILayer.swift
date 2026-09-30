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
    /// **닫힘 반경** (¼해상도 텍셀, 기본 8 = 소스 32px). 강한 마스크 근처의 **정지** 픽셀을 구멍에 넣는다 —
    /// 넓고 평평한 글자 획 안쪽은 고주파가 없어 마스크가 0이라, 테두리만 구멍이 되고 속이 남아 I 프레임에 윤곽선이
    /// 생겼다(2026-09-30 덤프 201510 "ㅋㅋㅋ"). 정지 픽셀만 넣으므로 움직이는 배경은 안 얼린다.
    public nonisolated(unsafe) static var closeR: Int = Int(Knob.string("MACFG_UILCLOSE") ?? "") ?? 8
    /// **강한 마스크의 느슨한 정지 기준** (루마 변화, 기본 0.08~0.20). 강한 마스크 픽셀은 조금 변해도(HUD 숫자,
    /// 글자 안티앨리어스 가장자리) 블렌드로 붙잡는 편이 정답에 가깝고(삼중항 +0.35dB), 크게 변한 픽셀(무기가 떠난
    /// 자리·글자 밖 움직이는 배경·스크롤)은 풀어줘야 유령 윤곽이 안 생긴다(덤프 201423). 예전의 "무조건 붙잡기"와
    /// "정지만"의 중간.
    public nonisolated(unsafe) static var strongLo: Float = Float(Knob.string("MACFG_UILSTRONGLO") ?? "") ?? 0.08
    public nonisolated(unsafe) static var strongHi: Float = Float(Knob.string("MACFG_UILSTRONGHI") ?? "") ?? 0.20
    /// **서브픽셀 정지 기준** (2×2 중 가장 덜 변한 서브픽셀의 루마 변화, 기본 0.06~0.12). 재인코딩 영상은 빠른 팬에서
    /// 정지 UI도 프레임마다 0.03~0.13씩 흔들리고(압축, 오프라인 168쌍 실측), 가는 UI(게이지 눈금·얇은 글자)는 2×2 평균이
    /// 배경과 섞여 평균 판정을 못 받는다. 서브픽셀 하나라도 거의 정지면 붙잡는다.
    public nonisolated(unsafe) static var subLo: Float = Float(Knob.string("MACFG_UILSUBLO") ?? "") ?? 0.06
    public nonisolated(unsafe) static var subHi: Float = Float(Knob.string("MACFG_UILSUBHI") ?? "") ?? 0.12
    /// 약한 마스크 항에도 서브픽셀 정지를 쓴다 (기본 ON, MACFG_UILSUBWEAK=0으로 끔). 가는 UI는 마스크가 강함 문턱(0.5)에
    /// 못 미친다. 실측: 삼중항 PSNR 세 엔진 +0.03~0.33dB, 무기 영역 조각 18246→15605, 띠 42.5→41.8%.
    public nonisolated(unsafe) static var subWeak: Bool = Knob.string("MACFG_UILSUBWEAK") != "0"
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
        let luma: any MTLTexture       // 소스 전해상도 r8 — 다음 프레임의 정지 판정용 (서브픽셀 단위)
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
    private var dilatePSO: (any MTLComputePipelineState)?
    private var alphaPushPSO: (any MTLComputePipelineState)?
    private var tilesPSO: (any MTLComputePipelineState)?
    private var pushPSO: (any MTLComputePipelineState)?
    private var pullPSO: (any MTLComputePipelineState)?
    private var fillPSO: (any MTLComputePipelineState)?
    private var compositePSO: (any MTLComputePipelineState)?
    private var initArgs: (any MTLBuffer)?
    private var pyr: [any MTLTexture] = []      // 공유 스크래치 (추출 한 번 안에서만 유효)
    private var ssQ: (any MTLTexture)?          // ¼해상도: 강한마스크×정지 (2×2 최대)
    private var nearQ: (any MTLTexture)?        // ¼해상도: ssQ의 최대 팽창 (닫힘)
    private var tmpQ: (any MTLTexture)?
    /// nearQ가 직전 프레임의 팽창 결과를 담고 있는가 (리셋·크기 변경 뒤 첫 프레임은 없음).
    private var nearValid = false
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
        dilatePSO = try pso("uilDilate")
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
        let qw = max(1, (w + 3) / 4), qh = max(1, (h + 3) / 4)
        ssQ = tex(qw, qh, .r8Unorm); nearQ = tex(qw, qh, .r8Unorm); tmpQ = tex(qw, qh, .r8Unorm)
        nearValid = false
        guard ssQ != nil, nearQ != nil, tmpQ != nil else { pyr = []; return false }
        DiagnosticLog.shared.log("[UILAYER] 자원 \(w)x\(h) 피라미드 \(pyr.count)단 (최하 \(pyr.last?.width ?? 0)x\(pyr.last?.height ?? 0)) 닫힘 R=\(Self.closeR)")
        return true
    }

    private func makeSlot() -> Frame? {
        let hw = (slotW + 1) / 2, hh = (slotH + 1) / 2
        let tx = (slotW + Self.tile - 1) / Self.tile, ty = (slotH + Self.tile - 1) / Self.tile
        guard let clean = tex(slotW, slotH, slotFmt), let hole = tex(hw, hh, .r8Unorm), let luma = tex(slotW, slotH, .r8Unorm),
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
        guard available, let dilatePSO, let alphaPushPSO, let tilesPSO, let pushPSO, let pullPSO, let fillPSO, let initArgs,
              ensure(w: source.width, h: source.height, fmt: source.pixelFormat),
              let ssQ, let nearQ, let tmpQ else { return nil }
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
        // 1) 구멍 (½해상도 단일 패스): hole = max(still × max(weak, near_직전), strong × still2)
        //    + 루마(다음 프레임 정지 판정) + 피라미드 0단(¼) + ssQ(¼, 다음 프레임 닫힘의 씨앗) + 타일 플래그.
        //    닫힘(near)은 **직전 프레임의 팽창 결과**를 쓴다 — 정적 글자 속을 채우는 용도라 한 프레임 늦어도 같고,
        //    그 덕에 패스가 하나로 끝난다(분리하면 4K +0.5ms, 2026-09-30 실측).
        enc.setComputePipelineState(alphaPushPSO)
        // prev 없으면 더미로 mask를 묶는다 (같은 디스패치에서 쓰는 텍스처를 읽기로 겹쳐 묶지 않는다)
        enc.setTexture(source, index: 0); enc.setTexture(usePrev ? prev!.luma : mask, index: 1)
        enc.setTexture(mask, index: 2); enc.setTexture(nearQ, index: 3)
        enc.setTexture(f.luma, index: 4); enc.setTexture(f.hole, index: 5); enc.setTexture(pyr[0], index: 6); enc.setTexture(ssQ, index: 7)
        var p = SIMD4<Float>(Self.stillLo, Self.stillHi, usePrev ? 1 : 0, nearValid && Self.closeR > 0 ? 1 : 0)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        var p2 = SIMD4<Float>(Self.holeLo, Self.holeHi, Self.strongLo, Self.strongHi)
        enc.setBytes(&p2, length: MemoryLayout<SIMD4<Float>>.size, index: 1)
        var p3 = SIMD4<Float>(Self.subLo, Self.subHi, Self.subWeak ? 1 : 0, 0)
        enc.setBytes(&p3, length: MemoryLayout<SIMD4<Float>>.size, index: 5)
        var tx = f.tilesX
        var tyN = UInt32((hh + 15) / 16)
        enc.setBytes(&tx, length: 4, index: 2); enc.setBytes(&tyN, length: 4, index: 3)
        enc.setBuffer(f.flags, offset: 0, index: 4)
        grid(enc, hw, hh)   // 스레드그룹 16×16 반해상도 = 소스 32×32 타일 하나 (타일 플래그의 전제)
        // 2) 다음 프레임용 닫힘: ssQ를 ¼해상도에서 분리형 최대 팽창 (가로→tmp, 세로→near)
        if Self.closeR > 0 {
            enc.setComputePipelineState(dilatePSO)
            var r = Int32(Self.closeR)
            for (src, dst, ax) in [(ssQ, tmpQ, SIMD2<Int32>(1, 0)), (tmpQ, nearQ, SIMD2<Int32>(0, 1))] {
                enc.setTexture(src, index: 0); enc.setTexture(dst, index: 1)
                var a2 = ax
                enc.setBytes(&a2, length: MemoryLayout<SIMD2<Int32>>.size, index: 0)
                enc.setBytes(&r, length: 4, index: 1)
                grid(enc, ssQ.width, ssQ.height)
            }
            nearValid = true
        }
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
    public func reset() { slots = []; nearValid = false }

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

    // 분리형 최대 팽창 (¼해상도). ax = (1,0) 가로 / (0,1) 세로, r = 반경.
    kernel void uilDilate(texture2d<half, access::read> src [[texture(0)]],
                          texture2d<half, access::write> dst [[texture(1)]],
                          constant int2& ax [[buffer(0)]],
                          constant int& r [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
        int W = int(dst.get_width()), H = int(dst.get_height());
        if (int(gid.x) >= W || int(gid.y) >= H) return;
        half mx = 0.0h;
        for (int k = -r; k <= r; k++) {
            int2 q = clamp(int2(gid) + ax * k, int2(0), int2(W - 1, H - 1));
            mx = max(mx, src.read(uint2(q)).r);
        }
        dst.write(half4(mx), gid);
    }

    // 구멍 (½해상도 단일 패스, 그룹 16×16 = 소스 32×32 타일). 소스는 ½ 픽셀 중심 bilinear = 소스 2×2 평균.
    //   still  = 1 − smoothstep(stillLo, stillHi, |루마 − 직전 루마|)       (0.02~0.06)
    //   still2 = 1 − smoothstep(strongLo, strongHi, 같은 값)                 (0.08~0.20, 강한 마스크용 느슨한 기준)
    //   hole   = max(still × max(weak, near_직전), strong × still2)
    // 약한 마스크·닫힘은 **실제 정지 픽셀만**, 강한 마스크는 **조금 변한 것까지만** 구멍이다. 예전의 "강한 마스크는
    // 무조건 구멍"은 움직인 무기·HUD의 옛 테두리와 글자 밖 4~8px 배경을 A·B 블렌드로 덮어 유령 윤곽을 만들었다
    // (덤프 201423). 닫힘은 넓고 평평한 글자 획 속(마스크 0)을 메운다(덤프 201510 "ㅋㅋㅋ"의 획 속 윤곽선).
    // 산출: luma(r16f)·hole(r8), 피라미드 0단(¼, 2×2 가중 평균), ssQ(¼, 강한×정지 2×2 최대), 타일 플래그(가장자리면 이웃도).
    kernel void uilAlphaPush(texture2d<half, access::read> cur [[texture(0)]],
                             texture2d<half, access::read> prevLuma [[texture(1)]],
                             texture2d<float, access::sample> mask [[texture(2)]],
                             texture2d<half, access::sample> nearQ [[texture(3)]],
                             texture2d<half, access::write> lumaOut [[texture(4)]],
                             texture2d<half, access::write> hole [[texture(5)]],
                             texture2d<half, access::write> pyr0 [[texture(6)]],
                             texture2d<half, access::write> ssQ [[texture(7)]],
                             constant float4& p [[buffer(0)]],        // stillLo, stillHi, hasPrev, hasNear
                             constant float4& hp [[buffer(1)]],       // holeLo, holeHi, strongLo, strongHi
                             constant uint& tilesX [[buffer(2)]],
                             constant uint& tilesY [[buffer(3)]],
                             device uint* flags [[buffer(4)]],
                             constant float4& sp [[buffer(5)]],       // subLo, subHi, subWeak (서브픽셀 정지)
                             uint2 gid [[thread_position_in_grid]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint2 lid [[thread_position_in_threadgroup]]) {
        threadgroup half4 red[16][16];
        threadgroup half redS[16][16];
        uint w = hole.get_width(), h = hole.get_height();
        half4 contrib = half4(0.0h);
        half ssv = 0.0h;
        if (gid.x < w && gid.y < h) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 uv = (float2(gid) + 0.5) / float2(w, h);
            // 서브픽셀 2×2를 직접 읽는다 — 루마는 전해상도로 저장해 다음 프레임이 서브픽셀 단위로 정지를 판정한다.
            uint W = cur.get_width(), H = cur.get_height();
            half3 c = half3(0.0h);
            float lsum = 0.0, psum = 0.0, dmin = 1.0;
            for (uint j = 0; j < 4; j++) {
                uint2 q = uint2(min(gid.x * 2 + (j & 1), W - 1), min(gid.y * 2 + (j >> 1), H - 1));
                half3 cq = cur.read(q).rgb;
                c += cq;
                half lq = dot(cq, kLum);
                lumaOut.write(half4(lq), q);
                lsum += float(lq);
                if (p.z > 0.5) {
                    float pq = float(prevLuma.read(q).r);
                    psum += pq;
                    dmin = min(dmin, fabs(float(lq) - pq));
                }
            }
            c *= 0.25h;
            float m = clamp(mask.sample(s, uv).r, 0.0, 1.0);
            float still = 1.0, still2 = 1.0, stillW = 1.0;
            if (p.z > 0.5) {
                float d = fabs(lsum - psum) * 0.25;   // 2×2 평균의 변화 (예전 ½해상도 판정과 같은 값)
                still = 1.0 - smoothstep(p.x, p.y, d);
                // 강한 마스크: 평균이 조금만 변했거나(느슨) **서브픽셀 하나라도 진짜 정지**면 붙잡는다 —
                // 게이지 눈금처럼 가는 UI는 2×2 평균이 배경과 섞여 빠른 팬에서 정지 판정을 못 받았다(오프라인 168쌍).
                still2 = max(1.0 - smoothstep(hp.z, hp.w, d), 1.0 - smoothstep(sp.x, sp.y, dmin));
                stillW = (sp.z > 0.5) ? max(still, 1.0 - smoothstep(sp.x, sp.y, dmin)) : still;
            }
            float weak = smoothstep(hp.x, hp.y, m);
            float strong = smoothstep(0.5, 0.9, m);
            float nr = (p.w > 0.5) ? float(nearQ.sample(s, uv).r) : 0.0;
            half hv = half(max(max(stillW * weak, still * nr), strong * still2));
            hole.write(half4(hv), gid);
            ssv = half(strong * still);
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
        redS[lid.y][lid.x] = ssv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((lid.x & 1) == 0 && (lid.y & 1) == 0) {
            uint2 q = gid / 2;
            half4 acc = red[lid.y][lid.x] + red[lid.y][lid.x + 1] + red[lid.y + 1][lid.x] + red[lid.y + 1][lid.x + 1];
            if (q.x < pyr0.get_width() && q.y < pyr0.get_height()) {
                half wn = (acc.a < 0.05h) ? 0.0h : min(acc.a / 4.0h, 1.0h);
                half3 cc = (acc.a > 0.0h) ? (acc.rgb / acc.a) : half3(0.0h);
                pyr0.write(half4(cc, wn), q);
            }
            half mx = max(max(redS[lid.y][lid.x], redS[lid.y][lid.x + 1]), max(redS[lid.y + 1][lid.x], redS[lid.y + 1][lid.x + 1]));
            if (q.x < ssQ.get_width() && q.y < ssQ.get_height()) ssQ.write(half4(mx), q);
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
