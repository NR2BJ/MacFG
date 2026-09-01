@preconcurrency import Metal
import Foundation
import CoreGraphics
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
    private var blurPso: (any MTLComputePipelineState)?
    private var meanTex: [any MTLTexture] = []   // ping-pong EMA(hp)
    private var sqTex: [any MTLTexture] = []      // ping-pong EMA(hp²)
    // 마스크도 ping-pong — cb1(blit+detector)을 copy 큐로 분리하면(O1-3), 다음 프레임 update가
    // 마스크를 쓰는 동안 이번 프레임 cb2(warp, work 큐)가 같은 마스크를 읽어 cross-queue 하자드가
    // 난다. 2버퍼로 번갈아 써서 읽는 버퍼와 쓰는 버퍼를 분리.
    private var maskTex: [any MTLTexture] = []
    /// 블러 전 raw 마스크 + 분리형 블러 중간 버퍼. 같은 cb 안에서 인코더 순서가 보장되므로
    /// ping-pong 불필요(위 maskTex의 2버퍼는 cross-queue 하자드용이라 사정이 다르다).
    private var rawTex: (any MTLTexture)?
    private var tmpTex: (any MTLTexture)?
    private var maskCur = 0
    private var cur = 0
    private var w = 0, h = 0
    private var needsReset = true
    private var frames = 0

    // ── 텍스트 박스 부스트 (Vision 검출 결과 — 일관성 신호가 놓치는 흐린 텍스트 보완)
    private let boxLock = NSLock()
    private var pendingBoxes: [CGRect]?          // 새 제출 (정규화, 좌상 원점)
    private var boxesStamp = 0                    // 제출 세대 (업로드 트리거)
    private var uploadedStamp = -1
    private var lastSubmitAt: CFTimeInterval = 0
    private var boostTex: [any MTLTexture] = []   // ping-pong (r8, 마스크 해상도)
    private var boostCur = 0
    private var boostBitmap: [UInt8] = []
    private var boostActive = false
    /// 텍스트 박스 유지 시간 — 이 시간 내 재검출 없으면 부스트 소멸 (채팅 사라짐 대응)
    public nonisolated(unsafe) static var boxTTL: CFTimeInterval = 6.0

    /// Vision 등 외부 검출기가 텍스트 박스를 제출 (아무 스레드) — 다음 update에서 업로드.
    /// rects: 정규화 좌표(0..1), 좌상 원점.
    /// generation: 인코딩 시점의 `generation` 값. reset() 이후 도착한 이전 캡처의 박스를
    /// 버린다 — 안 그러면 새 세션 첫 ~2초에 이전 장면의 텍스트 위치가 프리즈된다(리뷰 확정).
    public func submitTextBoxes(_ rects: [CGRect], generation: Int) {
        boxLock.lock()
        guard generation == resetGen else { boxLock.unlock(); return }
        pendingBoxes = rects
        boxesStamp += 1
        lastSubmitAt = CFAbsoluteTimeGetCurrent()
        boxLock.unlock()
    }

    private var resetGen = 0
    public var generation: Int { boxLock.lock(); defer { boxLock.unlock() }; return resetGen }

    /// 튜닝(오프라인 clo0.8 chi2.0 시작점). alpha=EMA율(~1/창길이). enabled=off면 no-op.
    public nonisolated(unsafe) static var enabled = true
    public nonisolated(unsafe) static var alpha: Float = 0.04   // ~25프레임 창 (스윕 최적)
    public nonisolated(unsafe) static var clo: Float = 0.5      // 스윕 최적 — 흐린 UI까지 잡되 무회귀
    public nonisolated(unsafe) static var chi: Float = 1.7
    public nonisolated(unsafe) static var strength: Float = 1.0  // 마스크 최대 프리즈 강도

    /// 워프가 샘플할 마스크 (없으면 nil). 워밍업 전(<8프레임)엔 nil 반환해 초기 노이즈 회피.
    /// 직전 update가 쓴 버퍼(maskCur)를 반환 — 다음 update는 반대 버퍼에 써서 하자드 회피.
    public var mask: (any MTLTexture)? {
        (Self.enabled && frames >= 8 && maskCur < maskTex.count) ? maskTex[maskCur] : nil
    }

    public init(device: any MTLDevice) { self.device = device }

    public func prepare() async throws {
        let lib = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        guard let fn = lib.makeFunction(name: "uiStaticUpdate"),
              let fb = lib.makeFunction(name: "uiMaskBlur") else { throw InterpolationError.shaderCompilationFailed }
        pso = try await device.makeComputePipelineState(function: fn)
        blurPso = try await device.makeComputePipelineState(function: fb)
    }

    /// 해상도 세팅/재세팅 — 소스 크기의 1/2(텍스트 보존 + 저비용).
    /// 마스크 격자 축소비 (소스 대비). **해상도가 올라가면 이 값도 올려야 한다.**
    ///
    /// 이 클래스의 공간 연산자는 전부 **마스크 픽셀** 단위다 — hp의 3x3 박스, 5탭 σ=2 블러.
    /// 축소비가 2로 고정이면 마스크 픽셀은 항상 소스 2px이므로, 같은 UI가 4K에서 픽셀로 2배
    /// 커질 때 연산자는 UI 대비 **절반 폭**이 된다. 그러면 hp 응답이 약해져 cons가 낮아지고
    /// UI를 덜 잡는다. 실측(2026-09-02, 같은 4K 소스를 4K/1080p로, 나머지 전부 고정):
    ///   1080p 커버 6.4% · ROI이득 +0.316 · full이득 −0.008  (배포 파라미터 전부 통과)
    ///   4K    커버 4.1% · ROI이득 +0.322 · full이득 −0.029  (배포 파라미터 **전부 탈락**)
    /// ROI 이득은 같은데 전체 손해만 3.5배다. 축소비를 해상도에 맞춰 키우면 연산자의 화면
    /// 상대 폭이 보존된다(4K에서 4 = 1080p에서 2와 같은 화면 비율).
    public nonisolated(unsafe) static var maskDiv: Int = 2

    private func ensure(srcW: Int, srcH: Int) {
        let dv = max(1, Self.maskDiv)
        let mw = max(64, srcW / dv), mh = max(64, srcH / dv)
        guard mw != w || mh != h || maskTex.isEmpty else { return }
        w = mw; h = mh
        func tex(_ fmt: MTLPixelFormat, _ usage: MTLTextureUsage) -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: mw, height: mh, mipmapped: false)
            d.usage = usage; d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        meanTex = [tex(.r16Float, [.shaderRead, .shaderWrite]), tex(.r16Float, [.shaderRead, .shaderWrite])].compactMap { $0 }
        sqTex = [tex(.r16Float, [.shaderRead, .shaderWrite]), tex(.r16Float, [.shaderRead, .shaderWrite])].compactMap { $0 }
        maskTex = [tex(.r16Float, [.shaderRead, .shaderWrite]), tex(.r16Float, [.shaderRead, .shaderWrite])].compactMap { $0 }
        rawTex = tex(.r16Float, [.shaderRead, .shaderWrite])
        tmpTex = tex(.r16Float, [.shaderRead, .shaderWrite])
        maskCur = 0
        // boost: CPU 업로드용 shared r8 ping-pong (GPU가 이전 장을 읽는 중에도 안전)
        func sharedTex() -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: mw, height: mh, mipmapped: false)
            d.usage = [.shaderRead]; d.storageMode = .shared
            return device.makeTexture(descriptor: d)
        }
        boostTex = [sharedTex(), sharedTex()].compactMap { $0 }
        boostBitmap = [UInt8](repeating: 0, count: mw * mh)
        uploadedStamp = -1
        needsReset = true
    }

    /// 불연속(캡처 시작/장면 전환) — 다음 업데이트에서 누적 리셋 (+이전 캡처의 텍스트 박스 폐기).
    public func reset() {
        needsReset = true; frames = 0
        boxLock.lock(); pendingBoxes = nil; boxesStamp += 1; lastSubmitAt = 0; resetGen += 1; boxLock.unlock()
    }

    /// 현재 소스 프레임으로 누적 갱신 + 마스크 산출. cb에 인코딩 (blit 직후 = 소스 준비됨).
    public func update(source: any MTLTexture, into cb: any MTLCommandBuffer) {
        guard Self.enabled, let pso else { return }
        ensure(srcW: source.width, srcH: source.height)
        guard meanTex.count == 2, sqTex.count == 2, maskTex.count == 2, boostTex.count == 2,
              let rawOut = rawTex,
              let enc = cb.makeComputeCommandEncoder() else { return }
        // 이번엔 반대 마스크 버퍼에 쓴다 — 직전 프레임 워프가 아직 옛 버퍼를 읽는 중일 수 있음.
        let maskWrite = 1 - maskCur
        // 텍스트 박스 갱신 확인 — 새 제출 또는 TTL 만료 시 CPU 비트맵 그려 ping-pong 업로드
        boxLock.lock()
        let stamp = boxesStamp
        let boxes = pendingBoxes
        let age = CFAbsoluteTimeGetCurrent() - lastSubmitAt
        boxLock.unlock()
        if stamp != uploadedStamp || (boostActive && age > Self.boxTTL) {
            uploadedStamp = stamp
            boostActive = age <= Self.boxTTL && !(boxes?.isEmpty ?? true)
            boostBitmap.withUnsafeMutableBufferPointer { _ = memset($0.baseAddress, 0, $0.count) }
            if age <= Self.boxTTL, let boxes {
                for r in boxes {
                    // 소폭 팽창 (텍스트 라인 박스는 타이트 — 가로 0.5%, 세로 라인높이 40%)
                    let x0 = max(0, Int((r.minX - 0.005) * CGFloat(w)))
                    let x1 = min(w, Int((r.maxX + 0.005) * CGFloat(w)))
                    let y0 = max(0, Int((r.minY - r.height * 0.4) * CGFloat(h)))
                    let y1 = min(h, Int((r.maxY + r.height * 0.4) * CGFloat(h)))
                    guard x1 > x0, y1 > y0 else { continue }
                    for y in y0..<y1 {
                        boostBitmap.withUnsafeMutableBufferPointer { buf in
                            _ = memset(buf.baseAddress! + y * w + x0, 255, x1 - x0)
                        }
                    }
                }
            }
            boostCur = 1 - boostCur
            boostTex[boostCur].replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                                       withBytes: boostBitmap, bytesPerRow: w)
        }

        let prev = cur, next = 1 - cur
        enc.setComputePipelineState(pso)
        enc.setTexture(source, index: 0)
        enc.setTexture(meanTex[prev], index: 1)
        enc.setTexture(sqTex[prev], index: 2)
        enc.setTexture(meanTex[next], index: 3)
        enc.setTexture(sqTex[next], index: 4)
        enc.setTexture(rawOut, index: 5)   // 블러 전 raw — 아래 두 패스를 거쳐 maskTex로 간다
        enc.setTexture(boostTex[boostCur], index: 6)
        var p = SIMD4<Float>(Self.alpha, Self.clo, Self.chi, needsReset ? 1 : 0)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        var strength = Self.strength
        enc.setBytes(&strength, length: MemoryLayout<Float>.size, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()

        // 분리형 가우시안 두 패스: raw → tmp(가로) → maskTex[maskWrite](세로).
        // 같은 커맨드 버퍼 안이라 인코더 순서가 의존성을 보장한다.
        if let blurPso, let tmp = tmpTex {
            for (srcT, dstT, ax) in [(rawOut, tmp, SIMD2<Int32>(1, 0)),
                                     (tmp, maskTex[maskWrite], SIMD2<Int32>(0, 1))] {
                guard let benc = cb.makeComputeCommandEncoder() else { break }
                benc.setComputePipelineState(blurPso)
                benc.setTexture(srcT, index: 0)
                benc.setTexture(dstT, index: 1)
                var a = ax
                benc.setBytes(&a, length: MemoryLayout<SIMD2<Int32>>.size, index: 0)
                benc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
                                          threadsPerThreadgroup: tg)
                benc.endEncoding()
            }
        }
        cur = next
        maskCur = maskWrite   // 방금 쓴 버퍼가 이제 유효 마스크
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
        texture2d<float, access::read>  boost [[texture(6)]],   // Vision 텍스트 박스 (r8)
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
        float m = smoothstep(p.y, p.z, cons) * structured;
        // Vision 텍스트 박스 부스트 — 일관성이 놓친 흐린 텍스트 라인을 커버 (오프라인 GT +0.002)
        m = max(m, boost.read(gid).r);
        maskOut.write(float4(m * strength), gid);
    }

    // **마스크 공간 가우시안 블러 (분리형 5탭, σ=2).**
    //
    // 왜 필요한가 (2026-08-28, 다중 에이전트 조사에서 실측으로 확정):
    // clo/chi/alpha는 research/rife/finetune/ui_tune_sweep.py로 스윕해 정한 값인데,
    // 그 하네스의 ema_mask() 마지막 줄은 `gblur(m, 5, 2)` — **마스크에 블러를 걸고 업샘플한다.**
    // 배포 셰이더에는 그 블러가 없었다. 그래서 같은 파라미터가 튜닝이 검증한 이득의
    // 절반만 냈다 (bench_frames 12시퀀스, short=360, 스윕과 동일 GT 프로토콜):
    //     스윕 하네스(블러 있음)  커버 15.18%  전체 +0.0092  ROI +0.0207
    //     배포 셰이더(블러 없음)  커버 11.50%  전체 +0.0047  ROI +0.0139
    //     블러만 추가             커버 11.47%  전체 +0.0084  ROI +0.0201   ← 회수
    // 결정적으로 **커버리지는 그대로인데 이득이 온다** — 마스크의 '면적'이 아니라
    // '공간 분포'가 전부였다. 블러가 없으면 마스크가 획 위에만 점점이 서고,
    // mfWarp의 mix(outc, nearestPix, uim)이 한 글자 안에서 프리즈와 워프를 격자로 섞는다.
    //
    // 하이패스 커널(3x3 박스 vs 스윕의 가우시안 σ1.5)도 불일치하지만 **그건 고치지 않는다** —
    // 같은 실측에서 hp만 맞추면 커버리지는 15%로 복원되는데 이득은 +0.0045/+0.0132로
    // 오히려 배포본보다 낮았다. 커버리지가 아니라 분포가 레버라는 증거이기도 하다.
    kernel void uiMaskBlur(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant int2 &axis [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
    {
        uint w = src.get_width(), h = src.get_height();
        if (gid.x >= w || gid.y >= h) return;
        // exp(-x²/2σ²), σ=2, 5탭 정규화
        const float wt[5] = { 0.1525, 0.2219, 0.2514, 0.2219, 0.1525 };
        float acc = 0.0;
        for (int k = -2; k <= 2; k++) {
            int2 q = int2(gid) + axis * k;
            q.x = clamp(q.x, 0, int(w) - 1);
            q.y = clamp(q.y, 0, int(h) - 1);
            acc += src.read(uint2(q)).r * wt[k + 2];
        }
        dst.write(float4(acc), gid);
    }
    """
}
