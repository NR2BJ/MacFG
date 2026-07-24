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
    private var meanTex: [any MTLTexture] = []   // ping-pong EMA(hp)
    private var sqTex: [any MTLTexture] = []      // ping-pong EMA(hp²)
    // 마스크도 ping-pong — cb1(blit+detector)을 copy 큐로 분리하면(O1-3), 다음 프레임 update가
    // 마스크를 쓰는 동안 이번 프레임 cb2(warp, work 큐)가 같은 마스크를 읽어 cross-queue 하자드가
    // 난다. 2버퍼로 번갈아 써서 읽는 버퍼와 쓰는 버퍼를 분리.
    private var maskTex: [any MTLTexture] = []
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
    public func submitTextBoxes(_ rects: [CGRect]) {
        boxLock.lock()
        pendingBoxes = rects
        boxesStamp += 1
        lastSubmitAt = CFAbsoluteTimeGetCurrent()
        boxLock.unlock()
    }

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
        guard let fn = lib.makeFunction(name: "uiStaticUpdate") else { throw InterpolationError.shaderCompilationFailed }
        pso = try await device.makeComputePipelineState(function: fn)
    }

    /// 해상도 세팅/재세팅 — 소스 크기의 1/2(텍스트 보존 + 저비용).
    private func ensure(srcW: Int, srcH: Int) {
        let mw = max(64, srcW / 2), mh = max(64, srcH / 2)
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
        boxLock.lock(); pendingBoxes = nil; boxesStamp += 1; lastSubmitAt = 0; boxLock.unlock()
    }

    /// 현재 소스 프레임으로 누적 갱신 + 마스크 산출. cb에 인코딩 (blit 직후 = 소스 준비됨).
    public func update(source: any MTLTexture, into cb: any MTLCommandBuffer) {
        guard Self.enabled, let pso else { return }
        ensure(srcW: source.width, srcH: source.height)
        guard meanTex.count == 2, sqTex.count == 2, maskTex.count == 2,
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
        enc.setTexture(maskTex[maskWrite], index: 5)
        enc.setTexture(boostTex[boostCur], index: 6)
        var p = SIMD4<Float>(Self.alpha, Self.clo, Self.chi, needsReset ? 1 : 0)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<Float>>.size, index: 0)
        var strength = Self.strength
        enc.setBytes(&strength, length: MemoryLayout<Float>.size, index: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
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
    """
}
