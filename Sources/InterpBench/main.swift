import Interpolation

/// InterpBench — 프레임 보간 엔진 벤치마크 (실사용 엔진 대상)
///
/// 사용법: swift run InterpBench [--width 3840] [--height 2160] [--frames 30]
///                              [--engines metalflow,applefi,blend] [--flow-base 960]
///
/// 측정: 처리 시간(ms/frame, GPU 커밋→완료), 일관성(σ), 정적/모션 PSNR.
/// 대상은 앱이 실제로 쓰는 PairInterpolationEngine(MetalFlow/AppleFI) + Blend 베이스라인.
/// 향후 MetalFlow 모델 튜닝(flowBase/커널/피라미드) 시 회귀 측정 도구.

import Metal
import Interpolation
import Foundation
import Monitoring
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

/// MTLTexture(bgra8) → PNG (육안 검증용)
func dumpPNG(_ tex: any MTLTexture, device: any MTLDevice, queue: any MTLCommandQueue, path: String) {
    let w = tex.width, h = tex.height
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.storageMode = .shared; desc.usage = [.shaderRead]
    guard let shared = device.makeTexture(descriptor: desc),
          let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return }
    blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: .init(x: 0, y: 0, z: 0),
              sourceSize: .init(width: w, height: h, depth: 1), to: shared,
              destinationSlice: 0, destinationLevel: 0, destinationOrigin: .init(x: 0, y: 0, z: 0))
    blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    shared.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: w, height: h, depth: 1)), mipmapLevel: 0)
    // BGRA → RGBA
    for i in stride(from: 0, to: bytes.count, by: 4) { bytes.swapAt(i, i + 2) }
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let img = ctx.makeImage(),
          let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
}

// MARK: - CLI Args

struct BenchConfig {
    var width: Int = 3840
    var height: Int = 2160
    var frames: Int = 30
    var engines: [String] = ["all"]
    var flowBase: Double? = nil
    var occDirectional = false
    var smoothness: Float? = nil
    var boundary: Float? = nil     // MetalFlow 폴백 블렌드 폭 (0=crisp … 1=soft)
    var pairDir: String? = nil      // 품질 A/B: 삼중항 디렉터리
    var pairEngine: String = "metalflow"
    var flowShort: Int? = nil       // RIFE flow 단변 (288/360/432)
    var rifeANE = false             // RIFE를 ANE로 (기본 GPU)
    var multiT = false              // 멀티-t 화질 벤치 (t별 PSNR — 24/30fps 경로 검증)
    var tripletsDir: String? = nil  // 실프레임 삼중항 디렉터리 (frame_NNN.png → A/GT/B 오프라인 측정)
    var qualityAB = false           // 같은 삼중항에 MetalFlow 화질 변경 4단계를 전부 돌려 비교
    var uiSweep = false             // B2: 정지-UI 마스크 파라미터 스윕 (--triplets와 함께)
    /// 디텍터 갱신 주기(프레임). 앱은 6(AppState `uiDetectFrame % 6`)이고 벤치 기본은 1이다 —
    /// 6으로 맞추면 EMA 시간 상수가 앱과 같아져 결과를 그대로 옮길 수 있다(대신 워밍업 450장).
    var uiStride = 1
    var tinStride: Int? = nil       // A5: 틴 밀도 — stride N에서 t=1/N..(N-1)/N을 실프레임 GT로 (--triplets와 함께)

    static func parse() -> BenchConfig {
        var config = BenchConfig()
        var args = CommandLine.arguments.dropFirst()
        while let arg = args.popFirst() {
            switch arg {
            case "--width":  if let v = args.popFirst() { config.width = Int(v) ?? config.width }
            case "--height": if let v = args.popFirst() { config.height = Int(v) ?? config.height }
            case "--frames": if let v = args.popFirst() { config.frames = Int(v) ?? config.frames }
            case "--engines": if let v = args.popFirst() { config.engines = v.split(separator: ",").map(String.init) }
            case "--flow-base": if let v = args.popFirst() { config.flowBase = Double(v) }
            case "--occ-dir": config.occDirectional = true
            case "--smoothness": if let v = args.popFirst() { config.smoothness = Float(v) }
            case "--boundary": if let v = args.popFirst() { config.boundary = Float(v) }
            case "--pair-dir": if let v = args.popFirst() { config.pairDir = v }
            case "--engine": if let v = args.popFirst() { config.pairEngine = v }
            case "--flow-short": if let v = args.popFirst() { config.flowShort = Int(v) }
            case "--rife-ane": config.rifeANE = true
            case "--multi-t": config.multiT = true
            case "--triplets": if let v = args.popFirst() { config.tripletsDir = v }
            case "--quality-ab": config.qualityAB = true
            case "--tin-density": if let v = args.popFirst() { config.tinStride = Int(v) }
            case "--ui-sweep": config.uiSweep = true
            case "--ui-stride": if let v = args.popFirst() { config.uiStride = max(1, Int(v) ?? 1) }
            default: break
            }
        }
        return config
    }
}

// MARK: - Test Pattern Generation

/// 비주기 테스트 패턴 (GPU compute). shift로 A/B/GT 생성 — 세 요소 모두 shift에 선형이라
/// GT(shift=중간)가 정확한 참 중간프레임:
///   ① 비주기 값노이즈 배경 (1× 병진) — 주기 패턴의 aliasing 이점 제거
///   ② 3× 빠른 오클루더 박스 — 가림/드러남 경계(플로우 최약점) 측정
///   ③ 회전 스포크 휠 (각속도 일정) — 블록매칭이 못 따라가는 회전
func createTestPattern(device: any MTLDevice, commandQueue: any MTLCommandQueue,
                       width: Int, height: Int, shift: Float) -> (any MTLTexture)? {
    let shaderSrc = """
    #include <metal_stdlib>
    using namespace metal;
    struct PatternParams { uint2 size; float shift; float barWidth; };

    float vhash(float2 p) {
        return fract(sin(dot(floor(p), float2(127.1, 311.7))) * 43758.5453);
    }
    float vnoise(float2 p) {
        float2 i = floor(p), f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = vhash(i), b = vhash(i + float2(1,0));
        float c = vhash(i + float2(0,1)), d = vhash(i + float2(1,1));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
    }

    kernel void generatePattern(
        texture2d<float, access::write> out [[texture(0)]],
        constant PatternParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.size.x || gid.y >= p.size.y) return;
        float2 px = float2(gid);
        float2 sz = float2(p.size);

        // ① 비주기 값노이즈 배경 (다중 옥타브, 1× 수평 병진)
        float2 bp = (px - float2(p.shift, 0.0)) * 0.012;
        float n = vnoise(bp) * 0.6 + vnoise(bp * 2.7) * 0.3 + vnoise(bp * 6.1) * 0.1;
        float3 c = float3(n * 0.85, n * 0.72, n * 0.55);

        // ③ 회전 스포크 휠 (중앙 좌측, 각속도 = shift 선형). 프레임당 ~12°(0.007×30/2)로
        // 현실적 빠른 회전 — 이전 43°/frame는 병리적이라 PSNR을 회전 실패가 지배했음.
        float2 wc = sz * float2(0.34, 0.5);
        float2 wd = px - wc; float wr = length(wd);
        float wheelR = min(sz.x, sz.y) * 0.20;
        if (wr < wheelR) {
            float ang = atan2(wd.y, wd.x) + p.shift * 0.007;
            float spokes = step(0.0, sin(ang * 8.0));
            float rings = step(0.5, fract(wr / (wheelR * 0.22)));
            c = mix(float3(0.12, 0.16, 0.42), float3(0.95, 0.88, 0.28), spokes * 0.7 + rings * 0.3);
            if (wr > wheelR * 0.93) c = float3(0.9);
        }

        // ② 3× 빠른 오클루더 박스 (배경/휠 위를 가로질러 가림·드러남 생성)
        float occX = sz.x * 0.30 + p.shift * 3.0;
        float2 bl = float2(occX, sz.y * 0.32);
        float2 tr = bl + float2(150.0, sz.y * 0.36);
        if (px.x >= bl.x && px.x < tr.x && px.y >= bl.y && px.y < tr.y) {
            // 오클루더 자체도 내부 텍스처(균일색이면 경계만 평가됨) — 대각 줄
            float band = step(0.5, fract((px.x - px.y) / 24.0));
            c = mix(float3(0.9, 0.35, 0.1), float3(0.7, 0.2, 0.05), band);
        }

        // ④ 고정 오버레이 (shift 무관 — 게임 HUD/조준점/글자 모사).
        // GT에도 같은 위치에 있으므로 보간이 이걸 흔들면 PSNR이 그대로 벌점.
        float2 ctr = sz * 0.5;
        float2 dc = abs(px - ctr);
        // 조준점: 십자 (팔 길이 30, 두께 3, 중심 갭 6)
        bool crossArm = (dc.x < 1.5 && dc.y > 6.0 && dc.y < 30.0) ||
                        (dc.y < 1.5 && dc.x > 6.0 && dc.x < 30.0);
        if (crossArm) c = float3(0.2, 1.0, 0.3);
        // 글자 블록: 좌상단 미세 가로줄 텍스트 모사 (300×80, 4px 주기)
        if (px.x > 40.0 && px.x < 340.0 && px.y > 40.0 && px.y < 120.0) {
            float line = step(0.5, fract(px.y / 4.0));
            float word = step(0.25, fract(px.x / 37.0));   // 단어 간격 모사
            c = mix(c, float3(0.95), line * word * 0.9);
        }

        // ⑤ 반투명 패널 위 정적 텍스트 (실증상: 자막/채팅 오버레이 — 반투명 배경으로 움직이는
        // 영상이 비쳐 픽셀값은 매 프레임 바뀌지만, 텍스트 글자 구조는 고정). 순수 |A-B| 정적판정이
        // 실패하는 핵심 케이스. panelAlpha만큼 배경① 비침 → 픽셀 요동, 텍스트는 고정 위치.
        if (px.x > sz.x * 0.55 && px.x < sz.x * 0.95 && px.y > sz.y * 0.68 && px.y < sz.y * 0.88) {
            // 패널 밑: 고대비 이동 대각 스트라이프 (실제 영상/게임처럼 강한 flow 유발 —
            // 이게 반투명 통해 텍스트를 끌어당김). shift에 선형 이동.
            float stripe = step(0.5, fract(((px.x + px.y) - p.shift * 2.5) / 34.0));
            c = mix(float3(0.10, 0.12, 0.18), float3(0.75, 0.80, 0.90), stripe);
            float panelAlpha = 0.5;                         // 50% 배경 비침 (움직임 요동원)
            c = mix(c, float3(0.06, 0.07, 0.10), panelAlpha);
            // 정적 텍스트 — 얇은 세로획(4px 주기, 1.5px 폭)이라 base 해상도(0.5×)에선 sub-pixel로
            // 움직이는 배경과 뭉개짐 = 실제 자막/채팅 텍스트 스케일. shift 무관 고정.
            float glyph = step(0.62, fract(px.x / 4.0)) * step(0.28, fract(px.y / 8.0));
            c = mix(c, float3(0.96, 0.96, 0.92), glyph * 0.95);
        }

        out.write(float4(c, 1.0), gid);
    }
    """

    guard let lib = try? device.makeLibrary(source: shaderSrc, options: nil),
          let fn = lib.makeFunction(name: "generatePattern"),
          let pso = try? device.makeComputePipelineState(function: fn) else { return nil }

    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
    )
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = .private
    guard let tex = device.makeTexture(descriptor: desc) else { return nil }

    struct PatternParams {
        var size: SIMD2<UInt32>
        var shift: Float
        var barWidth: Float
    }
    var params = PatternParams(
        size: SIMD2(UInt32(width), UInt32(height)),
        shift: shift,
        barWidth: 60.0
    )

    guard let cb = commandQueue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { return nil }
    enc.setComputePipelineState(pso)
    enc.setTexture(tex, index: 0)
    enc.setBytes(&params, length: MemoryLayout<PatternParams>.size, index: 0)
    enc.dispatchThreadgroups(
        MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return tex
}

/// 간이 PSNR — 지정 행(기본 중앙) 최대 400px 샘플링 근사 (GPU→CPU readback)
func computePSNR(device: any MTLDevice, commandQueue: any MTLCommandQueue,
                 texA: any MTLTexture, texB: any MTLTexture, sampleYFrac: Double = 0.5) -> Double {
    let w = min(texA.width, texB.width), h = min(texA.height, texB.height)
    let sampleY = min(h - 1, max(0, Int(Double(h) * sampleYFrac)))

    func readRow(_ tex: any MTLTexture) -> [UInt8] {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: tex.width, height: tex.height, mipmapped: false
        )
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let shared = device.makeTexture(descriptor: desc),
              let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: tex.width, height: tex.height, depth: 1),
                  to: shared, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var row = [UInt8](repeating: 0, count: tex.width * 4)
        shared.getBytes(&row, bytesPerRow: tex.width * 4,
                       from: MTLRegion(origin: MTLOrigin(x: 0, y: sampleY, z: 0),
                                       size: MTLSize(width: tex.width, height: 1, depth: 1)),
                       mipmapLevel: 0)
        return row
    }

    let rowA = readRow(texA)
    let rowB = readRow(texB)
    guard !rowA.isEmpty, !rowB.isEmpty else { return 0 }

    var mse: Double = 0
    let sampleCount = min(w, 400)
    let step = max(1, w / sampleCount)
    var count = 0
    for x in stride(from: 0, to: w, by: step) {
        for c in 0..<3 {
            let diff = Double(rowA[x * 4 + c]) - Double(rowB[x * 4 + c])
            mse += diff * diff
        }
        count += 1
    }
    mse /= Double(count * 3)
    if mse < 0.01 { return 99.0 }
    return 10.0 * log10(255.0 * 255.0 / mse)
}

// MARK: - Benchmark Runner

struct BenchResult {
    let engineName: String
    let avgMs: Double
    let minMs: Double
    let maxMs: Double
    let stdMs: Double
    let psnrStatic: Double
    let psnrMotion: Double
}

/// 단일 t=0.5 보간을 인코딩·커밋하고 완료까지 대기, 출력 텍스처 반환
func encodeOne(_ engine: any PairInterpolationEngine,
               a: any MTLTexture, b: any MTLTexture,
               tsA: CFTimeInterval, tsB: CFTimeInterval,
               queue: any MTLCommandQueue) async -> (any MTLTexture)? {
    guard let cb = queue.makeCommandBuffer() else { return nil }
    let result = engine.encodePair(stableA: a, stableB: b, tsA: tsA, tsB: tsB,
                                   tValues: [0.5], into: cb)
    cb.commit()
    await cb.completed()
    return result?.frames.first?.texture
}

/// 멀티-t 화질 벤치 — 24/30fps 소스의 복수 t 경로 검증.
/// 패턴이 shift에 선형이라 임의 t의 GT를 정확 생성: GT(t) = pattern(shift = 30·t).
/// t별 PSNR로 "t=0.5 predict + 선형 스케일 근사"와 "t별 exact predict"의 차이를 정량화.
func benchmarkMultiT(_ engine: any PairInterpolationEngine, key: String,
                     device: any MTLDevice, commandQueue: any MTLCommandQueue,
                     config: BenchConfig) async {
    do { try await engine.prepare(device: device) } catch {
        print("  ❌ prepare failed: \(error)"); return
    }
    let w = config.width, h = config.height
    guard let a = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 0),
          let b = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 30) else {
        print("  ❌ pattern failed"); return
    }
    // (라벨, t셋, 소스 fps 모사 dt)
    let cases: [(String, [Float], Double)] = [
        ("60fps(단일)", [0.5], 1.0 / 60.0),
        ("30fps→120", [0.25, 0.5, 0.75], 1.0 / 30.0),
        ("24fps→120", [0.2, 0.4, 0.6, 0.8], 1.0 / 24.0),
    ]
    print("▶ \(engine.name) (\(key)) — 멀티-t 화질")
    var ts = 0.0
    for (label, tset, dt) in cases {
        // 워밍업 2회 (파이프/EMA)
        for _ in 0..<2 {
            guard let cb = commandQueue.makeCommandBuffer() else { break }
            _ = engine.encodePair(stableA: a, stableB: b, tsA: ts, tsB: ts + dt, tValues: tset, into: cb)
            cb.commit(); await cb.completed(); ts += dt
        }
        var encodeMs: [Double] = []
        var frames: [(t: Float, texture: any MTLTexture, stamp: UInt64)] = []
        for i in 0..<8 {
            guard let cb = commandQueue.makeCommandBuffer() else { break }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = engine.encodePair(stableA: a, stableB: b, tsA: ts, tsB: ts + dt, tValues: tset, into: cb)
            cb.commit(); await cb.completed()
            encodeMs.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            ts += dt
            if i == 7, let result { frames = result.frames }
        }
        let avg = encodeMs.isEmpty ? 0 : encodeMs.reduce(0, +) / Double(encodeMs.count)
        var psnrParts: [String] = []
        for f in frames {
            guard let gt = createTestPattern(device: device, commandQueue: commandQueue,
                                             width: w, height: h, shift: 30 * f.t) else { continue }
            let p = computePSNR(device: device, commandQueue: commandQueue, texA: gt, texB: f.texture)
            // 반투명 패널 행(0.75) — 정적 텍스트 선명도 (핵심 지표)
            let pPanel = computePSNR(device: device, commandQueue: commandQueue, texA: gt, texB: f.texture, sampleYFrac: 0.75)
            psnrParts.append(String(format: "t=%.2f: %.1f/패널%.1fdB", f.t, p, pPanel))
        }
        let budget = dt * 1000
        print("  \(label)  encode=\(String(format: "%.1f", avg))ms/pair (예산 \(String(format: "%.0f", budget))ms)  \(psnrParts.joined(separator: "  "))")
        // 육안 덤프 (MACFG_DUMP=<dir>) — 60fps t=0.5 보간 + GT
        if let dir = Knob.string("MACFG_DUMP"), label.hasPrefix("60fps"),
           let mid = frames.first(where: { abs($0.t - 0.5) < 0.01 }),
           let gt = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 15) {
            dumpPNG(mid.texture, device: device, queue: commandQueue, path: "\(dir)/\(key)_interp.png")
            dumpPNG(gt, device: device, queue: commandQueue, path: "\(dir)/\(key)_gt.png")
        }
    }
    engine.shutdown()
    print()
}

/// 두 bgra8 텍스처의 증폭 차이(|a-b|*amp)를 PNG로 — 텍스트 뭉개짐 유무 결정적 판정
func dumpDiffPNG(_ a: any MTLTexture, _ b: any MTLTexture, amp: Int, device: any MTLDevice, queue: any MTLCommandQueue, path: String) {
    func readAll(_ tex: any MTLTexture) -> [UInt8] {
        let w = tex.width, h = tex.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared; desc.usage = [.shaderRead]
        guard let sh = device.makeTexture(descriptor: desc), let cb = queue.makeCommandBuffer(),
              let bl = cb.makeBlitCommandEncoder() else { return [] }
        bl.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: w, height: h, depth: 1), to: sh, destinationSlice: 0,
                destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var d = [UInt8](repeating: 0, count: w * h * 4)
        sh.getBytes(&d, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return d
    }
    let w = min(a.width, b.width), h = min(a.height, b.height)
    let ra = readAll(a), rb = readAll(b)
    guard !ra.isEmpty, !rb.isEmpty else { return }
    let rowA = a.width * 4, rowB = b.width * 4
    var out = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h { for x in 0..<w {
        for c in 0..<3 {
            let dv = abs(Int(ra[y * rowA + x * 4 + c]) - Int(rb[y * rowB + x * 4 + c])) * amp
            out[(y * w + x) * 4 + c] = UInt8(min(255, dv))
        }
    }}
    let cs = CGColorSpaceCreateDeviceRGB()
    out.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dst, img, nil); CGImageDestinationFinalize(dst)
    }
}

/// PNG → bgra8 MTLTexture (앱 덤프 프레임 로드)
func loadTexture(path: String, device: any MTLDevice) -> (any MTLTexture)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = img.width, h = img.height
    var data = [UInt8](repeating: 0, count: w * h * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    for i in stride(from: 0, to: data.count, by: 4) { data.swapAt(i, i + 2) }   // RGBA→BGRA
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.usage = [.shaderRead]; desc.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: desc) else { return nil }
    tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: data, bytesPerRow: w * 4)
    return tex
}

/// 전체 프레임 PSNR (그리드 서브샘플) — 실프레임 비교용
/// 블렌드 기준선 PSNR — (A+B)/2 vs GT. 모션보상 없이 두 프레임을 섞기만 한 것.
/// 엔진이 이걸 못 넘으면 "flow가 실제로 버는 게 없다"는 뜻이고, 크게 넘으면 모션보상이 작동 중이며
/// 남은 오차는 워프로 만들 수 없는 것(가려짐 해제·신규 픽셀)이라는 해석이 선다.
func computeBlendPSNR(device: any MTLDevice, queue: any MTLCommandQueue,
                      texA: any MTLTexture, texB: any MTLTexture, gt: any MTLTexture) -> Double {
    func readAll(_ tex: any MTLTexture) -> [UInt8] {
        let w = tex.width, h = tex.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared; desc.usage = [.shaderRead]
        guard let shared = device.makeTexture(descriptor: desc), let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: shared, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var b = [UInt8](repeating: 0, count: w * h * 4)
        shared.getBytes(&b, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return b
    }
    let a = readAll(texA), b = readAll(texB), g = readAll(gt)
    guard !a.isEmpty, a.count == b.count, a.count == g.count else { return 0 }
    var sum = 0.0; var n = 0
    var i = 0
    while i + 2 < a.count {
        for c in 0..<3 {
            let blend = (Double(a[i + c]) + Double(b[i + c])) * 0.5
            let d = blend - Double(g[i + c])
            sum += d * d; n += 1
        }
        i += 4 * 7   // 그리드 서브샘플 (computePSNRFull과 동급 근사)
    }
    guard n > 0 else { return 0 }
    let mse = sum / Double(n)
    return mse <= 0 ? 99 : 10 * log10(255 * 255 / mse)
}

/// **정적 영역 이탈량** — 텍스트/UI 흔들림을 재는 지표. (staticDev dB, 높을수록 안정)
///
/// 왜 필요한가: 삼중항 PSNR은 t=0.5에서 정답과의 **공간적 정확도**를 잰다. 그런데 사용자가
/// 실제로 거슬려 하는 "정적 UI 텍스트가 프레임마다 미세하게 흔들리는" 현상은 **시간적**이다.
/// 실측 2026-07-25: 화질 변경 3건이 삼중항 PSNR을 +0.361dB 올렸는데 눈으로는 더 나빠 보였다.
/// 두 측정이 서로 다른 것을 재고 있었던 것이다. 그 간극을 메운다.
///
/// 정의: 소스에서 정지한 픽셀(|A−B| ≤ tol)만 골라, 보간 결과가 A에서 얼마나 벗어났는지 잰다.
/// 안 움직여야 할 픽셀은 A와 같아야 하므로, 이탈량이 곧 흔들림이다.
/// - Returns: (dB, 정적 판정 픽셀 비율). 비율이 낮으면 표본이 적어 dB를 신뢰하면 안 된다.
func computeStaticDeviation(device: any MTLDevice, queue: any MTLCommandQueue,
                            texA: any MTLTexture, texB: any MTLTexture,
                            interp: any MTLTexture, tol: Double = 3.0) -> (db: Double, staticFrac: Double) {
    let a = readTextureBytes(texA, device: device, queue: queue)
    let b = readTextureBytes(texB, device: device, queue: queue)
    let p = readTextureBytes(interp, device: device, queue: queue)
    guard !a.isEmpty, a.count == b.count, a.count == p.count else { return (0, 0) }
    let w = min(min(texA.width, texB.width), interp.width)
    let h = min(min(texA.height, texB.height), interp.height)
    let rowA = texA.width * 4, rowB = texB.width * 4, rowP = interp.width * 4
    var sum = 0.0, n = 0.0, total = 0.0
    for y in stride(from: 0, to: h, by: 2) {
        for x in stride(from: 0, to: w, by: 2) {
            total += 1
            // 세 채널 모두 정지해야 정적으로 인정 — 한 채널만 보면 색만 바뀌는 이동을 놓친다
            var isStatic = true
            for c in 0..<3 where abs(Double(a[y * rowA + x * 4 + c]) - Double(b[y * rowB + x * 4 + c])) > tol {
                isStatic = false
            }
            guard isStatic else { continue }
            for c in 0..<3 {
                let d = Double(p[y * rowP + x * 4 + c]) - Double(a[y * rowA + x * 4 + c])
                sum += d * d; n += 1
            }
        }
    }
    guard n > 0 else { return (0, 0) }
    let mse = sum / n
    let db = mse <= 0 ? 99 : 10 * log10(255 * 255 / mse)
    return (db, total > 0 ? (n / 3) / total : 0)
}

/// 텍스처를 BGRA8 바이트로 읽어온다 (private면 shared로 블릿).
func readTextureBytes(_ tex: any MTLTexture, device: any MTLDevice, queue: any MTLCommandQueue) -> [UInt8] {
    let w = tex.width, h = tex.height
    if tex.storageMode == .shared {
        var b = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&b, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return b
    }
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.storageMode = .shared; desc.usage = [.shaderRead]
    guard let shared = device.makeTexture(descriptor: desc), let cb = queue.makeCommandBuffer(),
          let blit = cb.makeBlitCommandEncoder() else { return [] }
    blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1), to: shared, destinationSlice: 0,
              destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    var b = [UInt8](repeating: 0, count: w * h * 4)
    shared.getBytes(&b, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    return b
}

/// **선명도 비율 = 출력의 그래디언트 에너지 ÷ 정답의 그래디언트 에너지.**
///
/// PSNR만으로 화질을 판정하면 **흐리게 만드는 변경이 이긴다.** 잘 알려진 성질이고
/// 이 벤치에서도 그대로 보인다 — 모션 보상이 전혀 없는 `blend`(A와 B의 단순 평균)가
/// `hold`보다 2.45dB 높다. 평균을 내는 것만으로 점수가 오른다는 뜻이다.
/// 그래서 오늘 낸 화질 판정들(마스크 블러 +0.0084, occ-dir −0.118dB, flow 해상도 −0.12dB)이
/// 진짜 개선인지 **더 흐려진 것**인지 PSNR 하나로는 구분할 수 없다.
///
/// 1.0 = 정답만큼 선명 · <1 = 더 흐림 · >1 = 과선명(링잉·노이즈 증폭)
/// PSNR과 **함께** 읽어야 한다: PSNR이 올랐는데 이 값이 떨어졌다면 흐려서 이긴 것이다.
func computeSharpnessRatio(device: any MTLDevice, queue: any MTLCommandQueue,
                           out: any MTLTexture, gt: any MTLTexture) -> Double {
    func gradEnergy(_ tex: any MTLTexture) -> Double {
        let w = tex.width, h = tex.height
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: tex.pixelFormat,
                                                         width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]; d.storageMode = .shared
        guard let shared = device.makeTexture(descriptor: d),
              let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return 0 }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: shared, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var px = [UInt8](repeating: 0, count: w * h * 4)
        shared.getBytes(&px, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let row = w * 4
        func lum(_ x: Int, _ y: Int) -> Double {
            let i = y * row + x * 4
            return 0.299 * Double(px[i]) + 0.587 * Double(px[i + 1]) + 0.114 * Double(px[i + 2])
        }
        // 전진차분의 절대값 합 — PSNR과 같은 2픽셀 스트라이드로 비용을 맞춘다.
        var acc = 0.0, n = 0.0
        var y = 1
        while y < h - 1 {
            var x = 1
            while x < w - 1 {
                acc += abs(lum(x + 1, y) - lum(x - 1, y)) + abs(lum(x, y + 1) - lum(x, y - 1))
                n += 1; x += 2
            }
            y += 2
        }
        return n > 0 ? acc / n : 0
    }
    let g = gradEnergy(gt)
    guard g > 0 else { return 0 }
    return gradEnergy(out) / g
}

func computePSNRFull(device: any MTLDevice, queue: any MTLCommandQueue, texA: any MTLTexture, texB: any MTLTexture) -> Double {
    func readAll(_ tex: any MTLTexture) -> [UInt8] {
        let w = tex.width, h = tex.height
        if tex.storageMode == .shared {
            var b = [UInt8](repeating: 0, count: w * h * 4)
            tex.getBytes(&b, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            return b
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared; desc.usage = [.shaderRead]
        guard let shared = device.makeTexture(descriptor: desc), let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1), to: shared, destinationSlice: 0,
                  destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        var b = [UInt8](repeating: 0, count: w * h * 4)
        shared.getBytes(&b, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        return b
    }
    let w = min(texA.width, texB.width), h = min(texA.height, texB.height)
    let a = readAll(texA), b = readAll(texB)
    guard !a.isEmpty, !b.isEmpty else { return 0 }
    let rowA = texA.width * 4, rowB = texB.width * 4
    var mse = 0.0, count = 0.0
    for y in stride(from: 0, to: h, by: 2) {
        for x in stride(from: 0, to: w, by: 2) {
            for c in 0..<3 {
                let d = Double(a[y * rowA + x * 4 + c]) - Double(b[y * rowB + x * 4 + c])
                mse += d * d; count += 1
            }
        }
    }
    guard count > 0, mse > 0 else { return 99 }
    return 10 * log10(255 * 255 / (mse / count))
}

/// 실프레임 삼중항 모드 — frame_NNN.png 연속 3장 (A, GT, B)에서 encodePair(A,B,t=0.5) vs GT.
/// **A5 — 배수 천장: 앵커 근사가 극단 t에서 무너지는가 (실프레임).**
///
/// 24fps 소스를 144Hz에 ×6으로 채우면 틴이 5장(t=1/6..5/6)이다. 배포 경로는 앵커를 하나만
/// 뽑아 f0×(t/앵커), f1×((1−t)/(1−앵커))로 스케일하므로 t가 앵커에서 멀수록 근사가 무너진다.
/// 그게 사실이면 ×6은 성능이 아니라 **화질**로 막히는 것이고, 앵커를 늘리는 것 말고는 길이 없다.
///
/// **교락 제거가 이 모드의 요점이다.** 틴을 늘리면 브래킷(=소스 간격)이 같이 커져서 "배수"와
/// "모션 크기"가 섞인다. 그래서 두 팔을 **같은 브래킷 안에서** 비교한다:
///   deployed : tValues에 t를 전부 넘긴 한 번의 호출 = 엔진이 스스로 앵커를 정하는 배포 경로
///   exact    : t마다 tValues=[t]로 따로 호출 = 그 t에 앵커를 강제 (같은 브래킷, 같은 프레임)
/// 둘의 차이가 **순수한 근사 손실**이다. exact도 같이 나쁘면 원인은 근사가 아니라 브래킷이라
/// 앵커를 늘려도 소용없다.
///
/// `--multi-t`는 합성 패턴(순수 평행이동)을 쓰는데, 그건 선형 스케일 근사에 **가장 유리한**
/// 조건이라 "거기서 안 깨졌다"가 실콘텐츠를 보증하지 못한다. 여기서는 실프레임을 쓴다.
func runTinDensityMode(dir: String, strideN: Int, engineKeys: [String],
                       device: any MTLDevice, queue: any MTLCommandQueue) async {
    let fm = FileManager.default
    let files = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { $0.hasPrefix("frame_") && $0.hasSuffix(".png") }.sorted()
    guard files.count >= strideN + 1 else {
        print("❌ 프레임 부족 (\(files.count) < \(strideN + 1)) — \(dir)"); return
    }
    var frames: [any MTLTexture] = []
    for f in files { if let t = loadTexture(path: dir + "/" + f, device: device) { frames.append(t) } }
    guard frames.count >= strideN + 1 else { print("❌ 로드 실패"); return }
    let ts: [Float] = (1..<strideN).map { Float($0) / Float(strideN) }
    let srcFps = 60.0 / Double(strideN)
    print("▶ 틴 밀도: strideN=\(strideN) (브래킷 \(String(format: "%.1f", Double(strideN) * 1000.0 / 60.0))ms = 소스 \(String(format: "%.0f", srcFps))fps, 틴 \(ts.count)장)  \(dir)")
    print("  해상도 \(frames[0].width)x\(frames[0].height), 프레임 \(frames.count)장")

    var allEngines: [(String, any PairInterpolationEngine)] = [("metalflow", MetalFlowEngine())]
    if AppleFIEngine.isSupported { allEngines.append(("applefi", AppleFIEngine())) }
    if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) { allEngines.append(("rife", RIFEEngine())) }
    let sel = engineKeys.contains("all") ? allEngines : allEngines.filter { engineKeys.contains($0.0) }

    for (key, engine) in sel {
        do { try await engine.prepare(device: device) } catch { print("  \(key): prepare 실패"); continue }
        var dep: [Float: [Double]] = [:]      // t -> PSNR (배포 경로)
        var exa: [Float: [Double]] = [:]      // t -> PSNR (t별 앵커 강제)
        let dt = 1.0 / srcFps
        var n = 0
        for i in stride(from: 0, to: frames.count - strideN, by: max(1, strideN / 2)) {
            let a = frames[i], b = frames[i + strideN]
            n += 1
            // ① 배포 경로 — 한 번의 호출에 t 전부
            if let cb = queue.makeCommandBuffer() {
                let r = engine.encodePair(stableA: a, stableB: b, tsA: Double(i) * dt / Double(strideN),
                                          tsB: Double(i + strideN) * dt / Double(strideN), tValues: ts, into: cb)
                cb.commit(); await cb.completed()
                for f in (r?.frames ?? []) {
                    let k = Int((f.t * Float(strideN)).rounded())
                    guard k >= 1, k < strideN else { continue }
                    dep[f.t, default: []].append(
                        computePSNRFull(device: device, queue: queue, texA: f.texture, texB: frames[i + k]))
                }
            }
            // ② t별 앵커 강제 — 같은 쌍, 같은 브래킷
            for (k, t) in ts.enumerated() {
                guard let cb = queue.makeCommandBuffer() else { continue }
                let r = engine.encodePair(stableA: a, stableB: b, tsA: Double(i) * dt / Double(strideN),
                                          tsB: Double(i + strideN) * dt / Double(strideN), tValues: [t], into: cb)
                cb.commit(); await cb.completed()
                guard let tex = r?.frames.first?.texture else { continue }
                exa[t, default: []].append(
                    computePSNRFull(device: device, queue: queue, texA: tex, texB: frames[i + k + 1]))
            }
        }
        engine.shutdown()
        let avg: ([Double]) -> Double = { $0.isEmpty ? 0 : $0.reduce(0, +) / Double($0.count) }
        print("  \(key)  (쌍 \(n)개)")
        print("       t     배포     exact   근사손실")
        for t in ts {
            let d = avg(dep[t] ?? []), e = avg(exa[t] ?? [])
            let mark = (d == 0 || e == 0) ? "  (미산출)" : ""
            print(String(format: "   %6.3f %8.2f %8.2f %+9.2f%@", t, d, e, d - e, mark))
        }
    }
    print("\n판정: 같은 브래킷 안에서 극단 t(양끝)의 근사손실이 중앙 t보다 크게 나쁘면")
    print("      앵커 1개가 원인이다 = 앵커를 늘리면 낫는다.")
    print("      중앙과 극단이 비슷하면 브래킷(대모션) 탓이라 앵커를 늘려도 소용없고,")
    print("      배수 천장은 성능이 아니라 화질로 굳는다.")
}

func runTripletMode(dir: String, engineKeys: [String], device: any MTLDevice, queue: any MTLCommandQueue) async {
    let fm = FileManager.default
    let files = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { $0.hasPrefix("frame_") && $0.hasSuffix(".png") }.sorted()
    guard files.count >= 3 else { print("❌ 프레임 부족 (\(files.count)) — \(dir)"); return }
    print("▶ 실프레임 삼중항: \(files.count)장 → \(files.count - 2) 삼중항  (\(dir))")
    var frames: [any MTLTexture] = []
    for f in files { if let t = loadTexture(path: dir + "/" + f, device: device) { frames.append(t) } }
    guard frames.count >= 3 else { print("❌ 로드 실패"); return }
    print("  해상도 \(frames[0].width)x\(frames[0].height)")

    // ── 기준선: 보간을 아예 안 했을 때. 이게 없으면 엔진 점수의 의미를 알 수 없다.
    //  hold  = 직전 프레임 A를 그대로 표시 (프레임 홀드 = 보간 끔과 동등)
    //  엔진 PSNR이 hold보다 높아야 "보간이 실제로 벌었다"고 말할 수 있고,
    //  그 차이(gain)가 콘텐츠 난이도와 무관한 진짜 비교 지표다.
    var holdPSNRs: [Double] = []
    for i in 0..<(frames.count - 2) {
        holdPSNRs.append(computePSNRFull(device: device, queue: queue, texA: frames[i], texB: frames[i + 1]))
    }
    let holdAvg = holdPSNRs.isEmpty ? 0 : holdPSNRs.reduce(0, +) / Double(holdPSNRs.count)
    let holdMin = holdPSNRs.min() ?? 0
    print("  \("hold(보간없음)".padding(toLength: 16, withPad: " ", startingAt: 0)) avg=\(String(format: "%.2f", holdAvg))dB  min=\(String(format: "%.2f", holdMin))dB   ← 기준선①")
    var blendPSNRs: [Double] = []
    for i in 0..<(frames.count - 2) {
        blendPSNRs.append(computeBlendPSNR(device: device, queue: queue,
                                           texA: frames[i], texB: frames[i + 2], gt: frames[i + 1]))
    }
    let blendAvg = blendPSNRs.isEmpty ? 0 : blendPSNRs.reduce(0, +) / Double(blendPSNRs.count)
    let blendMin = blendPSNRs.min() ?? 0
    print("  \("blend(A+B)/2".padding(toLength: 16, withPad: " ", startingAt: 0)) avg=\(String(format: "%.2f", blendAvg))dB  min=\(String(format: "%.2f", blendMin))dB   ← 기준선② (모션보상 없음)")

    var allEngines: [(String, any PairInterpolationEngine)] = [("metalflow", MetalFlowEngine())]
    if AppleFIEngine.isSupported { allEngines.append(("applefi", AppleFIEngine())) }
    if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) { allEngines.append(("rife", RIFEEngine())) }
    let sel = engineKeys.contains("all") ? allEngines : allEngines.filter { engineKeys.contains($0.0) }

    let dumpDir = Knob.string("MACFG_DUMP")
    for (key, engine) in sel {
        do { try await engine.prepare(device: device) } catch { print("  \(key): prepare 실패"); continue }
        var psnrs: [Double] = []
        var sharps: [Double] = []
        var dt = 1.0 / 30.0
        for i in 0..<(frames.count - 2) {
            let a = frames[i], gt = frames[i + 1], b = frames[i + 2]
            // 워밍업 겸 실행 — 시간적 prior 있는 엔진 위해 순서대로
            guard let cb = queue.makeCommandBuffer() else { continue }
            let r = engine.encodePair(stableA: a, stableB: b, tsA: Double(i) * dt, tsB: Double(i + 2) * dt, tValues: [0.5], into: cb)
            cb.commit(); await cb.completed()
            guard let interp = r?.frames.first?.texture else { continue }
            let p = computePSNRFull(device: device, queue: queue, texA: interp, texB: gt)
            psnrs.append(p)
            sharps.append(computeSharpnessRatio(device: device, queue: queue, out: interp, gt: gt))
            if let dd = dumpDir, i == frames.count / 2 {
                dumpPNG(interp, device: device, queue: queue, path: "\(dd)/\(key)_t\(i)_interp.png")
                dumpPNG(gt, device: device, queue: queue, path: "\(dd)/\(key)_t\(i)_gt.png")
                dumpPNG(a, device: device, queue: queue, path: "\(dd)/\(key)_t\(i)_A.png")
                // 증폭 diff — interp vs GT (엔진 오차, 정적 텍스트 뭉개짐이면 여기 드러남)
                dumpDiffPNG(interp, gt, amp: 6, device: device, queue: queue, path: "\(dd)/\(key)_t\(i)_diff.png")
                // interp vs 소스A — 정적요소는 A와 동일해야(=검정), 다르면 정적요소 흔들림
                dumpDiffPNG(interp, a, amp: 6, device: device, queue: queue, path: "\(dd)/\(key)_t\(i)_diffA.png")
            }
        }
        engine.shutdown()
        let avg = psnrs.isEmpty ? 0 : psnrs.reduce(0, +) / Double(psnrs.count)
        let mn = psnrs.min() ?? 0
        let sorted = psnrs.sorted()
        let med = sorted.isEmpty ? 0 : sorted[sorted.count / 2]
        // **선명도를 PSNR 옆에 나란히 찍는다.** PSNR만 보면 흐리게 만드는 변경이 이긴다
        // (이 벤치에서 blend가 hold보다 2.45dB 높은 것이 그 증거다 — 평균만 내도 점수가 오른다).
        // sharp 1.0 = 정답만큼 선명, <1 = 더 흐림, >1 = 과선명. PSNR↑ + sharp↓ 면 흐려서 이긴 것이다.
        let shAvg = sharps.isEmpty ? 0 : sharps.reduce(0, +) / Double(sharps.count)
        print("  \(key.padding(toLength: 10, withPad: " ", startingAt: 0)) 삼중항 PSNR avg=\(String(format: "%.2f", avg))dB  med=\(String(format: "%.2f", med))dB  min=\(String(format: "%.2f", mn))dB  sharp=\(String(format: "%.3f", shAvg))  (n=\(psnrs.count))")
    }
    print("\n⏱  실프레임 = 합성보다 압축노이즈·반투명·대모션 모두 포함. 높을수록 정확.")
}

/// **화질 변경 A/B (결정론적)** — 같은 삼중항에 4단계를 전부 돌려 비교한다.
///
/// 왜 이 모드인가: 실사용 A/B는 매번 장면이 달라 판정이 흐려진다(사용자 실측:
/// "둘 다는 확실히 최악인데 각각은 off와 구분이 안 된다 — 매번 조건이 달라서 그런가").
/// 같은 입력에 4단계를 돌리면 그 confound가 사라지고, 상호작용(각각은 무해한데 함께면
/// 나빠지는지)도 드러난다.
///
/// 두 지표를 같이 본다 — 하나만 보면 이 세션에서 겪은 함정에 다시 빠진다:
///  - PSNR      : t=0.5 정답과의 공간적 정확도. 높을수록 정확.
///  - staticDev : 정지한 픽셀이 원본에서 벗어난 정도. 높을수록 **텍스트/UI가 안정**.
/// 7/25 변경은 PSNR을 올리면서 staticDev를 떨어뜨렸을 가능성이 크다(눈에 보인 게 그쪽이다).
/// **B2 정지-UI 마스크 스윕 — 정답(GT) 기준 짝지은 A/B.**
///
/// `UIStaticDetector`는 앱에만 배선돼 있고 InterpBench엔 참조가 0건이었다 — B2가 "튜닝 미완"이던
/// 진짜 이유는 **잴 경로가 없었던 것**이다.
///
/// **첫 판(2026-09-01)은 지표를 틀리게 잡았다. 그 실패를 여기 남긴다.**
/// `staticDev`(정지 픽셀이 A에서 벗어난 정도)와 그 반대 지표로 만든 `freezeErr`를 썼는데,
/// 둘 다 **산술적으로 퇴화**한다:
///   - staticDev는 `|A−B| ≤ tol(3)`인 픽셀만 골라 `|interp − A|`를 잰다. 그런데 t=0.5의 프리즈
///     결과는 (A+B)/2(MetalFlow·RIFE) 또는 B(AppleFI)라 그 게이트가 오차에 **상한을 씌운다**
///     (각각 ≤1.5, ≤3 ⇒ 항상 ≥44.6dB, ≥38.6dB). 반면 정답 GT는 그 게이트에 안 묶여 상한이 없다.
///     **즉 "얼리기"가 "정답"을 14~21dB 이긴다.** 마스크를 키우면 좋아지는 게 당연하다.
///   - `freezeErr`도 `|interp − A|`라 프리즈 타깃(=(A+B)/2)과 정답이 A에서 비슷한 거리에 있고,
///     커버가 7~17%라 희석돼 판정 문턱에 닿지 못한다 → **모든 행이 후보로 통과**했다(실측).
///
/// **그래서 정답을 기준으로, 마스크가 실제로 덮는 영역에서, 짝지어 잰다.**
/// 파라미터 점마다 같은 프레임·같은 마스크로 두 팔을 돌린다:
///   on  = setUIMask(mask) 로 보간   /   off = setUIMask(nil) 로 보간
/// 그리고 **마스크 영역(ROI)에서만** GT 대비 PSNR을 비교한다. 차이가 곧 "이 픽셀들을 얼려서
/// 정답에 더 가까워졌는가"이고, 이건 마스크를 키운다고 공짜로 오르지 않는다.
/// full은 화면 전체 — ROI가 좋아지면서 full이 나빠지면 다른 데를 망친 것이다.
///
/// **dB는 평균 내지 않고 MSE를 풀링한다.** 삼중항별 dB 평균은 준정지 구간(99dB 포화)이 열 전체를
/// 끌어올린다 — 기존 코드의 결함이었다.
///
/// **EMA 워밍업**: 디텍터는 `frames >= 8`이어야 마스크를 내주고 alpha(기본 0.04)는 25갱신 창이다.
/// 여기서는 매 프레임 갱신하므로 창의 3배(75장)를 흘린 뒤 측정한다.
/// **주의**: 앱은 6프레임마다 갱신하므로(AppState `uiDetectFrame % 6`) 실효 창이 150 소스프레임이다
/// — 벤치와 앱의 시간 상수가 다르다. 여기 결과를 앱 값으로 그대로 옮기지 말 것.
func runUIMaskSweep(dir: String, engineKey: String, stride uiStride: Int,
                    device: any MTLDevice, queue: any MTLCommandQueue) async {
    let fm = FileManager.default
    let files = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { $0.hasPrefix("frame_") && $0.hasSuffix(".png") }.sorted()
    guard files.count >= 100 else {
        print("❌ 프레임 \(files.count)장 — EMA 수렴에 최소 100장 필요. bench_frames는 전부 12장이라 못 쓴다.")
        return
    }
    var frames: [any MTLTexture] = []
    for f in files { if let t = loadTexture(path: dir + "/" + f, device: device) { frames.append(t) } }
    guard frames.count >= 100 else { print("❌ 로드 실패"); return }
    let W = frames[0].width, H = frames[0].height
    // **워밍업은 스트라이드에 비례한다.** EMA 창은 alpha 기준 25 **갱신**이고, 갱신은
    // uiStride 프레임마다 일어난다 → 소스 프레임 기준 창 = 25 × stride. 3 시상수를 흘린다.
    // stride 1(벤치 기본)이면 75장, stride 6(앱과 동일)이면 450장이 필요하다.
    let warm = min(75 * uiStride, max(20, frames.count / 2))
    print("▶ 정지-UI 마스크 (GT 기준 짝A/B): \(frames.count)장 \(W)x\(H) (마스크 \(W/2)x\(H/2))"
          + " 갱신주기 \(uiStride)프레임 워밍업 \(warm)  \(dir)")
    if warm < 75 * uiStride {
        print("  ⚠️ 워밍업이 \(75 * uiStride)장에 못 미친다(프레임 부족) — EMA 미수렴. 결과를 앱 값으로 옮기지 말 것.")
    }

    // 마지막 항은 마스크 축소비 — 소스/div. 연산자가 마스크 픽셀 단위라 이 값이 곧
    // "연산자가 화면의 몇 퍼센트를 덮는가"를 정한다. 4K에서 div 4 = 1080p에서 div 2와 동일 비율.
    let points: [(String, Float, Float, Float, Float, Int)] = [
        ("현재 .04/.5/1.7", 0.04, 0.5, 1.7, 1.0, 2),
        ("clo 0.8",        0.04, 0.8, 2.0, 1.0, 2),
        ("clo 1.2",        0.04, 1.2, 2.4, 1.0, 2),
        ("clo 0.3",        0.04, 0.3, 1.5, 1.0, 2),
        ("창 50f(.02)",    0.02, 0.5, 1.7, 1.0, 2),
        ("창 12f(.08)",    0.08, 0.5, 1.7, 1.0, 2),
        ("strength 0.6",   0.04, 0.5, 1.7, 0.6, 2),
        ("div4 clo .5",    0.04, 0.5, 1.7, 1.0, 4),
        ("div4 clo 0.3",   0.04, 0.3, 1.5, 1.0, 4),
        ("div4 clo 0.8",   0.04, 0.8, 2.0, 1.0, 4),
        ("div3 clo .5",    0.04, 0.5, 1.7, 1.0, 3),
        // 어디서 뒤집히는지 — div4가 세 팔 전부에서 이겼으므로 더 밀어 본다.
        ("div6 clo 0.3",   0.04, 0.3, 1.5, 1.0, 6),
        ("div8 clo 0.3",   0.04, 0.3, 1.5, 1.0, 8),
        ("div6 clo .15",   0.04, 0.15, 1.2, 1.0, 6),
        ("div4 clo .15",   0.04, 0.15, 1.2, 1.0, 4),
    ]
    // cons 분모의 잡음 바닥 축 — 배포 파라미터를 고정한 채 이것만 바꾼다.
    // 깨끗한 소스와 잡음 있는 소스에서 **최적점이 다르면** 코덱 의존성이 여기 있다는 뜻이다.
    let epsPoints: [Float] = [0.001, 0.002, 0.004, 0.008, 0.016]
    // **반경 축 — maskDiv와 달리 마스크 해상도는 소스/2로 유지한다.**
    // div를 키우면 분석 스케일과 마스크 해상도가 같이 내려가 업샘플에서 뭉개졌다(선명도 −0.0040).
    // 반경만 키우면 분석 스케일만 넓어지므로, 검출 이득을 얻으면서 블러가 없어야 한다 —
    // 그 예측이 맞는지가 이 축의 전부다. 0 = 소스 해상도에서 자동(1080p→1, 4K→2).
    let radiusPoints: [Int] = [1, 2, 3, 0]
    print("  설정             cover  경도>.75  ROI이득   full이득   선명도Δ  갱신비용")
    var allPoints = points.map { ($0.0, $0.1, $0.2, $0.3, $0.4, $0.5, Float(0.004), 1) }
    for e in epsPoints {
        allPoints.append((String(format: "eps %.3f", e), 0.04, 0.5, 1.7, 1.0, 2, e, 1))
    }
    for r in radiusPoints {
        allPoints.append((r == 0 ? "R auto" : "R \(r) (\(2*r+1)x\(2*r+1))",
                          0.04, 0.5, 1.7, 1.0, 2, 0.004, r))
    }
    for pt in allPoints {
        let (name, alpha, clo, chi, strength, div, eps, rad) = pt
        UIStaticDetector.noiseEps = eps
        UIStaticDetector.hpRadius = rad
        UIStaticDetector.enabled = true
        UIStaticDetector.alpha = alpha; UIStaticDetector.clo = clo
        UIStaticDetector.chi = chi; UIStaticDetector.strength = strength
        UIStaticDetector.maskDiv = div

        let det = UIStaticDetector(device: device)
        try? await det.prepare(); det.reset()
        let engine: any PairInterpolationEngine
        switch engineKey {
        case "applefi": engine = AppleFIEngine()
        case "rife":    engine = RIFEEngine()
        default:        engine = MetalFlowEngine()
        }
        do { try await engine.prepare(device: device) } catch { print("  \(name): prepare 실패"); continue }

        // **디텍터 갱신 비용을 같이 잰다.** 박스 반경을 R로 키우면 텍스처 샘플이 (2R+1)²로
        // 늘어난다(3x3=9 → 7x7=49, 5.4배). 이 패스는 4K에서 cb1 +3ms이고 workQueue 백로그가
        // 그걸 +11ms work로 증폭한다고 기록돼 있다 — 화질 이득을 성능으로 사는 것이면 알아야 한다.
        var updMs: [Double] = []
        for i in stride(from: 0, to: warm, by: uiStride) {
            guard let cb = queue.makeCommandBuffer() else { break }
            let t0 = CFAbsoluteTimeGetCurrent()
            det.update(source: frames[i], into: cb); cb.commit(); await cb.completed()
            updMs.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        let updAvg = updMs.count > 8
            ? updMs.suffix(updMs.count - 8).reduce(0, +) / Double(updMs.count - 8) : 0
        // MSE 풀링 누적 — (roiOn, roiOff, fullOn, fullOff) 각각 (제곱합, 표본수)
        var acc = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        var covs: [Double] = [], hards: [Double] = [], shOn: [Double] = [], shOff: [Double] = []
        // 측정 삼중항 상한 — 4K에서 CPU 오차 누적이 병목이라 전수는 비싸다. 40개면 MSE 풀링에
        // 충분하고(픽셀 단위 표본이 4K 40삼중항 = 8천만 개), 파라미터 점 7개 × 두 팔이 감당된다.
        let measureStep = max(2, 2 * ((frames.count - warm) / 2 / 40))
        var i = warm
        while i + 2 < frames.count {
            let a = frames[i], gt = frames[i + 1], b = frames[i + 2]
            // 앱과 같은 주기로만 갱신한다 — 매 프레임 갱신하면 EMA 시간 상수가 앱과 달라진다.
            if (i - warm) % uiStride == 0 {
                guard let cbu = queue.makeCommandBuffer() else { break }
                det.update(source: a, into: cbu); cbu.commit(); await cbu.completed()
            }
            guard let mask = det.mask else { i += measureStep; continue }
            let ms = readMaskStats(mask, device: device, queue: queue)
            covs.append(ms.mean); hards.append(ms.hard)
            // **마스크 시각화 (B2, 여태 미구현).** 숫자로는 "무엇을 놓쳤나"를 못 본다.
            // 소스 위에 마스크를 초록으로 얹어 한 장 남긴다 — 어디가 얼려지고 어디가 안
            // 얼려지는지가 한눈에 보인다. 반투명 채팅처럼 배경이 비쳐 시간축 일관성이 깨지는
            // 영역을 디텍터가 잡는지 여부가 이 그림에서 바로 갈린다.
            if let dd = Knob.string("MACFG_MASKDUMP"), covs.count == 5 {
                dumpMaskOverlay(src: a, mask: mask, device: device, queue: queue,
                                path: "\(dd)/mask_\(name.replacingOccurrences(of: " ", with: "_")).png")
            }
            var outs: [any MTLTexture] = []
            for useMask in [true, false] {
                engine.setUIMask(useMask ? mask : nil)
                guard let cb = queue.makeCommandBuffer() else { break }
                let r = engine.encodePair(stableA: a, stableB: b, tsA: Double(i) / 60.0,
                                          tsB: Double(i + 2) / 60.0, tValues: [0.5], into: cb)
                cb.commit(); await cb.completed()
                guard let t = r?.frames.first?.texture else { break }
                // 링 슬롯 재사용으로 덮이기 전에 즉시 읽어야 한다 — 두 팔을 모아 뒀다 재면 안 된다.
                outs.append(t)
                // **선명도를 두 팔 각각 잰다.** full이득이 양수인데 선명도가 떨어졌다면
                // 마스크가 UI를 얼린 게 아니라 전체를 흐리게 해서 PSNR을 번 것이다.
                let sh = computeSharpnessRatio(device: device, queue: queue, out: t, gt: gt)
                if useMask { shOn.append(sh) } else { shOff.append(sh) }
                let (rs, rn, fs, fn) = maskedSquaredError(device: device, queue: queue,
                                                          out: t, gt: gt, mask: mask)
                let k = useMask ? 0 : 1
                acc[k].0 += rs; acc[k].1 += rn
                acc[k + 2].0 += fs; acc[k + 2].1 += fn
            }
            _ = outs
            i += measureStep
        }
        engine.shutdown()
        let db: ((Double, Double)) -> Double = { p in
            guard p.1 > 0 else { return 0 }
            let mse = p.0 / p.1
            return mse <= 0 ? 99 : 10 * log10(255 * 255 / mse)
        }
        let rOn = db(acc[0]), rOff = db(acc[1]), fOn = db(acc[2]), fOff = db(acc[3])
        let mean: ([Double]) -> Double = { $0.isEmpty ? 0 : $0.reduce(0, +) / Double($0.count) }
        let C = mean(covs), Hd = mean(hards), dSharp = mean(shOn) - mean(shOff)
        let lbl = name.count >= 16 ? name : name + String(repeating: " ", count: 16 - name.count)
        // 선명도가 떨어지면(−0.002 초과) 이득의 출처를 의심해야 한다 — 블러로 번 것일 수 있다.
        let blurry = dSharp < -0.002
        let mark = blurry ? "  ⚠️흐려짐" : ((fOn - fOff) > -0.02 ? "  ✅" : "  ✗")
        print(lbl + String(format: " %5.1f%% %5.1f%%  %+7.3f  %+7.3f  %+7.4f %6.2fms",
                           C * 100, Hd * 100, rOn - rOff, fOn - fOff, dSharp, updAvg) + mark)
    }
    print("\n  ROI = 마스크가 덮는 픽셀만, GT 기준. ROI이득 = 마스크 켬 − 끔 (짝지은 같은 프레임·같은 마스크).")
    print("  ✅ = ROI이득 > +0.02dB 이면서 full이득 > −0.02dB. 마스크를 키운다고 공짜로 오르지 않는다.")
}

/// 마스크 가중 제곱오차와 전체 제곱오차를 **한 번에** 낸다 — (ROI합, ROI표본, 전체합, 전체표본).
/// dB로 바꾸지 않고 합을 돌려주는 이유: 호출측이 시퀀스 전체를 **MSE로 풀링**해야 하기 때문이다.
/// 삼중항별 dB를 평균하면 준정지 구간(99dB 포화)이 결과를 끌어올린다.
func maskedSquaredError(device: any MTLDevice, queue: any MTLCommandQueue,
                        out: any MTLTexture, gt: any MTLTexture,
                        mask: any MTLTexture, thresh: Double = 0.25)
                        -> (Double, Double, Double, Double) {
    let o = readTextureBytes(out, device: device, queue: queue)
    let g = readTextureBytes(gt, device: device, queue: queue)
    guard !o.isEmpty, o.count == g.count else { return (0, 0, 0, 0) }
    let m = readMaskFloats(mask, device: device, queue: queue)
    guard !m.isEmpty else { return (0, 0, 0, 0) }
    let w = min(out.width, gt.width), h = min(out.height, gt.height)
    let rowO = out.width * 4, rowG = gt.width * 4
    let mw = mask.width, mh = mask.height
    var rs = 0.0, rn = 0.0, fs = 0.0, fn = 0.0
    for y in stride(from: 0, to: h, by: 2) {
        let my = min(mh - 1, y * mh / h)
        for x in stride(from: 0, to: w, by: 2) {
            let mx = min(mw - 1, x * mw / w)
            let inROI = Double(m[my * mw + mx]) >= thresh
            for c in 0..<3 {
                let d = Double(o[y * rowO + x * 4 + c]) - Double(g[y * rowG + x * 4 + c])
                let sq = d * d
                fs += sq; fn += 1
                if inROI { rs += sq; rn += 1 }
            }
        }
    }
    return (rs, rn, fs, fn)
}

/// 소스 위에 마스크를 얹은 진단 이미지. 초록 = 마스크가 얼리는 곳(밝을수록 강하게).
/// 숫자(커버·경도)는 **얼마나**를 말하지만 **어디를**은 말하지 않는다. 정지 UI 검출의
/// 실패는 대개 위치의 문제다 — 반투명 채팅을 통째로 놓쳤는지, 움직이는 배경을 잘못 잡았는지.
func dumpMaskOverlay(src: any MTLTexture, mask: any MTLTexture,
                     device: any MTLDevice, queue: any MTLCommandQueue, path: String) {
    let s = readTextureBytes(src, device: device, queue: queue)
    let m = readMaskFloats(mask, device: device, queue: queue)
    guard !s.isEmpty, !m.isEmpty else { return }
    let w = src.width, h = src.height, mw = mask.width, mh = mask.height
    var out = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        let my = min(mh - 1, y * mh / h)
        for x in 0..<w {
            let mx = min(mw - 1, x * mw / w)
            let v = Double(min(max(m[my * mw + mx], 0), 1))
            let i = y * w * 4 + x * 4
            // BGRA. 원본을 어둡게 깔고 마스크를 초록으로 더한다 — 겹쳐도 원본 구조가 보인다.
            let b = Double(s[i]), g = Double(s[i + 1]), r = Double(s[i + 2])
            out[i]     = UInt8(min(255, b * (1 - v * 0.6)))
            out[i + 1] = UInt8(min(255, g * (1 - v * 0.6) + 255 * v * 0.7))
            out[i + 2] = UInt8(min(255, r * (1 - v * 0.6)))
            out[i + 3] = 255
        }
    }
    // BGRA → RGBA 후 PNG (dumpPNG과 같은 경로)
    for i in stride(from: 0, to: out.count, by: 4) { out.swapAt(i, i + 2) }
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &out, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let img = ctx.makeImage(),
          let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                    UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
    print("  [MASKDUMP] \(path)")
}

/// r16Float 마스크를 Float 배열로 읽는다.
func readMaskFloats(_ mask: any MTLTexture, device: any MTLDevice, queue: any MTLCommandQueue) -> [Float] {
    let w = mask.width, h = mask.height
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: w, height: h, mipmapped: false)
    d.storageMode = .shared; d.usage = [.shaderRead, .shaderWrite]
    guard let dst = device.makeTexture(descriptor: d),
          let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return [] }
    blit.copy(from: mask, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: dst, destinationSlice: 0, destinationLevel: 0,
              destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    var raw = [UInt16](repeating: 0, count: w * h)
    dst.getBytes(&raw, bytesPerRow: w * 2, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    return raw.map { Float(Float16(bitPattern: $0)) }
}

/// 마스크 평균(프리즈 비율)과 **경도** — 0.75를 넘는 픽셀 비율.
///
/// 평균만 보면 속는다. 격자를 거칠게 하면(maskDiv↑) 업샘플이 마스크를 뭉개서 **화면 전체에
/// 옅은 블렌드**가 되는데, 평균은 그것도 "커버가 늘었다"로 읽는다. 그건 UI를 얼린 게 아니라
/// 전부를 살짝 흐리게 한 것이고, PSNR은 블러를 보상하므로 점수만 오른다(이 저장소에 기록된
/// 함정). 경도가 낮은데 평균만 높으면 그 경우다.
func readMaskStats(_ mask: any MTLTexture, device: any MTLDevice, queue: any MTLCommandQueue)
                   -> (mean: Double, hard: Double) {
    let f = readMaskFloats(mask, device: device, queue: queue)
    guard !f.isEmpty else { return (0, 0) }
    var sum = 0.0, hard = 0.0
    for v in f { sum += Double(v); if v > 0.75 { hard += 1 } }
    return (sum / Double(f.count), hard / Double(f.count))
}

func runQualityABMode(dir: String, device: any MTLDevice, queue: any MTLCommandQueue) async {
    let fm = FileManager.default
    let files = ((try? fm.contentsOfDirectory(atPath: dir)) ?? [])
        .filter { $0.hasPrefix("frame_") && $0.hasSuffix(".png") }.sorted()
    guard files.count >= 3 else { print("❌ 프레임 부족 (\(files.count)) — \(dir)"); return }
    var frames: [any MTLTexture] = []
    for f in files { if let t = loadTexture(path: dir + "/" + f, device: device) { frames.append(t) } }
    guard frames.count >= 3 else { print("❌ 로드 실패"); return }
    print("▶ 화질 A/B: \(frames.count)장 → \(frames.count - 2) 삼중항, \(frames[0].width)x\(frames[0].height)  (\(dir))")

    // (이름, conf 계열 켬, static 계열 조임)
    let stages: [(String, Bool, Bool)] = [
        ("0 이전동작", false, false),
        ("1 conf만",   true,  false),
        ("2 static만", false, true),
        ("3 둘다(현재)", true,  true),
    ]
    print("  \("단계".padding(toLength: 14, withPad: " ", startingAt: 0))  PSNR avg    PSNR min   staticDev  (정적비율)")

    var rows: [(String, Double, Double, Double)] = []
    for (name, conf, stat) in stages {
        MetalFlowEngine.confRel = conf ? 0.3 : 0.0
        MetalFlowEngine.confMax = conf ? 0.5 : 0.0
        MetalFlowEngine.staticLo = stat ? 0.004 : 0.008
        MetalFlowEngine.staticHi = stat ? 0.02 : 0.04

        let engine = MetalFlowEngine()
        do { try await engine.prepare(device: device) } catch { print("  \(name): prepare 실패"); continue }
        var psnrs: [Double] = [], devs: [Double] = [], fracs: [Double] = []
        let dt = 1.0 / 30.0
        for i in 0..<(frames.count - 2) {
            let a = frames[i], gt = frames[i + 1], b = frames[i + 2]
            guard let cb = queue.makeCommandBuffer() else { continue }
            let r = engine.encodePair(stableA: a, stableB: b, tsA: Double(i) * dt, tsB: Double(i + 2) * dt,
                                      tValues: [0.5], into: cb)
            cb.commit(); await cb.completed()
            guard let interp = r?.frames.first?.texture else { continue }
            psnrs.append(computePSNRFull(device: device, queue: queue, texA: interp, texB: gt))
            let sd = computeStaticDeviation(device: device, queue: queue, texA: a, texB: b, interp: interp)
            if sd.staticFrac > 0.02 { devs.append(sd.db); fracs.append(sd.staticFrac) }
        }
        engine.shutdown()
        let pAvg = psnrs.isEmpty ? 0 : psnrs.reduce(0, +) / Double(psnrs.count)
        let pMin = psnrs.min() ?? 0
        let dAvg = devs.isEmpty ? 0 : devs.reduce(0, +) / Double(devs.count)
        let fAvg = fracs.isEmpty ? 0 : fracs.reduce(0, +) / Double(fracs.count)
        rows.append((name, pAvg, pMin, dAvg))
        print(String(format: "  %@  %7.3f dB  %7.3f dB  %7.3f dB   (%.0f%%)",
                     name.padding(toLength: 14, withPad: " ", startingAt: 0), pAvg, pMin, dAvg, fAvg * 100))
    }

    guard let base = rows.first else { return }
    print("\n  ── 0(이전동작) 대비 차이 ──")
    for r in rows.dropFirst() {
        print(String(format: "  %@  PSNR %+.3f dB   staticDev %+.3f dB",
                     r.0.padding(toLength: 14, withPad: " ", startingAt: 0), r.1 - base.1, r.3 - base.3))
    }
    print("""

      PSNR      = 정답과의 공간 정확도 (높을수록 정확)
      staticDev = 정지 픽셀이 원본에서 벗어난 정도 (높을수록 텍스트/UI가 안정)
      정적비율  = 삼중항에서 정지로 판정된 픽셀 비율. 낮으면 staticDev 표본이 적어 신뢰도 낮음.
      기준선 없음이 정상: hold/blend는 정지 픽셀에서 정의상 완벽해 비교 대상이 아니다.
      단계별로 PSNR과 staticDev가 **반대로** 움직이면 그게 이 변경의 거래 조건이다.
    """)
}

func benchmarkEngine(_ engine: any PairInterpolationEngine,
                     device: any MTLDevice,
                     commandQueue: any MTLCommandQueue,
                     config: BenchConfig) async -> BenchResult? {
    let w = config.width, h = config.height
    let dt = 1.0 / 60.0

    do {
        try await engine.prepare(device: device)
    } catch {
        print("  ❌ prepare failed: \(error)")
        return nil
    }

    guard let frameA = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 0),
          let frameB = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 30),
          let groundTruth = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 15) else {
        print("  ❌ test pattern creation failed")
        return nil
    }

    // Warmup 3 (엔진 내부 파이프/시간적 prior 워밍업)
    var ts = 0.0
    for _ in 0..<3 {
        _ = await encodeOne(engine, a: frameA, b: frameB, tsA: ts, tsB: ts + dt, queue: commandQueue)
        ts += dt
    }

    var times: [Double] = []
    var lastOutput: (any MTLTexture)?
    for i in 0..<config.frames {
        let start = CFAbsoluteTimeGetCurrent()
        let out = await encodeOne(engine, a: frameA, b: frameB, tsA: ts, tsB: ts + dt, queue: commandQueue)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        ts += dt
        times.append(elapsed)
        if i == config.frames - 1 { lastOutput = out }
    }
    guard !times.isEmpty else { return nil }

    let avg = times.reduce(0, +) / Double(times.count)
    let minT = times.min() ?? 0
    let maxT = times.max() ?? 0
    let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
    let std = sqrt(variance)

    // 정적 PSNR: A==B 보간은 자기 자신이어야 함 (엔진 리셋 후 측정)
    engine.reset()
    var psnrStatic: Double = 0
    if let staticOut = await encodeOne(engine, a: frameA, b: frameA, tsA: ts, tsB: ts + dt, queue: commandQueue) {
        psnrStatic = computePSNR(device: device, commandQueue: commandQueue, texA: frameA, texB: staticOut)
    }

    // 모션 PSNR: 보간(t=0.5) vs ground truth(shift 15)
    var psnrMotion: Double = 0
    if let lastOutput {
        psnrMotion = computePSNR(device: device, commandQueue: commandQueue, texA: groundTruth, texB: lastOutput)
    }

    engine.shutdown()
    return BenchResult(
        engineName: engine.name,
        avgMs: avg, minMs: minT, maxMs: maxT, stdMs: std,
        psnrStatic: psnrStatic, psnrMotion: psnrMotion
    )
}

// MARK: - Main

func main() async {
    let config = BenchConfig.parse()
    if let base = config.flowBase { MetalFlowEngine.flowBaseLongSide = base }
    MetalFlowEngine.occlusionDirectional = config.occDirectional
    if let sm = config.smoothness { MetalFlowEngine.motionSmoothness = sm }
    if let bs = config.boundary { MetalFlowEngine.boundarySoftness = bs }

    // **적용된 설정을 되찍는다 — 플래그가 안 먹은 것을 '변화 없음'으로 오독하지 않게.**
    // 2026-08-31에 여섯 조건이 소수점까지 동일하게 나와 발견인 줄 알았는데, 셸 단어분리로
    // 인자가 통째로 전달돼 전부 기본값으로 돈 것이었다. 그 다음엔 --flow-base 1440을 줬는데
    // **기본값이 이미 1440**이라 또 무변화였다. 이 파일 48행 주석이 같은 함정을 이미 기록해뒀다
    // ("실측 시도가 전부 같은 값으로 돌았다"). 실패가 침묵이면 반드시 되찍어야 한다.
    print(String(format: "⚙ flowBase=%.0f  smoothness=%.2f  boundary=%.2f  occDir=%@  rifeShort=%d",
                 MetalFlowEngine.flowBaseLongSide, MetalFlowEngine.motionSmoothness,
                 MetalFlowEngine.boundarySoftness,
                 MetalFlowEngine.occlusionDirectional ? "on" : "off", RIFEEngine.flowShortSide))

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ Metal not available"); return
    }
    guard let commandQueue = device.makeCommandQueue() else {
        print("❌ Cannot create command queue"); return
    }

    // 품질 A/B 페어모드: 삼중항 디렉터리 처리 후 종료
    if let fs = config.flowShort { RIFEEngine.flowShortSide = fs }
    if config.rifeANE { RIFEEngine.useGPU = false }
    RIFEEngine.adaptiveLadder = false   // 벤치 = 고정 조건 (flowShortSide/useGPU 존중)
    if let pairDir = config.pairDir {
        if let sm = config.smoothness { MetalFlowEngine.motionSmoothness = sm }
        await runPairMode(engineKey: config.pairEngine, dir: pairDir, device: device, queue: commandQueue)
        return
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MacFG Interpolation Benchmark                         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Device: \(device.name.padding(toLength: 49, withPad: " ", startingAt: 0)) ║")
    let flowNote = config.flowBase.map { " flowBase=\(Int($0))" } ?? ""
    print("║  Resolution: \(config.width)x\(config.height) (\(config.frames) frames)\(flowNote)".padding(toLength: 63, withPad: " ", startingAt: 0) + "║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    // 앱이 실제로 쓰는 엔진 + Blend 베이스라인
    var allEngines: [(String, any PairInterpolationEngine)] = [
        ("metalflow", MetalFlowEngine()),
    ]
    if AppleFIEngine.isSupported {
        allEngines.append(("applefi", AppleFIEngine()))
    } else {
        print("ℹ️  AppleFI unsupported on this system — skipped")
    }
    if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) {
        allEngines.append(("rife", RIFEEngine()))
    }
    allEngines.append(("blend", LegacyPairEngine(BlendInterpolator())))

    let selectedEngines: [(String, any PairInterpolationEngine)]
    if config.engines.contains("all") {
        selectedEngines = allEngines
    } else {
        selectedEngines = allEngines.filter { config.engines.contains($0.0) }
    }

    if let td = config.tripletsDir {
        if config.uiSweep {
            await runUIMaskSweep(dir: td, engineKey: config.engines.first ?? "metalflow",
                                 stride: config.uiStride, device: device, queue: commandQueue)
        } else if let stride = config.tinStride {
            await runTinDensityMode(dir: td, strideN: stride, engineKeys: config.engines,
                                    device: device, queue: commandQueue)
        } else if config.qualityAB {
            await runQualityABMode(dir: td, device: device, queue: commandQueue)
        } else {
            await runTripletMode(dir: td, engineKeys: config.engines, device: device, queue: commandQueue)
        }
        return
    }

    if config.multiT {
        for (key, engine) in selectedEngines {
            await benchmarkMultiT(engine, key: key, device: device, commandQueue: commandQueue, config: config)
        }
        return
    }

    var results: [BenchResult] = []
    for (key, engine) in selectedEngines {
        print("▶ \(engine.name) (\(key))")
        print("  warming up + benchmarking \(config.frames) frames...")
        if let result = await benchmarkEngine(engine, device: device, commandQueue: commandQueue, config: config) {
            results.append(result)
            let budget = result.avgMs <= 8.0 ? "✅" : "⚠️"
            print("  ⏱  avg=\(String(format: "%.1f", result.avgMs))ms  min=\(String(format: "%.1f", result.minMs))ms  max=\(String(format: "%.1f", result.maxMs))ms  σ=\(String(format: "%.1f", result.stdMs))ms  \(budget) 8ms")
            print("  📊 PSNR static=\(String(format: "%.1f", result.psnrStatic))dB  motion=\(String(format: "%.1f", result.psnrMotion))dB")
            print()
        } else {
            print("  ❌ benchmark failed\n")
        }
    }

    if !results.isEmpty {
        print("┌─────────────────────────┬────────┬────────┬────────┬───────┬────────┬────────┐")
        print("│ Engine                  │ avg ms │ min ms │ max ms │ σ ms  │ static │ motion │")
        print("├─────────────────────────┼────────┼────────┼────────┼───────┼────────┼────────┤")
        for r in results {
            let name = r.engineName.padding(toLength: 23, withPad: " ", startingAt: 0)
            print("│ \(name) │ \(String(format: "%6.1f", r.avgMs)) │ \(String(format: "%6.1f", r.minMs)) │ \(String(format: "%6.1f", r.maxMs)) │ \(String(format: "%5.1f", r.stdMs)) │ \(String(format: "%5.1f", r.psnrStatic))dB│ \(String(format: "%5.1f", r.psnrMotion))dB│")
        }
        print("└─────────────────────────┴────────┴────────┴────────┴───────┴────────┴────────┘")
        print()
        print("⏱  8ms = 120Hz 무지터 예산 · σ 낮을수록 일관적 · PSNR 높을수록 정확 (motion은 합성 패턴 기준 근사)")
    }
}

await main()
