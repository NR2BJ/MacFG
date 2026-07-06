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
    var pairDir: String? = nil      // 품질 A/B: 삼중항 디렉터리
    var pairEngine: String = "metalflow"
    var flowShort: Int? = nil       // RIFE flow 단변 (288/360/432)
    var rifeANE = false             // RIFE를 ANE로 (기본 GPU)
    var multiT = false              // 멀티-t 화질 벤치 (t별 PSNR — 24/30fps 경로 검증)
    var tripletsDir: String? = nil  // 실프레임 삼중항 디렉터리 (frame_NNN.png → A/GT/B 오프라인 측정)

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
            case "--pair-dir": if let v = args.popFirst() { config.pairDir = v }
            case "--engine": if let v = args.popFirst() { config.pairEngine = v }
            case "--flow-short": if let v = args.popFirst() { config.flowShort = Int(v) }
            case "--rife-ane": config.rifeANE = true
            case "--multi-t": config.multiT = true
            case "--triplets": if let v = args.popFirst() { config.tripletsDir = v }
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
        var frames: [(t: Float, texture: any MTLTexture)] = []
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
        if let dir = ProcessInfo.processInfo.environment["MACFG_DUMP"], label.hasPrefix("60fps"),
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

    var allEngines: [(String, any PairInterpolationEngine)] = [("metalflow", MetalFlowEngine())]
    if AppleFIEngine.isSupported { allEngines.append(("applefi", AppleFIEngine())) }
    if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) { allEngines.append(("rife", RIFEEngine())) }
    let sel = engineKeys.contains("all") ? allEngines : allEngines.filter { engineKeys.contains($0.0) }

    let dumpDir = ProcessInfo.processInfo.environment["MACFG_DUMP"]
    for (key, engine) in sel {
        do { try await engine.prepare(device: device) } catch { print("  \(key): prepare 실패"); continue }
        var psnrs: [Double] = []
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
        print("  \(key.padding(toLength: 10, withPad: " ", startingAt: 0)) 삼중항 PSNR avg=\(String(format: "%.2f", avg))dB  med=\(String(format: "%.2f", med))dB  min=\(String(format: "%.2f", mn))dB  (n=\(psnrs.count))")
    }
    print("\n⏱  실프레임 = 합성보다 압축노이즈·반투명·대모션 모두 포함. 높을수록 정확.")
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

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ Metal not available"); return
    }
    guard let commandQueue = device.makeCommandQueue() else {
        print("❌ Cannot create command queue"); return
    }

    // 품질 A/B 페어모드: 삼중항 디렉터리 처리 후 종료
    if let fs = config.flowShort { RIFEEngine.flowShortSide = fs }
    if config.rifeANE { RIFEEngine.useGPU = false }
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
        await runTripletMode(dir: td, engineKeys: config.engines, device: device, queue: commandQueue)
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
