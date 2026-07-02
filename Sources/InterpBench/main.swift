/// InterpBench — 4K 프레임 보간 엔진 벤치마크
///
/// 사용법: swift run InterpBench [--width 3840] [--height 2160] [--frames 30] [--engines all]
///
/// 측정 항목:
/// 1. 처리 시간 (ms/frame) — GPU 커밋부터 완료까지
/// 2. PSNR (dB) — 정적 프레임 기준 자기 자신 대비 (하한선)
/// 3. 모션 SSIM 근사 — 이동 패턴에서 보간 품질
/// 4. 일관성 — 처리 시간의 표준편차 (지터 예측)

import Metal
import Interpolation
import Foundation

// MARK: - CLI Args

struct BenchConfig {
    var width: Int = 3840
    var height: Int = 2160
    var frames: Int = 30
    var engines: [String] = ["all"]

    static func parse() -> BenchConfig {
        var config = BenchConfig()
        var args = CommandLine.arguments.dropFirst()
        while let arg = args.popFirst() {
            switch arg {
            case "--width":  if let v = args.popFirst() { config.width = Int(v) ?? config.width }
            case "--height": if let v = args.popFirst() { config.height = Int(v) ?? config.height }
            case "--frames": if let v = args.popFirst() { config.frames = Int(v) ?? config.frames }
            case "--engines": if let v = args.popFirst() { config.engines = v.split(separator: ",").map(String.init) }
            default: break
            }
        }
        return config
    }
}

// MARK: - Test Pattern Generation

/// 움직이는 줄무늬 패턴 생성 (GPU compute)
func createTestPattern(device: any MTLDevice, commandQueue: any MTLCommandQueue,
                       width: Int, height: Int, shift: Float) -> (any MTLTexture)? {
    let shaderSrc = """
    #include <metal_stdlib>
    using namespace metal;
    struct PatternParams { uint2 size; float shift; float barWidth; };
    kernel void generatePattern(
        texture2d<float, access::write> out [[texture(0)]],
        constant PatternParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.size.x || gid.y >= p.size.y) return;
        float x = float(gid.x) + p.shift;
        float y = float(gid.y);
        // 수평 이동하는 색상 줄무늬
        float stripe = sin(x / p.barWidth * 3.14159) * 0.5 + 0.5;
        float vstripe = cos(y / (p.barWidth * 1.5) * 3.14159) * 0.3 + 0.5;
        // 원형 오브젝트 (모션 테스트용)
        float2 center = float2(float(p.size.x) * 0.5 + p.shift * 3.0,
                               float(p.size.y) * 0.5 + p.shift * 1.5);
        float dist = length(float2(gid) - center);
        float circle = smoothstep(100.0, 80.0, dist);
        float r = mix(stripe, 1.0, circle);
        float g = mix(vstripe, 0.3, circle);
        float b = mix(stripe * vstripe, 0.1, circle);
        out.write(float4(r, g, b, 1.0), gid);
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

/// PSNR 계산 (GPU compute → CPU readback)
func computePSNR(device: any MTLDevice, commandQueue: any MTLCommandQueue,
                 texA: any MTLTexture, texB: any MTLTexture) -> Double {
    // 간이 PSNR: 중앙 라인 100px 샘플링으로 근사
    let w = texA.width, h = texA.height
    let sampleY = h / 2

    // CPU readable copy
    func readRow(_ tex: any MTLTexture) -> [UInt8] {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false
        )
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let shared = device.makeTexture(descriptor: desc),
              let cb = commandQueue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { return [] }
        blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: shared, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var row = [UInt8](repeating: 0, count: w * 4)
        shared.getBytes(&row, bytesPerRow: w * 4,
                       from: MTLRegion(origin: MTLOrigin(x: 0, y: sampleY, z: 0),
                                       size: MTLSize(width: w, height: 1, depth: 1)),
                       mipmapLevel: 0)
        return row
    }

    let rowA = readRow(texA)
    let rowB = readRow(texB)
    guard !rowA.isEmpty, !rowB.isEmpty else { return 0 }

    var mse: Double = 0
    let sampleCount = min(w, 400) // 최대 400픽셀 샘플
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
    if mse < 0.01 { return 99.0 } // 거의 동일
    return 10.0 * log10(255.0 * 255.0 / mse)
}

// MARK: - Benchmark Runner

struct BenchResult {
    let engineName: String
    let avgMs: Double
    let minMs: Double
    let maxMs: Double
    let stdMs: Double
    let psnrStatic: Double   // 정적 프레임 PSNR (A==B일 때)
    let psnrMotion: Double   // 모션 프레임 PSNR (ground truth 대비)
}

func benchmarkEngine(_ engine: any FrameInterpolator,
                     device: any MTLDevice,
                     commandQueue: any MTLCommandQueue,
                     config: BenchConfig) async -> BenchResult? {
    let w = config.width, h = config.height

    // Prepare
    do {
        try await engine.prepare(device: device)
    } catch {
        print("  ❌ prepare failed: \(error)")
        return nil
    }

    // Generate test frames: A→B 이동 패턴 (shift 0 → 30px)
    let frameA = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 0)
    let frameB = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 30)
    let groundTruth = createTestPattern(device: device, commandQueue: commandQueue, width: w, height: h, shift: 15)

    guard let frameA, let frameB, let groundTruth else {
        print("  ❌ test pattern creation failed")
        return nil
    }

    // Warmup 3 frames
    for _ in 0..<3 {
        guard let cb = commandQueue.makeCommandBuffer() else { continue }
        let _ = try? engine.interpolate(frameA: frameA, frameB: frameB, t: 0.5, commandBuffer: cb)
        cb.commit()
        await cb.completed()
    }

    // Benchmark
    var times: [Double] = []
    var lastOutput: (any MTLTexture)?

    for i in 0..<config.frames {
        guard let cb = commandQueue.makeCommandBuffer() else { continue }
        let start = CFAbsoluteTimeGetCurrent()

        let result = try? engine.interpolate(frameA: frameA, frameB: frameB, t: 0.5, commandBuffer: cb)
        cb.commit()
        await cb.completed()

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)

        if i == config.frames - 1, let result {
            lastOutput = result
        }
    }

    guard !times.isEmpty else { return nil }

    let avg = times.reduce(0, +) / Double(times.count)
    let minT = times.min() ?? 0
    let maxT = times.max() ?? 0
    let variance = times.map { ($0 - avg) * ($0 - avg) }.reduce(0, +) / Double(times.count)
    let std = sqrt(variance)

    // PSNR: 정적 테스트 (A==A 보간 → 자기 자신이어야 함)
    var psnrStatic: Double = 0
    if let cb = commandQueue.makeCommandBuffer() {
        if let staticResult = try? engine.interpolate(frameA: frameA, frameB: frameA, t: 0.5, commandBuffer: cb) {
            cb.commit()
            await cb.completed()
            psnrStatic = computePSNR(device: device, commandQueue: commandQueue, texA: frameA, texB: staticResult)
        } else {
            cb.commit()
            await cb.completed()
        }
    }

    // PSNR: 모션 테스트 (보간 결과 vs ground truth)
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

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ Metal not available"); return
    }
    guard let commandQueue = device.makeCommandQueue() else {
        print("❌ Cannot create command queue"); return
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MacFG Interpolation Benchmark                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Device: \(device.name.padding(toLength: 49, withPad: " ", startingAt: 0)) ║")
    print("║  Resolution: \(config.width)x\(config.height) (\(config.frames) frames)".padding(toLength: 63, withPad: " ", startingAt: 0) + "║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    // 사용할 엔진들 구성
    let allEngines: [(String, any FrameInterpolator)] = [
        ("identity", IdentityCopyInterpolator()),
        ("mvboost", MVBoostInterpolator()),
        ("blend", BlendInterpolator()),
        ("fastmotion", FastMotionInterpolator()),
    ]

    let selectedEngines: [(String, any FrameInterpolator)]
    if config.engines.contains("all") {
        selectedEngines = allEngines
    } else {
        selectedEngines = allEngines.filter { config.engines.contains($0.0) }
    }

    var results: [BenchResult] = []

    for (key, engine) in selectedEngines {
        print("▶ \(engine.name) (\(key))")
        print("  warming up + benchmarking \(config.frames) frames...")

        if let result = await benchmarkEngine(engine, device: device, commandQueue: commandQueue, config: config) {
            results.append(result)
            let budget8ms = result.avgMs <= 8.0 ? "✅" : "❌"
            print("  ⏱  avg=\(String(format: "%.1f", result.avgMs))ms  min=\(String(format: "%.1f", result.minMs))ms  max=\(String(format: "%.1f", result.maxMs))ms  σ=\(String(format: "%.1f", result.stdMs))ms  \(budget8ms) 8ms budget")
            print("  📊 PSNR static=\(String(format: "%.1f", result.psnrStatic))dB  motion=\(String(format: "%.1f", result.psnrMotion))dB")
            print()
        } else {
            print("  ❌ benchmark failed\n")
        }
    }

    // Summary table
    if !results.isEmpty {
        print("┌─────────────────────────┬────────┬────────┬────────┬───────┬────────┬────────┐")
        print("│ Engine                  │ avg ms │ min ms │ max ms │ σ ms  │ static │ motion │")
        print("├─────────────────────────┼────────┼────────┼────────┼───────┼────────┼────────┤")
        for r in results {
            let name = r.engineName.padding(toLength: 23, withPad: " ", startingAt: 0)
            let budget = r.avgMs <= 8.0 ? "✅" : "❌"
            print("│ \(name) │ \(String(format: "%5.1f", r.avgMs))\(budget)│ \(String(format: "%6.1f", r.minMs)) │ \(String(format: "%6.1f", r.maxMs)) │ \(String(format: "%5.1f", r.stdMs)) │ \(String(format: "%5.1f", r.psnrStatic))dB│ \(String(format: "%5.1f", r.psnrMotion))dB│")
        }
        print("└─────────────────────────┴────────┴────────┴────────┴───────┴────────┴────────┘")
        print()
        print("📋 PSNR: 높을수록 좋음 (>30dB good, >40dB excellent)")
        print("⏱  8ms budget: 120Hz 디스플레이에서 지터 없는 최대 처리 시간")
        print("σ: 낮을수록 일관적 (지터 적음)")
    }
}

// Entry point
await main()
