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

// MARK: - CLI Args

struct BenchConfig {
    var width: Int = 3840
    var height: Int = 2160
    var frames: Int = 30
    var engines: [String] = ["all"]
    var flowBase: Double? = nil

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
            default: break
            }
        }
        return config
    }
}

// MARK: - Test Pattern Generation

/// 수평 이동 줄무늬 + 원형 오브젝트 패턴 (GPU compute). shift로 A/B/GT 생성.
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
        float stripe = sin(x / p.barWidth * 3.14159) * 0.5 + 0.5;
        float vstripe = cos(y / (p.barWidth * 1.5) * 3.14159) * 0.3 + 0.5;
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

/// 간이 PSNR — 중앙 라인 최대 400px 샘플링 근사 (GPU→CPU readback)
func computePSNR(device: any MTLDevice, commandQueue: any MTLCommandQueue,
                 texA: any MTLTexture, texB: any MTLTexture) -> Double {
    let w = min(texA.width, texB.width), h = min(texA.height, texB.height)
    let sampleY = h / 2

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

    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ Metal not available"); return
    }
    guard let commandQueue = device.makeCommandQueue() else {
        print("❌ Cannot create command queue"); return
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
    allEngines.append(("blend", LegacyPairEngine(BlendInterpolator())))

    let selectedEngines: [(String, any PairInterpolationEngine)]
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
