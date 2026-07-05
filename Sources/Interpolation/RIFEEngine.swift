@preconcurrency import Metal
@preconcurrency import CoreML
import Foundation
import Monitoring
import os

/// RIFE v4.25 신경망 보간 엔진 — CoreML flow(저해상도) + Metal 풀해상도 워프.
///
/// 실측 근거 (M4, 2026-07-05, research/rife/):
/// - 풀그래프(내부 warp 포함) .mlpackage: GPU가 전 사이즈에서 ANE보다 ~2배 빠름
///   (288/360/432 = 8.4/12.1/16.7ms GPU vs 13.9/20.8/30.2ms ANE — resample이 ANE에서 비효율)
/// - 화질(실영상 고모션 SSIM): rife 0.846~0.854 vs MetalFlow 0.785 / AppleFI 0.789.
///   flow 해상도 288→432 차이는 미미 (이득은 해상도가 아니라 학습된 flow 구조에서 옴)
/// - 실프레임 패리티: CoreML fp16 = PyTorch 대비 42~44dB (품질 보존)
///
/// 파이프라인 (encodePair는 렌더 스레드 — predict를 여기서 블로킹하면 120Hz 락 즉사):
///   encodePair(CPU ~0.3ms):
///     ① pack 커널을 자체 cb에 인코딩·커밋 (stableA/B → 모델크기 fp16 CHW 버퍼 + 루마 히스토그램)
///     ② pack cb 완료 핸들러 → 워커 큐에서 CoreML predict (8-17ms, flow/mask를 공유 버퍼에 기록)
///        → 성공/실패 무관 MTLSharedEvent signal (실패 시 버퍼는 0 = 50/50 블렌드로 무해 강등)
///     ③ 호출자 cb: encodeWait(event) → unpack(버퍼→flow/mask 텍스처) → t별 풀해상도 워프
///   GPU는 이벤트를 논블로킹 대기 — 렌더 스레드/표시 경로는 영향 없음.
///   타임라인 등재가 predict만큼 늦어지는 것은 적응형 지연 슬롯이 흡수.
///
/// 멀티 t: 단일 t는 정확 timestep으로 predict (arbitrary-t 활용 — vsync 그리드 정렬).
/// 복수 t는 t=0.5 한 번 predict 후 선형 스케일 워프 (scale0=t/0.5, scale1=(1-t)/0.5).
public final class RIFEEngine: PairInterpolationEngine, @unchecked Sendable {
    public let name = "Neural (RIFE)"

    /// flow 콘텐츠 단변 — 288/360/432 (Models/rife{N}.mlpackage 필요)
    public nonisolated(unsafe) static var flowShortSide: Int = 360
    /// CoreML 유닛 — true=GPU(전 사이즈 최속), false=ANE(느리지만 GPU를 비움)
    public nonisolated(unsafe) static var useGPU: Bool = true

    /// 모델 파일 존재 여부 (엔진 등록 가드용)
    public static func modelAvailable(short: Int) -> Bool {
        modelSourceURL(short: short) != nil
    }

    private let sceneCutIntersectionThreshold = 0.5

    private var device: (any MTLDevice)?
    private var model: MLModel?
    private var modelW = 0
    private var modelH = 0
    private var packQueue: (any MTLCommandQueue)?
    private var packPSO: (any MTLComputePipelineState)?
    private var unpackPSO: (any MTLComputePipelineState)?
    private var warpPSO: (any MTLComputePipelineState)?

    /// predict 파이프라인 슬롯 — 3개 링 (pack/predict/warp 중첩 + 배압)
    private final class Slot: @unchecked Sendable {
        let packBuf: any MTLBuffer      // 6ch fp16 CHW (모델 입력)
        let flowBuf: any MTLBuffer      // 4ch fp16 CHW (모델 출력 backing)
        let maskBuf: any MTLBuffer      // 1ch fp16 (pre-sigmoid)
        let statsBuf: any MTLBuffer     // uint32 x64 루마 히스토그램 (A32+B32)
        let flowTex: any MTLTexture     // rgba16Float 모델 크기
        let maskTex: any MTLTexture     // r16Float 모델 크기
        let event: any MTLSharedEvent
        var useCount: UInt64 = 0
        var busy = false
        init(packBuf: any MTLBuffer, flowBuf: any MTLBuffer, maskBuf: any MTLBuffer,
             statsBuf: any MTLBuffer, flowTex: any MTLTexture, maskTex: any MTLTexture,
             event: any MTLSharedEvent) {
            self.packBuf = packBuf; self.flowBuf = flowBuf; self.maskBuf = maskBuf
            self.statsBuf = statsBuf; self.flowTex = flowTex; self.maskTex = maskTex
            self.event = event
        }
    }
    private var slots: [Slot] = []
    private let slotLock = NSLock()

    // 출력 텍스처 링 (소스 크기 BGRA)
    private var outputPool: [any MTLTexture] = []
    private var outputIndex = 0
    private var outputWidth = 0
    private var outputHeight = 0

    /// predict 전용 직렬 워커 — 렌더/메인과 완전 분리
    private let worker = DispatchQueue(label: "com.macfg.rife.predict", qos: .userInitiated)
    private let cancelled = OSAllocatedUnfairLock(initialState: false)

    private let logger = Logger(subsystem: "com.macfg", category: "RIFE")
    private var pairCount = 0
    private let predictMsEMA = OSAllocatedUnfairLock(initialState: 0.0)

    public init() {}

    // MARK: - Model discovery

    /// 모델 탐색: 앱번들 Resources(.mlmodelc/.mlpackage) → $MACFG_MODELS → 실행파일 상위 Models/ → cwd/Models
    private static func modelSourceURL(short: Int) -> URL? {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let res = Bundle.main.resourceURL {
            candidates.append(res.appendingPathComponent("rife\(short).mlmodelc"))
            candidates.append(res.appendingPathComponent("rife\(short).mlpackage"))
        }
        if let env = ProcessInfo.processInfo.environment["MACFG_MODELS"] {
            candidates.append(URL(fileURLWithPath: env).appendingPathComponent("rife\(short).mlpackage"))
        }
        var dir = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.deletingLastPathComponent()
        for _ in 0..<7 {
            candidates.append(dir.appendingPathComponent("Models/rife\(short).mlpackage"))
            dir.deleteLastPathComponent()
        }
        candidates.append(URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("Models/rife\(short).mlpackage"))
        return candidates.first { fm.fileExists(atPath: $0.path) }
    }

    /// .mlpackage는 컴파일 필요 — Application Support에 mtime 스탬프 캐시
    private static func compiledModelURL(source: URL, short: Int) async throws -> URL {
        if source.pathExtension == "mlmodelc" { return source }
        let fm = FileManager.default
        let cacheRoot = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MacFG/mlcache")
        try? fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cached = cacheRoot.appendingPathComponent("rife\(short).mlmodelc")
        let stamp = cacheRoot.appendingPathComponent("rife\(short).stamp")
        let weightPath = source.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        let mtime = (try? fm.attributesOfItem(atPath: weightPath.path)[.modificationDate] as? Date)
            .map { String($0.timeIntervalSince1970) } ?? "unknown"
        if fm.fileExists(atPath: cached.path),
           let prev = try? String(contentsOf: stamp, encoding: .utf8), prev == mtime {
            return cached
        }
        let tmp = try await MLModel.compileModel(at: source)
        try? fm.removeItem(at: cached)
        try fm.moveItem(at: tmp, to: cached)
        try? mtime.write(to: stamp, atomically: true, encoding: .utf8)
        return cached
    }

    // MARK: - Prepare

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        let short = Self.flowShortSide
        guard let src = Self.modelSourceURL(short: short) else {
            logger.error("rife\(short) model not found")
            throw InterpolationError.notPrepared
        }
        let compiled = try await Self.compiledModelURL(source: src, short: short)
        let config = MLModelConfiguration()
        config.computeUnits = Self.useGPU ? .cpuAndGPU : .cpuAndNeuralEngine
        let model = try await MLModel.load(contentsOf: compiled, configuration: config)
        self.model = model

        // 모델 입력 크기 추출 (x: [1,6,H,W])
        guard let xDesc = model.modelDescription.inputDescriptionsByName["x"],
              let shape = xDesc.multiArrayConstraint?.shape, shape.count == 4 else {
            throw InterpolationError.notPrepared
        }
        modelH = shape[2].intValue
        modelW = shape[3].intValue

        packQueue = device.makeCommandQueue()
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ name: String) async throws -> any MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw InterpolationError.shaderCompilationFailed }
            return try await device.makeComputePipelineState(function: fn)
        }
        packPSO = try await pso("rifePack")
        unpackPSO = try await pso("rifeUnpack")
        warpPSO = try await pso("rifeWarp")

        // 슬롯 3개
        slots = []
        let planeBytes = modelW * modelH * 2
        for _ in 0..<3 {
            guard let packBuf = device.makeBuffer(length: planeBytes * 6, options: .storageModeShared),
                  let flowBuf = device.makeBuffer(length: planeBytes * 4, options: .storageModeShared),
                  let maskBuf = device.makeBuffer(length: planeBytes, options: .storageModeShared),
                  let statsBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
                  let event = device.makeSharedEvent() else {
                throw InterpolationError.textureAllocationFailed
            }
            let flowDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: modelW, height: modelH, mipmapped: false)
            flowDesc.usage = [.shaderRead, .shaderWrite]
            flowDesc.storageMode = .private
            let maskDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r16Float, width: modelW, height: modelH, mipmapped: false)
            maskDesc.usage = [.shaderRead, .shaderWrite]
            maskDesc.storageMode = .private
            guard let flowTex = device.makeTexture(descriptor: flowDesc),
                  let maskTex = device.makeTexture(descriptor: maskDesc) else {
                throw InterpolationError.textureAllocationFailed
            }
            slots.append(Slot(packBuf: packBuf, flowBuf: flowBuf, maskBuf: maskBuf,
                              statsBuf: statsBuf, flowTex: flowTex, maskTex: maskTex, event: event))
        }
        cancelled.withLock { $0 = false }
        DiagnosticLog.shared.log("[RIFE] prepared: \(modelW)x\(modelH) flow (\(short)p), unit=\(Self.useGPU ? "GPU" : "ANE")")
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
        guard let model, let packQueue, let packPSO, let unpackPSO, let warpPSO,
              !tValues.isEmpty, tsB > tsA else { return nil }

        // 슬롯 확보 — 모두 predict 중이면 이 쌍은 스킵 (자연 배압: 소스만 표시)
        let slot: Slot? = slotLock.withLock {
            if let s = slots.first(where: { !$0.busy }) { s.busy = true; s.useCount += 1; return s }
            return nil
        }
        guard let slot else { return nil }
        let signalValue = slot.useCount

        ensureOutputPool(width: stableB.width, height: stableB.height)
        let ts = tValues.prefix(6).map { min(max($0, 0.01), 0.99) }
        guard outputPool.count >= ts.count else { releaseSlot(slot); return nil }

        // predict timestep: 단일 t는 정확 위상, 복수 t는 0.5 기준 + 선형 스케일
        let tPred: Float = ts.count == 1 ? min(max(ts[0], 0.05), 0.95) : 0.5

        // ── ① pack: 자체 cb (flow/mask 0클리어 → 실패 시 50/50 블렌드 강등 보장)
        guard let packCB = packQueue.makeCommandBuffer() else { releaseSlot(slot); return nil }
        if let fill = packCB.makeBlitCommandEncoder() {
            fill.fill(buffer: slot.flowBuf, range: 0..<slot.flowBuf.length, value: 0)
            fill.fill(buffer: slot.maskBuf, range: 0..<slot.maskBuf.length, value: 0)
            fill.fill(buffer: slot.statsBuf, range: 0..<(64 * 4), value: 0)
            fill.endEncoding()
        }
        guard let packEnc = packCB.makeComputeCommandEncoder() else { releaseSlot(slot); return nil }
        packEnc.setComputePipelineState(packPSO)
        packEnc.setTexture(stableA, index: 0)
        packEnc.setTexture(stableB, index: 1)
        packEnc.setBuffer(slot.packBuf, offset: 0, index: 0)
        packEnc.setBuffer(slot.statsBuf, offset: 0, index: 1)
        var packParams = SIMD2<UInt32>(UInt32(modelW), UInt32(modelH))
        packEnc.setBytes(&packParams, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        dispatch(packEnc, width: modelW, height: modelH, pso: packPSO)
        packEnc.endEncoding()

        // ── ② pack 완료 → 워커에서 predict → 어떤 경로든 event signal
        let mW = modelW, mH = modelH
        let workerRef = worker
        let cancelledRef = cancelled
        let emaRef = predictMsEMA
        let loggerRef = logger
        packCB.addCompletedHandler { [weak model] _ in
            workerRef.async {
                defer { slot.event.signaledValue = signalValue }
                guard let model, !cancelledRef.withLock({ $0 }) else { return }
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let xArr = try MLMultiArray(
                        dataPointer: slot.packBuf.contents(),
                        shape: [1, 6, NSNumber(value: mH), NSNumber(value: mW)],
                        dataType: .float16,
                        strides: [NSNumber(value: 6 * mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                    let tArr = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
                    tArr[0] = NSNumber(value: tPred)
                    let flowBack = try MLMultiArray(
                        dataPointer: slot.flowBuf.contents(),
                        shape: [1, 4, NSNumber(value: mH), NSNumber(value: mW)],
                        dataType: .float16,
                        strides: [NSNumber(value: 4 * mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                    let maskBack = try MLMultiArray(
                        dataPointer: slot.maskBuf.contents(),
                        shape: [1, 1, NSNumber(value: mH), NSNumber(value: mW)],
                        dataType: .float16,
                        strides: [NSNumber(value: mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                    let opts = MLPredictionOptions()
                    opts.outputBackings = ["flow": flowBack, "mask": maskBack]
                    let input = try MLDictionaryFeatureProvider(dictionary: ["x": xArr, "t": tArr])
                    let result = try model.prediction(from: input, options: opts)
                    // backing 미사용 폴백 — 반환 배열이 우리 버퍼가 아니면 복사
                    if let f = result.featureValue(for: "flow")?.multiArrayValue,
                       f.dataPointer != slot.flowBuf.contents() {
                        f.withUnsafeBytes { raw in
                            slot.flowBuf.contents().copyMemory(
                                from: raw.baseAddress!, byteCount: min(raw.count, slot.flowBuf.length))
                        }
                    }
                    if let m = result.featureValue(for: "mask")?.multiArrayValue,
                       m.dataPointer != slot.maskBuf.contents() {
                        m.withUnsafeBytes { raw in
                            slot.maskBuf.contents().copyMemory(
                                from: raw.baseAddress!, byteCount: min(raw.count, slot.maskBuf.length))
                        }
                    }
                    let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    emaRef.withLock { $0 = $0 == 0 ? ms : $0 * 0.95 + ms * 0.05 }
                } catch {
                    loggerRef.error("predict failed: \(error.localizedDescription)")
                }
            }
        }
        packCB.commit()

        // ── ③ 호출자 cb: 이벤트 대기 → unpack → t별 워프
        commandBuffer.encodeWaitForEvent(slot.event, value: signalValue)
        guard let upEnc = commandBuffer.makeComputeCommandEncoder() else { releaseSlot(slot); return nil }
        upEnc.setComputePipelineState(unpackPSO)
        upEnc.setBuffer(slot.flowBuf, offset: 0, index: 0)
        upEnc.setBuffer(slot.maskBuf, offset: 0, index: 1)
        upEnc.setTexture(slot.flowTex, index: 0)
        upEnc.setTexture(slot.maskTex, index: 1)
        dispatch(upEnc, width: modelW, height: modelH, pso: unpackPSO)
        upEnc.endEncoding()

        var frames: [(t: Float, texture: any MTLTexture)] = []
        for t in ts {
            let output = outputPool[outputIndex]
            outputIndex = (outputIndex + 1) % outputPool.count
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { break }
            enc.setComputePipelineState(warpPSO)
            enc.setTexture(stableA, index: 0)
            enc.setTexture(stableB, index: 1)
            enc.setTexture(slot.flowTex, index: 2)
            enc.setTexture(slot.maskTex, index: 3)
            enc.setTexture(output, index: 4)
            var wp = WarpParams(
                flowScale: SIMD2<Float>(Float(output.width) / Float(modelW), Float(output.height) / Float(modelH)),
                scale0: t / tPred,
                scale1: (1 - t) / (1 - tPred))
            enc.setBytes(&wp, length: MemoryLayout<WarpParams>.size, index: 0)
            dispatch(enc, width: output.width, height: output.height, pso: warpPSO)
            enc.endEncoding()
            frames.append((t, output))
        }

        // 슬롯 반납은 호출자 cb 완료 시 (unpack/warp가 버퍼·텍스처를 다 읽은 뒤)
        let dumpPair = pairCount
        commandBuffer.addCompletedHandler { [weak self] _ in
            // 디버그: 패리티 진단용 버퍼 덤프 (MACFG_RIFE_DUMP=<dir>, 4번째 쌍 = 벤치 최종)
            if dumpPair == 3, let dir = ProcessInfo.processInfo.environment["MACFG_RIFE_DUMP"] {
                for (nm, buf) in [("pack", slot.packBuf), ("flow", slot.flowBuf), ("mask", slot.maskBuf)] {
                    let data = Data(bytes: buf.contents(), count: buf.length)
                    try? data.write(to: URL(fileURLWithPath: "\(dir)/rife_\(nm).bin"))
                }
            }
            self?.releaseSlot(slot)
        }

        pairCount += 1
        if pairCount <= 3 || pairCount % 600 == 0 {
            let ema = predictMsEMA.withLock { $0 }
            DiagnosticLog.shared.log("[RIFE] pair #\(pairCount) t×\(frames.count) predictEMA=\(String(format: "%.1f", ema))ms")
        }

        // 장면 전환: pack 히스토그램 교집합 (pack cb는 이 시점 완료가 보장됨 — 호출자 cb 완료 후 평가)
        let threshold = sceneCutIntersectionThreshold
        let statsBuf = slot.statsBuf
        let evaluator: @Sendable () -> Bool = {
            let ptr = statsBuf.contents().bindMemory(to: UInt32.self, capacity: 64)
            var totalA: UInt64 = 0
            var intersect: UInt64 = 0
            for i in 0..<32 {
                let a = UInt64(ptr[i]); let b = UInt64(ptr[32 + i])
                totalA += a; intersect += min(a, b)
            }
            guard totalA > 1000 else { return false }
            return Double(intersect) / Double(totalA) < threshold
        }
        return frames.isEmpty ? nil : PairEncodeResult(frames: frames, sceneCutEvaluator: evaluator)
    }

    private struct WarpParams {
        var flowScale: SIMD2<Float>
        var scale0: Float
        var scale1: Float
    }

    private func releaseSlot(_ slot: Slot) {
        slotLock.withLock { slot.busy = false }
    }

    private func ensureOutputPool(width: Int, height: Int) {
        guard let device else { return }
        guard width != outputWidth || height != outputHeight || outputPool.isEmpty else { return }
        outputWidth = width
        outputHeight = height
        outputPool = []
        outputIndex = 0
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        for _ in 0..<8 {
            if let tex = device.makeTexture(descriptor: desc) { outputPool.append(tex) }
        }
        DiagnosticLog.shared.log("[RIFE] output pool \(width)x\(height) x\(outputPool.count)")
    }

    private func dispatch(_ enc: any MTLComputeCommandEncoder, width: Int, height: Int, pso: any MTLComputePipelineState) {
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (width + 15) / 16, height: (height + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    // MARK: - Reset / Shutdown

    public func reset() {
        // 무상태 (쌍 독립 — 시간적 prior 없음). 진행 중 슬롯은 자연 완료.
    }

    public func shutdown() {
        cancelled.withLock { $0 = true }
        worker.sync {}          // 진행 중 predict 드레인 (이후 워커 항목은 cancelled로 즉시 signal)
        model = nil
        slots = []
        outputPool = []
        packQueue = nil
    }

    // MARK: - Shaders

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct WarpParams { float2 flowScale; float scale0; float scale1; };
    constant float3 kLuma = float3(0.2126, 0.7152, 0.0722);

    // stableA/B → 모델크기 fp16 CHW 6플레인 (RGB 0-1) + 루마 히스토그램 (장면 전환용)
    kernel void rifePack(
        texture2d<float, access::sample> imgA [[texture(0)]],
        texture2d<float, access::sample> imgB [[texture(1)]],
        device half* outBuf [[buffer(0)]],
        device atomic_uint* hist [[buffer(1)]],
        constant uint2& size [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= size.x || gid.y >= size.y) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(size);
        float3 a = imgA.sample(s, uv).rgb;
        float3 b = imgB.sample(s, uv).rgb;
        uint plane = size.x * size.y;
        uint idx = gid.y * size.x + gid.x;
        outBuf[idx]             = half(a.r);
        outBuf[plane + idx]     = half(a.g);
        outBuf[2 * plane + idx] = half(a.b);
        outBuf[3 * plane + idx] = half(b.r);
        outBuf[4 * plane + idx] = half(b.g);
        outBuf[5 * plane + idx] = half(b.b);
        if ((gid.x & 1) == 0 && (gid.y & 1) == 0) {
            uint binA = uint(clamp(dot(a, kLuma), 0.0, 0.999) * 32.0);
            uint binB = uint(clamp(dot(b, kLuma), 0.0, 0.999) * 32.0);
            atomic_fetch_add_explicit(&hist[binA], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&hist[32u + binB], 1u, memory_order_relaxed);
        }
    }

    // flow/mask fp16 CHW 버퍼 → 텍스처 (워프에서 하드웨어 bilinear 샘플용)
    kernel void rifeUnpack(
        device const half* flowBuf [[buffer(0)]],
        device const half* maskBuf [[buffer(1)]],
        texture2d<float, access::write> flowTex [[texture(0)]],
        texture2d<float, access::write> maskTex [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = flowTex.get_width();
        uint h = flowTex.get_height();
        if (gid.x >= w || gid.y >= h) return;
        uint plane = w * h;
        uint idx = gid.y * w + gid.x;
        flowTex.write(float4(float(flowBuf[idx]), float(flowBuf[plane + idx]),
                             float(flowBuf[2 * plane + idx]), float(flowBuf[3 * plane + idx])), gid);
        maskTex.write(float4(float(maskBuf[idx])), gid);
    }

    // 풀해상도 워프: I_t = warp(A, f_t0)·σ(mask) + warp(B, f_t1)·(1-σ(mask))
    // flow는 모델픽셀 단위 → flowScale로 풀해상도 변환. scale0/1은 선형 t 재배치.
    // clamp_to_edge = grid_sample padding_mode='border' 등가.
    kernel void rifeWarp(
        texture2d<float, access::sample> imgA [[texture(0)]],
        texture2d<float, access::sample> imgB [[texture(1)]],
        texture2d<float, access::sample> flowTex [[texture(2)]],
        texture2d<float, access::sample> maskTex [[texture(3)]],
        texture2d<float, access::write> outTex [[texture(4)]],
        constant WarpParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = outTex.get_width();
        uint h = outTex.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 sizeF = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / sizeF;
        float4 f = flowTex.sample(s, uv);
        float m = 1.0 / (1.0 + exp(-maskTex.sample(s, uv).r));
        float2 f0 = f.xy * p.flowScale * p.scale0;
        float2 f1 = f.zw * p.flowScale * p.scale1;
        float3 a = imgA.sample(s, uv + f0 / sizeF).rgb;
        float3 b = imgB.sample(s, uv + f1 / sizeF).rgb;
        outTex.write(float4(a * m + b * (1.0 - m), 1.0), gid);
    }
    """
}
