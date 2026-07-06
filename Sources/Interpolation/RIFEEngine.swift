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
/// 멀티 t (앵커 방식): predict 예산(쌍 간격×0.8 ÷ predictEMA)이 허용하는 수만큼 **앵커 t를
/// exact predict**하고, 각 요청 t는 최근접 앵커 flow를 선형 스케일해 워프한다.
/// - 전부 감당되면 t별 전부 exact (arbitrary-t 풀활용).
/// - 일부만 되면 연속 청크 중앙값 앵커 — 스케일 편차를 최소화 (기존 "0.5 하나서 ±60% 스케일"이
///   극단 t 화질을 MetalFlow 이하로 떨어뜨리던 것(1080p 합성 t=0.25 23.8dB vs exact 26.8dB 실측)
///   을 ±20~33% 스케일로 축소).
/// - 예산 부족(EMA 미확립 포함)이면 앵커 1개(t=0.5) = 기존 근사와 동일.
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

    /// 0.25: 빠른 게임(오버워치 시점 회전/이펙트)이 히스토그램을 크게 흔들어 0.5에선 초당 수 회
    /// 오검출 → 보간 폐기 → 25~50ms 구멍(실측 cut=1~9/2s). 진짜 하드컷은 교집합이 ~0이라 안전.
    private let sceneCutIntersectionThreshold = 0.25

    private var device: (any MTLDevice)?
    private var model: MLModel?
    private var modelW = 0
    private var modelH = 0
    private var packQueue: (any MTLCommandQueue)?
    private var packPSO: (any MTLComputePipelineState)?
    private var unpackPSO: (any MTLComputePipelineState)?
    private var warpPSO: (any MTLComputePipelineState)?

    /// 앵커별 flow/mask 세트 최대 수 (예산이 넉넉하면 t별 전부 exact)
    static let maxAnchors = 4

    /// predict 파이프라인 슬롯 — 3개 링 (pack/predict/warp 중첩 + 배압)
    private final class Slot: @unchecked Sendable {
        let packBuf: any MTLBuffer          // 6ch fp16 CHW (모델 입력 — 앵커 공유)
        let flowBufs: [any MTLBuffer]       // 앵커별 4ch fp16 CHW (모델 출력 backing)
        let maskBufs: [any MTLBuffer]       // 앵커별 1ch fp16 (pre-sigmoid)
        let statsBuf: any MTLBuffer         // uint32 x64 루마 히스토그램 (A32+B32)
        let confBuf: any MTLBuffer          // uint32 x3 — 워프 conf 실측 (sum255/count/flowSum8)
        let flowTexs: [any MTLTexture]      // 앵커별 rgba16Float 모델 크기
        let maskTexs: [any MTLTexture]      // 앵커별 r16Float 모델 크기
        let event: any MTLSharedEvent
        var useCount: UInt64 = 0
        var busy = false
        init(packBuf: any MTLBuffer, flowBufs: [any MTLBuffer], maskBufs: [any MTLBuffer],
             statsBuf: any MTLBuffer, confBuf: any MTLBuffer,
             flowTexs: [any MTLTexture], maskTexs: [any MTLTexture],
             event: any MTLSharedEvent) {
            self.packBuf = packBuf; self.flowBufs = flowBufs; self.maskBufs = maskBufs
            self.statsBuf = statsBuf; self.confBuf = confBuf
            self.flowTexs = flowTexs; self.maskTexs = maskTexs
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
    /// 최근 predict 시간 링(≤15) — 앵커 예산은 **중앙값**으로 판단. EMA(α=0.05)는 콜드스타트
    /// 60ms급 첫 predict에 오염돼 실제값(12ms) 복귀까지 수십 쌍 → 앵커가 영영 미발동했음(실측).
    private let predictMsRing = OSAllocatedUnfairLock(initialState: [Double]())
    private func predictMsMedian() -> Double {
        predictMsRing.withLock { ring in
            guard !ring.isEmpty else { return 0 }
            let s = ring.sorted()
            return s[s.count / 2]
        }
    }

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

    /// 모델 로드 (단변/유닛 지정) — prepare와 사다리 스위치가 공유
    private static func loadModel(short: Int, gpu: Bool) async throws -> (MLModel, Int, Int) {
        guard let src = modelSourceURL(short: short) else { throw InterpolationError.notPrepared }
        let compiled = try await compiledModelURL(source: src, short: short)
        let config = MLModelConfiguration()
        config.computeUnits = gpu ? .cpuAndGPU : .cpuAndNeuralEngine
        let model = try await MLModel.load(contentsOf: compiled, configuration: config)
        guard let xDesc = model.modelDescription.inputDescriptionsByName["x"],
              let shape = xDesc.multiArrayConstraint?.shape, shape.count == 4 else {
            throw InterpolationError.notPrepared
        }
        return (model, shape[3].intValue, shape[2].intValue)
    }

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        let short = Self.flowShortSide
        let (model, w, h) = try await Self.loadModel(short: short, gpu: Self.useGPU)
        self.model = model
        modelW = w
        modelH = h
        currentShort = short
        currentGPU = Self.useGPU

        packQueue = device.makeCommandQueue()
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ name: String) async throws -> any MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw InterpolationError.shaderCompilationFailed }
            return try await device.makeComputePipelineState(function: fn)
        }
        packPSO = try await pso("rifePack")
        unpackPSO = try await pso("rifeUnpack")
        warpPSO = try await pso("rifeWarp")

        slots = try Self.buildSlots(device: device, modelW: modelW, modelH: modelH)
        cancelled.withLock { $0 = false }
        DiagnosticLog.shared.log("[RIFE] prepared: \(modelW)x\(modelH) flow (\(short)p), unit=\(Self.useGPU ? "GPU" : "ANE")")
    }

    /// 슬롯 3개 × 앵커별 flow/mask 세트 (360 기준 슬롯당 ~11MB, 총 ~34MB)
    private static func buildSlots(device: any MTLDevice, modelW: Int, modelH: Int) throws -> [Slot] {
        var slots: [Slot] = []
        let planeBytes = modelW * modelH * 2
        for _ in 0..<3 {
            guard let packBuf = device.makeBuffer(length: planeBytes * 6, options: .storageModeShared),
                  let statsBuf = device.makeBuffer(length: 64 * 4, options: .storageModeShared),
                  let confBuf = device.makeBuffer(length: 3 * 4, options: .storageModeShared),
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
            var flowBufs: [any MTLBuffer] = []
            var maskBufs: [any MTLBuffer] = []
            var flowTexs: [any MTLTexture] = []
            var maskTexs: [any MTLTexture] = []
            for _ in 0..<Self.maxAnchors {
                guard let fb = device.makeBuffer(length: planeBytes * 4, options: .storageModeShared),
                      let mb = device.makeBuffer(length: planeBytes, options: .storageModeShared),
                      let ft = device.makeTexture(descriptor: flowDesc),
                      let mt = device.makeTexture(descriptor: maskDesc) else {
                    throw InterpolationError.textureAllocationFailed
                }
                flowBufs.append(fb); maskBufs.append(mb); flowTexs.append(ft); maskTexs.append(mt)
            }
            slots.append(Slot(packBuf: packBuf, flowBufs: flowBufs, maskBufs: maskBufs,
                              statsBuf: statsBuf, confBuf: confBuf,
                              flowTexs: flowTexs, maskTexs: maskTexs, event: event))
        }
        return slots
    }

    // MARK: - 적응 사다리 (GPU 과부하 → 288+ANE로 predict 이전, 여유 복귀 시 원복)

    /// 현재 로드된 모델 단변/유닛
    private var currentShort = 0
    private var currentGPU = true
    /// 전환 중 (encodePair는 nil 반환 — 소스만 표시, ~1s)
    private let switching = OSAllocatedUnfairLock(initialState: false)
    private var lastSwitchAt: CFTimeInterval = 0
    private var ladderPairs = 0
    private var ladderExhausts = 0
    private var gapMsEMA: Double = 0

    /// 과부하/여유 판정 → 필요 시 모델·유닛 핫스왑 킥. encodePair(렌더 스레드)에서 호출.
    /// 판정: predict 중앙값이 쌍 간격의 90%↑(지속 불가) 또는 슬롯 고갈 5%↑ → (288, ANE)로
    /// — predict를 ANE로 옮겨 GPU를 MetalFX 업스케일/표시에 돌려준다 (1080p60+4K뷰어에서
    /// GPU 150%+ 포화 → tick 108Hz 붕괴 실측). 중앙값이 간격의 35%↓(예: 24fps 소스)면 원복.
    private func maybeAdapt(gapS: Double, exhausted: Bool) {
        ladderPairs += 1
        if exhausted { ladderExhausts += 1 }
        gapMsEMA = gapMsEMA == 0 ? gapS * 1000 : gapMsEMA * 0.9 + gapS * 1000 * 0.1
        guard ladderPairs >= 180 else { return }   // ~3s @60fps 윈도
        let exhaustRate = Double(ladderExhausts) / Double(ladderPairs)
        ladderPairs = 0
        ladderExhausts = 0
        let med = predictMsMedian()
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastSwitchAt > 10, med > 0, gapMsEMA > 1 else { return }

        let overloaded = med > gapMsEMA * 0.9 || exhaustRate > 0.05
        let underloaded = med < gapMsEMA * 0.35
        if overloaded, currentGPU, Self.modelAvailable(short: 288) {
            kickSwitch(short: 288, gpu: false,
                       reason: "과부하 med=\(String(format: "%.1f", med))ms/gap=\(String(format: "%.1f", gapMsEMA))ms exhaust=\(String(format: "%.0f", exhaustRate * 100))%")
        } else if underloaded, !currentGPU {
            kickSwitch(short: Self.flowShortSide, gpu: true,
                       reason: "여유 복귀 med=\(String(format: "%.1f", med))ms/gap=\(String(format: "%.1f", gapMsEMA))ms")
        }
    }

    private func kickSwitch(short: Int, gpu: Bool, reason: String) {
        guard switching.withLock({ if $0 { return false }; $0 = true; return true }) else { return }
        lastSwitchAt = CFAbsoluteTimeGetCurrent()
        DiagnosticLog.shared.log("[RIFE] ladder → \(short)p/\(gpu ? "GPU" : "ANE") (\(reason))")
        Task { [weak self] in
            guard let self, let device = self.device else { return }
            defer { self.switching.withLock { $0 = false } }
            do {
                let (newModel, w, h) = try await Self.loadModel(short: short, gpu: gpu)
                // 인플라이트 predict/warp 완료 대기 (슬롯 전부 반납까지, 최대 2s)
                for _ in 0..<40 {
                    let allFree = self.slotLock.withLock { self.slots.allSatisfy { !$0.busy } }
                    if allFree { break }
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
                let newSlots = try Self.buildSlots(device: device, modelW: w, modelH: h)
                self.slotLock.withLock {
                    self.model = newModel
                    self.modelW = w
                    self.modelH = h
                    self.slots = newSlots
                    self.currentShort = short
                    self.currentGPU = gpu
                }
                self.predictMsRing.withLock { $0 = [] }   // 새 속도 — 예산 재학습
                DiagnosticLog.shared.log("[RIFE] ladder 전환 완료: \(w)x\(h) (\(short)p/\(gpu ? "GPU" : "ANE"))")
            } catch {
                DiagnosticLog.shared.log("[RIFE] ladder 전환 실패: \(error)")
            }
        }
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
        guard let packQueue, let packPSO, let unpackPSO, let warpPSO,
              !tValues.isEmpty, tsB > tsA else { return nil }
        if switching.withLock({ $0 }) { return nil }   // 사다리 전환 중 — 소스만 표시 (~1s)

        // 슬롯+모델 원자 스냅샷 — 사다리 스왑(모델·크기·슬롯 동시 교체)과 일관성 보장
        let acquired: (Slot, MLModel, Int, Int)? = slotLock.withLock {
            guard let m = model else { return nil }
            if let s = slots.first(where: { !$0.busy }) { s.busy = true; s.useCount += 1; return (s, m, modelW, modelH) }
            return nil
        }
        maybeAdapt(gapS: tsB - tsA, exhausted: acquired == nil)
        guard let (slot, model, mW0, mH0) = acquired else { return nil }
        let signalValue = slot.useCount

        ensureOutputPool(width: stableB.width, height: stableB.height)
        let ts = tValues.prefix(6).map { min(max($0, 0.01), 0.99) }
        guard outputPool.count >= ts.count else { releaseSlot(slot); return nil }

        // ── 앵커 선택: predict 예산(쌍 간격×0.8 ÷ 중앙값) 내에서 최대한 exact
        let med = predictMsMedian()
        let budgetMs = (tsB - tsA) * 1000.0 * 0.8
        let affordable = med > 0 ? max(1, Int(budgetMs / med)) : 1
        let anchorCount = min(ts.count, affordable, Self.maxAnchors)
        var anchors: [Float]
        if anchorCount >= ts.count {
            anchors = Array(ts)                             // 전부 exact (arbitrary-t 풀활용)
        } else if anchorCount <= 1 {
            anchors = [0.5]                                 // 예산 부족 — 기존 근사
        } else {
            // 연속 청크 중앙값 — 각 t의 최근접 앵커 스케일 편차 최소화
            anchors = (0..<anchorCount).map { j in
                let lo = j * ts.count / anchorCount
                let hi = (j + 1) * ts.count / anchorCount - 1
                return (ts[lo] + ts[hi]) / 2
            }
        }
        anchors = anchors.map { min(max($0, 0.05), 0.95) }

        // ── ① pack: 자체 cb (flow/mask 0클리어 → 실패 시 50/50 블렌드 강등 보장)
        guard let packCB = packQueue.makeCommandBuffer() else { releaseSlot(slot); return nil }
        if let fill = packCB.makeBlitCommandEncoder() {
            for i in 0..<anchors.count {
                fill.fill(buffer: slot.flowBufs[i], range: 0..<slot.flowBufs[i].length, value: 0)
                fill.fill(buffer: slot.maskBufs[i], range: 0..<slot.maskBufs[i].length, value: 0)
            }
            fill.fill(buffer: slot.statsBuf, range: 0..<(64 * 4), value: 0)
            fill.fill(buffer: slot.confBuf, range: 0..<(3 * 4), value: 0)
            fill.endEncoding()
        }
        guard let packEnc = packCB.makeComputeCommandEncoder() else { releaseSlot(slot); return nil }
        packEnc.setComputePipelineState(packPSO)
        packEnc.setTexture(stableA, index: 0)
        packEnc.setTexture(stableB, index: 1)
        packEnc.setBuffer(slot.packBuf, offset: 0, index: 0)
        packEnc.setBuffer(slot.statsBuf, offset: 0, index: 1)
        var packParams = SIMD2<UInt32>(UInt32(mW0), UInt32(mH0))
        packEnc.setBytes(&packParams, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        dispatch(packEnc, width: mW0, height: mH0, pso: packPSO)
        packEnc.endEncoding()

        // ── ② pack 완료 → 워커에서 predict → 어떤 경로든 event signal
        let mW = mW0, mH = mH0
        let workerRef = worker
        let cancelledRef = cancelled
        let ringRef = predictMsRing
        let loggerRef = logger
        let anchorsRef = anchors
        packCB.addCompletedHandler { [weak model] _ in
            workerRef.async {
                defer { slot.event.signaledValue = signalValue }
                guard let model, !cancelledRef.withLock({ $0 }) else { return }
                for (i, anchorT) in anchorsRef.enumerated() {
                    if cancelledRef.withLock({ $0 }) { return }
                    do {
                        let t0 = CFAbsoluteTimeGetCurrent()
                        let xArr = try MLMultiArray(
                            dataPointer: slot.packBuf.contents(),
                            shape: [1, 6, NSNumber(value: mH), NSNumber(value: mW)],
                            dataType: .float16,
                            strides: [NSNumber(value: 6 * mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                        let tArr = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
                        tArr[0] = NSNumber(value: anchorT)
                        let flowBack = try MLMultiArray(
                            dataPointer: slot.flowBufs[i].contents(),
                            shape: [1, 4, NSNumber(value: mH), NSNumber(value: mW)],
                            dataType: .float16,
                            strides: [NSNumber(value: 4 * mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                        let maskBack = try MLMultiArray(
                            dataPointer: slot.maskBufs[i].contents(),
                            shape: [1, 1, NSNumber(value: mH), NSNumber(value: mW)],
                            dataType: .float16,
                            strides: [NSNumber(value: mH * mW), NSNumber(value: mH * mW), NSNumber(value: mW), 1])
                        let opts = MLPredictionOptions()
                        opts.outputBackings = ["flow": flowBack, "mask": maskBack]
                        let input = try MLDictionaryFeatureProvider(dictionary: ["x": xArr, "t": tArr])
                        let result = try model.prediction(from: input, options: opts)
                        // backing 미사용 폴백 — 반환 배열이 우리 버퍼가 아니면 복사
                        if let f = result.featureValue(for: "flow")?.multiArrayValue,
                           f.dataPointer != slot.flowBufs[i].contents() {
                            f.withUnsafeBytes { raw in
                                slot.flowBufs[i].contents().copyMemory(
                                    from: raw.baseAddress!, byteCount: min(raw.count, slot.flowBufs[i].length))
                            }
                        }
                        if let m = result.featureValue(for: "mask")?.multiArrayValue,
                           m.dataPointer != slot.maskBufs[i].contents() {
                            m.withUnsafeBytes { raw in
                                slot.maskBufs[i].contents().copyMemory(
                                    from: raw.baseAddress!, byteCount: min(raw.count, slot.maskBufs[i].length))
                            }
                        }
                        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                        ringRef.withLock { ring in
                            ring.append(ms)
                            if ring.count > 15 { ring.removeFirst() }
                        }
                    } catch {
                        loggerRef.error("predict failed (anchor \(anchorT)): \(error.localizedDescription)")
                    }
                }
            }
        }
        packCB.commit()

        // ── ③ 호출자 cb: 이벤트 대기 → 앵커별 unpack → t별 최근접 앵커 워프
        commandBuffer.encodeWaitForEvent(slot.event, value: signalValue)
        for i in 0..<anchors.count {
            guard let upEnc = commandBuffer.makeComputeCommandEncoder() else { releaseSlot(slot); return nil }
            upEnc.setComputePipelineState(unpackPSO)
            upEnc.setBuffer(slot.flowBufs[i], offset: 0, index: 0)
            upEnc.setBuffer(slot.maskBufs[i], offset: 0, index: 1)
            upEnc.setTexture(slot.flowTexs[i], index: 0)
            upEnc.setTexture(slot.maskTexs[i], index: 1)
            dispatch(upEnc, width: mW0, height: mH0, pso: unpackPSO)
            upEnc.endEncoding()
        }

        var frames: [(t: Float, texture: any MTLTexture)] = []
        for t in ts {
            // 최근접 앵커 (동률이면 아무쪽 — 스케일 편차 동일)
            var ai = 0
            var best = Float.greatestFiniteMagnitude
            for (j, a) in anchors.enumerated() where abs(t - a) < best { best = abs(t - a); ai = j }
            let anchor = anchors[ai]
            let output = outputPool[outputIndex]
            outputIndex = (outputIndex + 1) % outputPool.count
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { break }
            enc.setComputePipelineState(warpPSO)
            enc.setTexture(stableA, index: 0)
            enc.setTexture(stableB, index: 1)
            enc.setTexture(slot.flowTexs[ai], index: 2)
            enc.setTexture(slot.maskTexs[ai], index: 3)
            enc.setTexture(output, index: 4)
            var wp = WarpParams(
                flowScale: SIMD2<Float>(Float(output.width) / Float(mW0), Float(output.height) / Float(mH0)),
                scale0: t / anchor,
                scale1: (1 - t) / (1 - anchor),
                tPhase: t)
            enc.setBytes(&wp, length: MemoryLayout<WarpParams>.size, index: 0)
            enc.setBuffer(slot.confBuf, offset: 0, index: 1)
            dispatch(enc, width: output.width, height: output.height, pso: warpPSO)
            enc.endEncoding()
            frames.append((t, output))
        }

        // 슬롯 반납은 호출자 cb 완료 시 (unpack/warp가 버퍼·텍스처를 다 읽은 뒤)
        let dumpPair = pairCount
        commandBuffer.addCompletedHandler { [weak self] _ in
            // 디버그: 패리티 진단용 버퍼 덤프 (MACFG_RIFE_DUMP=<dir>, 4번째 쌍 = 벤치 최종)
            if dumpPair == 3, let dir = ProcessInfo.processInfo.environment["MACFG_RIFE_DUMP"] {
                for (nm, buf) in [("pack", slot.packBuf), ("flow", slot.flowBufs[0]), ("mask", slot.maskBufs[0])] {
                    let data = Data(bytes: buf.contents(), count: buf.length)
                    try? data.write(to: URL(fileURLWithPath: "\(dir)/rife_\(nm).bin"))
                }
            }
            self?.releaseSlot(slot)
        }

        if pairCount % 120 == 0 {
            let confBufRef = slot.confBuf
            let pc = pairCount
            commandBuffer.addCompletedHandler { _ in
                let p = confBufRef.contents().bindMemory(to: UInt32.self, capacity: 3)
                let cnt = max(1.0, Double(p[1]))
                let confAvg = Double(p[0]) / 255.0 / cnt
                let flowAvg = Double(p[2]) / 8.0 / cnt
                DiagnosticLog.shared.log("[RIFE-CONF] pair#\(pc) confAvg=\(String(format: "%.2f", confAvg)) |flow|avg=\(String(format: "%.1f", flowAvg))px (샘플 \(Int(cnt)))")
            }
        }
        pairCount += 1
        if pairCount <= 3 || pairCount % 600 == 0 || anchors.count > 1 {
            let medNow = predictMsMedian()
            DiagnosticLog.shared.log("[RIFE] pair #\(pairCount) t×\(frames.count) anchors=\(anchors.count) predictMed=\(String(format: "%.1f", medNow))ms")
            if ProcessInfo.processInfo.environment["MACFG_RIFE_VERBOSE"] != nil {
                print("  [RIFE] pair#\(pairCount) anchors=\(anchors.map { String(format: "%.2f", $0) }.joined(separator: ",")) med=\(String(format: "%.1f", medNow))")
            }
        }

        // 장면 전환: pack 히스토그램 교집합 (pack cb는 이 시점 완료가 보장됨 — 호출자 cb 완료 후 평가)
        let threshold = sceneCutIntersectionThreshold
        let statsBuf = slot.statsBuf
        let evaluator: @Sendable () -> Bool = {
            let ptr = statsBuf.contents().bindMemory(to: UInt32.self, capacity: 64)
            var hA = [Double](repeating: 0, count: 32)
            var hB = [Double](repeating: 0, count: 32)
            var totalA = 0.0, totalB = 0.0, sumA = 0.0, sumB = 0.0
            for i in 0..<32 {
                hA[i] = Double(ptr[i]); hB[i] = Double(ptr[32 + i])
                totalA += hA[i]; totalB += hB[i]
                sumA += hA[i] * Double(i); sumB += hB[i] * Double(i)
            }
            guard totalA > 1000, totalB > 0 else { return false }
            // 밝기 정렬 교집합 — 평균 bin 차이만큼 B를 시프트해 비교. 플래시/이펙트(균일 밝기
            // 변화)는 정렬돼 통과하고, 진짜 컷(구조 변화)만 낮게 남는다 (게임 오검출 차단).
            let shift = Int((sumB / totalB - sumA / totalA).rounded())
            var intersect = 0.0
            for i in 0..<32 {
                let j = i + shift
                if j >= 0, j < 32 { intersect += min(hA[i], hB[j]) }
            }
            return intersect / totalA < threshold
        }
        return frames.isEmpty ? nil : PairEncodeResult(frames: frames, sceneCutEvaluator: evaluator)
    }

    private struct WarpParams {
        var flowScale: SIMD2<Float>
        var scale0: Float
        var scale1: Float
        var tPhase: Float        // 표시 위상 t — flow 불신 시 원본 A/B 블렌드 가중
        var pad: Float = 0
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

    struct WarpParams { float2 flowScale; float scale0; float scale1; float tPhase; float pad; };
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
        device atomic_uint* confStats [[buffer(1)]],
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
        const float3 kL = float3(0.299, 0.587, 0.114);
        float2 uvA = uv + f0 / sizeF;
        float2 uvB = uv + f1 / sizeF;
        float3 a = imgA.sample(s, uvA).rgb;
        float3 b = imgB.sample(s, uvB).rgb;
        float3 warped = a * m + b * (1.0 - m);
        float3 a0 = imgA.sample(s, uv).rgb;
        float3 b0 = imgB.sample(s, uv).rgb;
        // 소스 불일치(±0.75px 4점 미니블러 — 코덱 노이즈 내성)
        float2 po = 0.75 / sizeF;
        float dBlur = 0.0;
        dBlur += dot(abs(imgA.sample(s, uv + float2( po.x,  po.y)).rgb - imgB.sample(s, uv + float2( po.x,  po.y)).rgb), kL);
        dBlur += dot(abs(imgA.sample(s, uv + float2(-po.x,  po.y)).rgb - imgB.sample(s, uv + float2(-po.x,  po.y)).rgb), kL);
        dBlur += dot(abs(imgA.sample(s, uv + float2( po.x, -po.y)).rgb - imgB.sample(s, uv + float2( po.x, -po.y)).rgb), kL);
        dBlur += dot(abs(imgA.sample(s, uv + float2(-po.x, -po.y)).rgb - imgB.sample(s, uv + float2(-po.x, -po.y)).rgb), kL);
        dBlur *= 0.25;
        // 워프 후 불일치도 ±1.5px 미니블러 — coarse flow(288→풀해상도 6.7×)의 1~3px 양자화
        //    어긋남은 정상인데, 단일점 비교는 텍스처에서 이를 오류로 오판해 conf를 깎아
        //    절반 크로스페이드(부드러움 소실, 실측)를 만든다. 블러 비교는 소소한 어긋남을
        //    용서하고 진짜 오정합(블러 반경 초과)만 벌점.
        float2 pw = 1.5 / sizeF;
        float errW = 0.0;
        errW += dot(abs(imgA.sample(s, uvA + float2( pw.x,  pw.y)).rgb - imgB.sample(s, uvB + float2( pw.x,  pw.y)).rgb), kL);
        errW += dot(abs(imgA.sample(s, uvA + float2(-pw.x,  pw.y)).rgb - imgB.sample(s, uvB + float2(-pw.x,  pw.y)).rgb), kL);
        errW += dot(abs(imgA.sample(s, uvA + float2( pw.x, -pw.y)).rgb - imgB.sample(s, uvB + float2( pw.x, -pw.y)).rgb), kL);
        errW += dot(abs(imgA.sample(s, uvA + float2(-pw.x, -pw.y)).rgb - imgB.sample(s, uvB + float2(-pw.x, -pw.y)).rgb), kL);
        errW *= 0.25;
        // 일관성 비율 판정 — flow 정확: errW << dBlur → 워프 신뢰(보간 부드러움).
        // flow 오류(오위치 유령): errW ≥ dBlur → 제자리 크로스페이드 강등.
        // 정적 UI(조준점/글자): dBlur≈0 + errW(배경 워프)큼 → 비율↑ → 원본 크로스페이드 = 고정.
        // (별도 정적 마스크는 제거 — 느린 팬까지 얼려 60fps 계단을 만들었음, 실측 '부드러움 없음'.)
        // 모션 비례 관용 — 큰 모션(8~25px, 실측 게임)일수록 coarse flow의 수 px 오차는 불가피
        // + 크로스페이드는 명백한 이중상이라 워프가 낫다. 관용 없인 conf~0.6 물타기(실측
        // '보간 못 느낌'). 정적 UI는 ratio 5~10이라 관용 2.5배에도 철벽.
        float fmag = (length(f0) + length(f1)) * 0.5;
        float tol = 1.0 + min(fmag * 0.06, 1.5);
        float ratio = errW / ((dBlur + 0.01) * tol);
        float conf = 1.0 - smoothstep(1.0, 1.8, ratio);
        float3 srcBlend = mix(a0, b0, p.tPhase);
        float3 outc = mix(srcBlend, warped, conf);
        // 하드 정적 마스크 (MetalFlow와 동일 문턱 0.008~0.04) — 소스가 안 변한 픽셀(dBlur≈0)은
        // flow가 *일관되게* 끌어도(비율 판정은 errW 작아 안 걸림) 무조건 소스 고정. RIFE엔 이게
        // 없어 정적 UI(채팅/HUD)가 flow에 끌려 다중 고스트로 찢겼다(실프레임 diff 확인). MetalFlow가
        // 게임에서 UI 안정적인 핵심. 문턱이 슬로우팬(|A-B|>0.008)은 안 얼려 60fps 스텝 회피.
        float staticW = 1.0 - smoothstep(0.008, 0.04, dBlur);
        outc = mix(outc, srcBlend, staticW);
        // conf/flow 실측 (16px 격자 스파스 — '보간이 실제로 얼마나 걸리는가' 계측)
        if ((gid.x & 15u) == 0u && (gid.y & 15u) == 0u) {
            atomic_fetch_add_explicit(&confStats[0], uint(conf * 255.0), memory_order_relaxed);
            atomic_fetch_add_explicit(&confStats[1], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&confStats[2], uint(clamp(fmag, 0.0, 500.0) * 8.0), memory_order_relaxed);
        }
        outTex.write(float4(outc, 1.0), gid);
    }
    """
}
