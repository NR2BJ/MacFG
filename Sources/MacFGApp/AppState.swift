import SwiftUI
@preconcurrency import Metal
import QuartzCore
import CaptureKit
import Overlay
import FramePacing
import Interpolation
import Monitoring
import os

/// 보간 엔진 선택. appleFI(ANE 720p+마스크 합성) vs metalFlow(LSFG 방식 순수 GPU) 비교 가능.
/// blend는 미지원 폴백/디버그용 (UI 미노출, `--auto-mode blend`).
enum RenderMode: String, CaseIterable, Identifiable {
    case appleFI
    case metalFlow
    case blend

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .appleFI: "Apple FI"
        case .metalFlow: "Metal Flow"
        case .blend: "Blend 2x"
        }
    }

    /// UI에 노출할 엔진들
    static let userSelectable: [RenderMode] = [.appleFI, .metalFlow]
}

/// 출력 타임라인 항목 — 표시할 텍스처와 콘텐츠 시각
private struct TimelineEntry {
    let timestamp: CFTimeInterval
    let texture: any MTLTexture
    let isInterpolated: Bool
    /// 레이턴시 측정용 원본 캡처 시각 (보간 프레임은 B의 캡처 시각)
    let captureTimestamp: CFTimeInterval
}

/// GPU 완료/presented 핸들러(백그라운드 스레드) → 렌더 틱(MainActor) 전달함
private final class RenderMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [TimelineEntry] = []
    private var releasedTextures: [ObjectIdentifier] = []
    private var presentedRecords: [(presentedAt: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool)] = []

    func postCompleted(entries newEntries: [TimelineEntry], released: ObjectIdentifier, workLatencyMs: Double, sceneCut: Bool) {
        lock.lock()
        entries.append(contentsOf: newEntries)
        releasedTextures.append(released)
        workLatencies.append(workLatencyMs)
        if workLatencies.count > 480 { workLatencies.removeFirst(240) }
        if sceneCut { sceneCutCount += 1 }
        lock.unlock()
    }

    func drainWorkLatencies() -> [Double] {
        lock.lock()
        defer { workLatencies = []; lock.unlock() }
        return workLatencies
    }

    func drainSceneCutCount() -> Int {
        lock.lock()
        defer { sceneCutCount = 0; lock.unlock() }
        return sceneCutCount
    }

    private var workLatencies: [Double] = []
    private var sceneCutCount = 0

    func postPresented(at time: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool) {
        lock.lock()
        presentedRecords.append((time, captureTs, isInterp))
        if presentedRecords.count > 480 { presentedRecords.removeFirst(240) }
        lock.unlock()
    }

    func drain() -> ([TimelineEntry], [ObjectIdentifier], [(presentedAt: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool)]) {
        lock.lock()
        defer {
            entries = []
            releasedTextures = []
            presentedRecords = []
            lock.unlock()
        }
        return (entries, releasedTextures, presentedRecords)
    }
}

/// 앱 전역 상태 관리
///
/// 렌더 루프 설계 (타임스탬프 스케줄러):
/// - 캡처 스레드가 큐에 쌓은 프레임을 매 틱 drain
/// - 새 프레임마다: 안정 텍스처로 blit + 직전 프레임과 보간 인코딩 (work queue, 비동기 완료)
/// - 완료된 프레임은 (콘텐츠 시각, 텍스처) 타임라인에 등재
/// - 매 디스플레이 틱: targetTimestamp - latencyOffset 시각에 해당하는 항목을 골라
///   present(at: targetTimestamp) 로 vsync 정렬 표시
/// - 새로 표시할 것이 없으면 present 스킵 (컴포지터가 직전 프레임 유지)
/// 출력이 시간의 함수가 되므로 캡처 지터/컨텐츠 fps 변화에 자가 보정된다.
@MainActor
@Observable
public final class AppState {
    // MARK: - State
    var isCapturing = false
    var captureMethod: String = "None"
    var trackingMethod: String = "None"
    var inputFPS: Double = 0
    var outputFPS: Double = 0
    var latencyMs: Double = 0
    var selectedWindowID: CGWindowID?
    var selectedWindowName: String = ""
    var availableWindows: [WindowInfo] = []
    var isInterpolationEnabled: Bool = true
    var interpolationEngine: String = "None"
    var selectedRenderMode: RenderMode = .appleFI
    var selectedOverlayPlacement: OverlayPlacement = .coverSource

    // MARK: - Components
    let device: any MTLDevice
    /// 보간/복사 작업용 큐 — present 큐와 분리해 4-5ms 보간 작업이 present를 막지 않게 한다
    private var workQueue: (any MTLCommandQueue)?
    /// present 전용 큐 (틱당 ~0.3ms 렌더패스만)
    private var presentQueue: (any MTLCommandQueue)?
    private let captureManager = CaptureManager()
    private var overlayManager: OverlayManager?
    private let performanceMonitor = PerformanceMonitor()
    private var displaySync: DisplayLinkSync?
    private var pairEngine: (any PairInterpolationEngine)?
    private let mailbox = RenderMailbox()
    private let logger = Logger(subsystem: "com.macfg", category: "AppState")

    // MARK: - Stats Timer
    private var statsTimer: Timer?

    public init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        self.workQueue = device.makeCommandQueue()
        self.presentQueue = device.makeCommandQueue()
        self.overlayManager = OverlayManager(device: device)
        self.interpolationEngine = selectedRenderMode.displayName
        // 뷰어 창 X 버튼 → 캡처 정지
        self.overlayManager?.onViewerClosed = { [weak self] in
            Task { @MainActor in
                guard let self, self.isCapturing else { return }
                await self.stopCapture()
            }
        }
    }

    // MARK: - Window Discovery

    func refreshWindowList() {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let infoList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
            return
        }

        availableWindows = infoList.compactMap { info in
            guard let windowID = info[kCGWindowNumber as String] as? CGWindowID,
                  let ownerName = info[kCGWindowOwnerName as String] as? String,
                  let layer = info[kCGWindowLayer as String] as? Int,
                  layer == 0,
                  let bounds = info[kCGWindowBounds as String] as? [String: CGFloat],
                  let width = bounds["Width"], let height = bounds["Height"],
                  width > 50, height > 50
            else { return nil }

            let name = info[kCGWindowName as String] as? String ?? ""
            let displayName = name.isEmpty ? ownerName : "\(ownerName) — \(name)"

            return WindowInfo(
                windowID: windowID,
                ownerName: ownerName,
                windowName: name,
                displayName: displayName,
                width: Int(width),
                height: Int(height)
            )
        }.filter { $0.ownerName != "MacFGApp" }
    }

    // MARK: - Auto Start (CLI 자체 테스트용)

    /// `--auto-capture-title <substr> [--auto-mode <mode>] [--auto-placement <cover|beside>]`
    func processAutoStartArguments() async {
        let args = ProcessInfo.processInfo.arguments
        guard let idx = args.firstIndex(of: "--auto-capture-title"), idx + 1 < args.count else { return }
        let titleSub = args[idx + 1].lowercased()

        if let mIdx = args.firstIndex(of: "--auto-mode"), mIdx + 1 < args.count,
           let mode = RenderMode(rawValue: args[mIdx + 1]) {
            selectedRenderMode = mode
        }
        if let pIdx = args.firstIndex(of: "--auto-placement"), pIdx + 1 < args.count {
            let v = args[pIdx + 1]
            selectedOverlayPlacement = (v == "viewer" || v == "beside") ? .viewerWindow : .coverSource
        }
        if let fIdx = args.firstIndex(of: "--flow-base"), fIdx + 1 < args.count,
           let base = Double(args[fIdx + 1]), base >= 120, base <= 2048 {
            MetalFlowEngine.flowBaseLongSide = base
            DiagnosticLog.shared.log("[AUTO] flowBaseLongSide=\(Int(base))")
        }

        // 창 목록에 대상이 뜰 때까지 재시도 (최대 15초)
        for _ in 0..<30 {
            refreshWindowList()
            if let target = availableWindows.first(where: { $0.displayName.lowercased().contains(titleSub) }) {
                selectedWindowID = target.windowID
                selectedWindowName = target.displayName
                DiagnosticLog.shared.log("[AUTO] capturing '\(target.displayName)' mode=\(selectedRenderMode.rawValue) placement=\(selectedOverlayPlacement.rawValue)")
                await startCapture()
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        DiagnosticLog.shared.log("[AUTO] target window not found: \(titleSub)")
    }

    // MARK: - Capture Control

    func startCapture() async {
        guard let windowID = selectedWindowID else {
            logger.warning("No window selected")
            return
        }

        do {
            try await captureManager.startCapture(windowID: windowID, device: device)
            captureMethod = captureManager.activeMethod.rawValue

            overlayManager?.setPlacement(selectedOverlayPlacement)
            try overlayManager?.start(windowID: windowID)
            trackingMethod = overlayManager?.trackingMethod ?? "Unknown"

            // 엔진 준비를 먼저 끝낸 뒤 렌더 루프 시작
            await configurePairEngine()

            let sync = DisplayLinkSync(device: device)
            self.displaySync = sync
            sync.start { [weak self] timestamp, targetTimestamp in
                MainActor.assumeIsolated {
                    self?.onDisplayLinkTick(timestamp: timestamp, targetTimestamp: targetTimestamp)
                }
            }

            statsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.updateStats()
                }
            }

            isCapturing = true
            logger.info("Capture started: \(self.captureMethod) + \(self.trackingMethod)")
            DiagnosticLog.shared.log("Capture started: \(captureMethod) + \(trackingMethod) mode=\(selectedRenderMode.rawValue)")
        } catch {
            logger.error("Failed to start capture: \(error)")
            return
        }
    }

    func stopCapture() async {
        displaySync?.stop()
        displaySync = nil

        await captureManager.stopCapture()
        overlayManager?.stop()

        pairEngine?.shutdown()
        pairEngine = nil
        interpolationEngine = "None"

        statsTimer?.invalidate()
        statsTimer = nil

        isCapturing = false
        captureMethod = "None"
        trackingMethod = "None"
        resetScheduler()
        logger.info("Capture stopped")
        DiagnosticLog.shared.log("Capture stopped")
    }

    private func resetScheduler() {
        timeline = []
        inFlightTextures = []
        stablePool = []
        stablePoolWidth = 0
        stablePoolHeight = 0
        prevStable = nil
        lastPresentedTimestamp = 0
        lastPresentedTexture = nil
        lastAcceptedTimestamp = 0
        lastAcceptedFingerprint = 0
        snappedLastTimestamp = 0
        sourceIntervalEMA = 0
        hasReceivedFirstFrame = false
        presentedTimes = []
        _ = mailbox.drain()
    }

    // MARK: - Scheduler State

    private var timeline: [TimelineEntry] = []
    private var stablePool: [any MTLTexture] = []
    private var stablePoolWidth = 0
    private var stablePoolHeight = 0
    private var inFlightTextures: Set<ObjectIdentifier> = []
    /// timestamp = 스냅된 콘텐츠 시각 (타임라인/보간용), rawTimestamp = SCK 원본 시각 (연속성 검사용)
    private var prevStable: (texture: any MTLTexture, timestamp: CFTimeInterval, rawTimestamp: CFTimeInterval)?
    private var lastAcceptedTimestamp: CFTimeInterval = 0
    private var lastAcceptedFingerprint: UInt64 = 0
    /// 케이던스 스냅 상태 — 캡처 ts는 디스플레이 vsync 그리드(144Hz 등)에 양자화되어
    /// 60fps 콘텐츠가 13.9/20.8ms 교대 간격으로 도착한다. 위상고정 그리드에 스냅해
    /// 보간 midpoint가 실제 콘텐츠 타임라인의 중점에 놓이게 한다.
    private var snappedLastTimestamp: CFTimeInterval = 0
    private var diagResyncCount = 0
    private var lastPresentedTimestamp: CFTimeInterval = 0
    private var lastPresentedTexture: (any MTLTexture)?
    private var sourceIntervalEMA: Double = 0
    private var hasReceivedFirstFrame: Bool = false
    private var presentedTimes: [CFTimeInterval] = []
    private var latencySamplesMs: [Double] = []

    /// 출력 지연: 콘텐츠 시각을 이만큼 과거로 조준한다.
    /// 보간 프레임 I(A,B)가 B 도착 + GPU/ANE 완료 후 표시 슬롯에 준비되어 있으려면
    /// 소스 간격의 ~1.25배 + 워크 마진이 필요. (60fps 소스 기준 ~25ms)
    private var latencyOffset: Double {
        let interval = sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0
        return min(max(interval, 1.0 / 120.0), 1.0 / 24.0) * 1.25 + 0.004
    }

    // ── 진단 ──
    private var diagTick: Int = 0
    private var diagSourceCount = 0
    private var diagDupSkipCount = 0
    private var diagTsRejectCount = 0        // 타임스탬프 비전진으로 스킵 (중복 프레임 재전송)
    private var diagPresentCount = 0
    private var diagInterpPresentCount = 0
    private var diagPoolExhaustCount = 0
    private var diagInterpEncodedCount = 0
    private var diagFrameTypes: [String] = []
    private var diagSrcIntMin: Double = .infinity  // 콘텐츠 간격 min/max (VFR 판별)
    private var diagSrcIntMax: Double = 0
    private var diagDrainDepthSum: Int = 0         // 매 틱 drain한 프레임 수 (버스트 판별)
    private var diagDrainDepthMax: Int = 0
    private var diagDrainSamples: Int = 0
    // 보간 스킵 사유별 카운터 (interpEnc=0 재발 시 원인 특정)
    private var diagSkipToggleOff = 0
    private var diagSkipEngineNil = 0
    private var diagSkipNoPrev = 0
    private var diagSkipContentFast = 0
    private var diagSkipBigGap = 0
    private var diagSkipDiscontinuity = 0
    private var diagSkipEngineFail = 0
    private var diagSkipOther = 0

    // MARK: - Render Loop

    private func onDisplayLinkTick(timestamp: CFTimeInterval, targetTimestamp: CFTimeInterval) {
        diagTick += 1
        overlayManager?.updateTracking()

        // 1) 완료된 GPU 작업 수거 → 타임라인 등재
        let (newEntries, released, presented) = mailbox.drain()
        for id in released { inFlightTextures.remove(id) }
        if !newEntries.isEmpty {
            timeline.append(contentsOf: newEntries)
            timeline.sort { $0.timestamp < $1.timestamp }
        }
        for record in presented {
            performanceMonitor.recordRenderTime()
            presentedTimes.append(record.presentedAt)
            if presentedTimes.count > 240 { presentedTimes.removeFirst(120) }
            let latency = (record.presentedAt - record.captureTs) * 1000.0
            if latency > 0 && latency < 500 {
                latencySamplesMs.append(latency)
                if latencySamplesMs.count > 240 { latencySamplesMs.removeFirst(120) }
            }
        }

        // 2) 캡처 프레임 drain → work 인코딩
        let slots = captureManager.drainFrames()
        var sawTexture = false
        var depth = 0
        for slot in slots where slot.texture != nil {
            sawTexture = true
            depth += 1
            ingest(slot)
        }
        if !slots.isEmpty {
            diagDrainDepthSum += depth
            diagDrainSamples += 1
            if depth > diagDrainDepthMax { diagDrainDepthMax = depth }
        }

        // 대상 창 종료 감지 — SCK는 마지막 슬롯을 계속 반환하므로 창 추적 실패로 판단
        if sawTexture {
            hasReceivedFirstFrame = true
        }
        if isCapturing && hasReceivedFirstFrame && (overlayManager?.trackingFailureCount ?? 0) > 240 {
            logger.info("Target window closed, stopping")
            Task { await stopCapture() }
            return
        }

        // 3) 타임라인 정리
        timeline.removeAll { $0.timestamp <= lastPresentedTimestamp }
        if timeline.count > 12 {
            timeline.removeFirst(timeline.count - 12)
        }

        // 4) 표시할 항목 선택 — targetTimestamp에서 latencyOffset만큼 과거의 콘텐츠.
        // 미표시 항목 중 가장 오래된 것부터 순서대로 (늦게 도착한 보간 프레임도 순서 보존).
        // 최신-우선으로 고르면 워크 완료가 한 틱만 늦어도 보간 프레임이 영구 드랍된다.
        let target = targetTimestamp - latencyOffset
        let interval = min(max(sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0, 1.0 / 120.0), 1.0 / 24.0)
        let staleCutoff = target - interval * 0.9
        let candidates = timeline.filter { $0.timestamp > lastPresentedTimestamp && $0.timestamp <= target + 0.001 }
        var pick = candidates.first
        for next in candidates.dropFirst() {
            // 이미 고른 항목이 심하게 과거(백로그)면 다음 항목으로 건너뛰어 따라잡는다
            if let current = pick, current.timestamp < staleCutoff {
                pick = next
            } else {
                break
            }
        }
        if let pick {
            presentEntry(pick, at: targetTimestamp)
        }

        maybeLogDiagnostics()
    }

    /// 새 캡처 프레임 수용: 중복 제거 → 안정 복사 + 보간 인코딩 (비동기 GPU)
    private func ingest(_ slot: FrameSlot) {
        guard let sourceTexture = slot.texture else { return }

        // 타임스탬프 역행/중복 제거
        if slot.timestamp <= lastAcceptedTimestamp + 0.0005 { diagTsRejectCount += 1; return }
        // 수용 정책: 픽셀 변화(fingerprint) 우선. SCK status는 fingerprint가 없을 때만 폴백.
        // (게이트를 1/120으로 연 뒤 SCK가 60fps 창에도 status=complete를 ~112fps로 남발하는 것을
        //  실측 — status를 믿으면 간격 EMA가 반토막나 "이미 빠른 콘텐츠" 가드가 보간을 꺼버림)
        let fingerprintChanged = slot.contentFingerprint != 0 && slot.contentFingerprint != lastAcceptedFingerprint
        let accept = fingerprintChanged || (slot.contentFingerprint == 0 && slot.contentChanged)
        if !accept {
            diagDupSkipCount += 1
            return
        }

        overlayManager?.setCaptureColorSpace(slot.colorSpace)

        let delta = lastAcceptedTimestamp > 0 ? slot.timestamp - lastAcceptedTimestamp : 0
        if delta > 0 && delta < 0.5 {
            sourceIntervalEMA = sourceIntervalEMA == 0 ? delta : sourceIntervalEMA * 0.9 + delta * 0.1
            if delta < diagSrcIntMin { diagSrcIntMin = delta }
            if delta > diagSrcIntMax { diagSrcIntMax = delta }
        }
        let previousAcceptedTs = lastAcceptedTimestamp
        lastAcceptedTimestamp = slot.timestamp
        lastAcceptedFingerprint = slot.contentFingerprint
        performanceMonitor.recordFrameArrival()
        diagSourceCount += 1

        // 타임스탬프를 콘텐츠 케이던스 그리드에 스냅 (양자화 지터 제거)
        let snappedTs = snapTimestamp(raw: slot.timestamp, rawDelta: delta)

        guard let workQueue,
              let stable = acquireStableTexture(width: sourceTexture.width, height: sourceTexture.height),
              let cb = workQueue.makeCommandBuffer() else {
            diagPoolExhaustCount += 1
            // 이 프레임은 수용 실패 — 상태를 되돌려 다음 프레임이 정상 쌍(연속성 유지)을 만들게 함
            lastAcceptedTimestamp = previousAcceptedTs
            snappedLastTimestamp = 0
            return
        }

        // 안정 복사 (SCK IOSurface 재활용에서 분리)
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(
                from: sourceTexture, sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: sourceTexture.width, height: sourceTexture.height, depth: 1),
                to: stable, destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
        }

        // 보간: 직전 소스와의 쌍. 갭이 크면(일시정지 후 재개) 스킵하고 연속성 리셋.
        // 갭 적응 다중 t: 소스 프레임이 드랍되어 갭이 디스플레이 슬롯 여러 개를 덮으면
        // (예: 60fps에서 한 장 빠짐 → 33ms 갭 @120Hz = 슬롯 4개) 그만큼 위상을 나눠 채운다.
        // 이게 없으면 드랍 지점마다 16.7ms+ 표시 구멍 = "평균 fps는 높은데 1% low가 낮은" 체감.
        var interpResult: PairEncodeResult?
        var pairStartTs: CFTimeInterval = 0
        var pairGap: CFTimeInterval = 0
        let refreshRate = displaySync?.refreshRate ?? 120.0
        // 보간 스킵 사유 진단 (재현 시 원인 즉시 특정용)
        if !isInterpolationEnabled { diagSkipToggleOff += 1 }
        else if pairEngine == nil { diagSkipEngineNil += 1 }
        else if prevStable == nil { diagSkipNoPrev += 1 }
        let wantInterpolation = isInterpolationEnabled && pairEngine != nil
        if wantInterpolation, let prev = prevStable {
            let gap = snappedTs - prev.timestamp
            let contentAlreadyFast = sourceIntervalEMA > 0 && sourceIntervalEMA < 1.3 / refreshRate
            if gap > 0 && gap < 0.25 && !contentAlreadyFast
                && prev.texture.width == stable.width && prev.texture.height == stable.height
                && previousAcceptedTs == prev.rawTimestamp {
                let displayInterval = 1.0 / max(refreshRate, 30)
                let slots = max(1, Int((gap / displayInterval).rounded()))
                let n = min(max(slots - 1, 1), 4)
                let tValues = (1...n).map { Float($0) / Float(n + 1) }
                interpResult = pairEngine?.encodePair(
                    stableA: prev.texture, stableB: stable,
                    tsA: prev.timestamp, tsB: snappedTs,
                    tValues: tValues,
                    into: cb
                )
                pairStartTs = prev.timestamp
                pairGap = gap
                if interpResult != nil {
                    diagInterpEncodedCount += tValues.count
                } else {
                    diagSkipEngineFail += 1
                }
            } else if contentAlreadyFast {
                diagSkipContentFast += 1
            } else if gap >= 0.25 {
                diagSkipBigGap += 1
                pairEngine?.reset()
            } else if previousAcceptedTs != prev.rawTimestamp {
                diagSkipDiscontinuity += 1
            } else {
                diagSkipOther += 1
            }
        }

        prevStable = (stable, snappedTs, slot.timestamp)
        inFlightTextures.insert(ObjectIdentifier(stable))

        let entryTs = snappedTs
        let mailboxRef = mailbox
        let stableRef: any MTLTexture = stable
        let interpFrames = interpResult?.frames ?? []
        let cutEvaluator = interpResult?.sceneCutEvaluator
        let startTs = pairStartTs
        let gapRef = pairGap
        cb.addCompletedHandler { _ in
            // 장면 전환이면 보간 프레임 폐기 — 무관한 두 샷 사이의 모핑 프레임 방지
            let isSceneCut = cutEvaluator?() ?? false
            var entries: [TimelineEntry] = []
            if !isSceneCut {
                for frame in interpFrames {
                    let ts = startTs + gapRef * Double(frame.t)
                    entries.append(TimelineEntry(timestamp: ts, texture: frame.texture, isInterpolated: true, captureTimestamp: entryTs))
                }
            }
            entries.append(TimelineEntry(timestamp: entryTs, texture: stableRef, isInterpolated: false, captureTimestamp: entryTs))
            // 캡처 시각 → 타임라인 등재까지의 파이프라인 지연 (스케줄러 offset 튜닝 지표)
            let workLatency = (CACurrentMediaTime() - entryTs) * 1000.0
            mailboxRef.postCompleted(entries: entries, released: ObjectIdentifier(stableRef), workLatencyMs: workLatency, sceneCut: isSceneCut)
        }
        cb.commit()
    }

    private func presentEntry(_ entry: TimelineEntry, at targetTimestamp: CFTimeInterval) {
        guard let presentQueue, let cb = presentQueue.makeCommandBuffer() else { return }
        guard let drawable = overlayManager?.encodeRenderFrame(texture: entry.texture, into: cb) else {
            cb.commit()
            return
        }

        lastPresentedTimestamp = entry.timestamp
        lastPresentedTexture = entry.texture
        diagPresentCount += 1
        if entry.isInterpolated { diagInterpPresentCount += 1 }
        diagFrameTypes.append(entry.isInterpolated ? "I" : "S")
        if diagFrameTypes.count > 60 { diagFrameTypes.removeFirst(30) }

        let mailboxRef = mailbox
        let captureTs = entry.captureTimestamp
        let isInterp = entry.isInterpolated
        drawable.addPresentedHandler { d in
            mailboxRef.postPresented(at: d.presentedTime, captureTs: captureTs, isInterp: isInterp)
        }
        cb.present(drawable, atTime: targetTimestamp)
        cb.commit()
    }

    /// 캡처 타임스탬프를 콘텐츠 케이던스 그리드에 위상고정(PLL) 스냅.
    ///
    /// 소스 창은 자기 디스플레이 vsync에서만 프레임을 올릴 수 있어, 144Hz 모니터의
    /// 60fps 영상은 13.9/20.8ms 교대 간격으로 캡처된다 (144/60=2.4, 비정수).
    /// 이 지터를 그대로 쓰면 보간 midpoint가 실제 콘텐츠 중점에서 ±3.5ms 어긋나
    /// 미세 저더가 생긴다. EMA 주기 그리드에 스냅하되 잔차의 10%만 반영해
    /// 실제 클럭에서 벗어나지 않게 위상 추적한다.
    private func snapTimestamp(raw: CFTimeInterval, rawDelta: Double) -> CFTimeInterval {
        let interval = sourceIntervalEMA
        guard interval > 0.002, snappedLastTimestamp > 0 else {
            snappedLastTimestamp = raw
            return raw
        }
        // 케이던스 급변(더 빠른 콘텐츠 전환/일시정지 재개) → 재동기
        if rawDelta < interval * 0.3 || rawDelta > 0.5 {
            snappedLastTimestamp = raw
            diagResyncCount += 1
            return raw
        }
        let steps = max(1.0, (rawDelta / interval).rounded())
        var snapped = snappedLastTimestamp + steps * interval
        // 위상 오차가 크면 그리드가 어긋난 것 → 재동기.
        // 허용창 0.5는 비정수 주사율/콘텐츠 비율의 vsync 양자화(±displayInterval/2)
        // + SCK 전달 지터를 흡수 — 0.35는 144Hz@60fps에서 재동기 폭풍(2초당 30회) 유발 실측.
        if abs(raw - snapped) > interval * 0.5 {
            snappedLastTimestamp = raw
            diagResyncCount += 1
            return raw
        }
        snapped += (raw - snapped) * 0.1
        snapped = max(snapped, snappedLastTimestamp + 0.001) // 단조 증가 (VT 세션 요구)
        snappedLastTimestamp = snapped
        return snapped
    }

    /// 사용 중이지 않은 풀 텍스처 획득 (타임라인/직전 소스/마지막 표시/인플라이트 제외)
    private func acquireStableTexture(width: Int, height: Int) -> (any MTLTexture)? {
        if width != stablePoolWidth || height != stablePoolHeight {
            stablePool = []
            stablePoolWidth = width
            stablePoolHeight = height
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead]
            desc.storageMode = .private
            for _ in 0..<6 {
                if let tex = device.makeTexture(descriptor: desc) {
                    stablePool.append(tex)
                }
            }
            // 크기가 바뀌면 이전 참조는 모두 무효
            timeline = []
            prevStable = nil
            lastPresentedTexture = nil
            lastPresentedTimestamp = 0
            pairEngine?.reset()
        }

        var busy = Set<ObjectIdentifier>()
        for entry in timeline { busy.insert(ObjectIdentifier(entry.texture)) }
        if let prev = prevStable { busy.insert(ObjectIdentifier(prev.texture)) }
        if let last = lastPresentedTexture { busy.insert(ObjectIdentifier(last)) }
        busy.formUnion(inFlightTextures)

        return stablePool.first { !busy.contains(ObjectIdentifier($0)) }
    }

    // MARK: - Diagnostics

    private func maybeLogDiagnostics() {
        guard diagTick % 240 == 0 else { return }  // 진단 중 2초 주기 (@120Hz)

        let workLats = mailbox.drainWorkLatencies()
        let avgWork = workLats.isEmpty ? 0 : workLats.reduce(0, +) / Double(workLats.count)
        let maxWork = workLats.max() ?? 0
        let cuts = mailbox.drainSceneCutCount()

        // presented 간격 통계 (실제 glass 시각 기반 — 스무스니스의 ground truth)
        var intervals: [Double] = []
        if presentedTimes.count >= 2 {
            for i in 1..<presentedTimes.count {
                let d = (presentedTimes[i] - presentedTimes[i - 1]) * 1000.0
                if d > 0 && d < 100 { intervals.append(d) }
            }
        }
        let avgInterval = intervals.isEmpty ? 0 : intervals.reduce(0, +) / Double(intervals.count)
        let variance = intervals.isEmpty ? 0 : intervals.map { ($0 - avgInterval) * ($0 - avgInterval) }.reduce(0, +) / Double(intervals.count)
        let maxInterval = intervals.max() ?? 0

        let avgLatency = latencySamplesMs.isEmpty ? 0 : latencySamplesMs.reduce(0, +) / Double(latencySamplesMs.count)
        let pattern = diagFrameTypes.suffix(24).joined()
        let srcFps = sourceIntervalEMA > 0 ? 1.0 / sourceIntervalEMA : 0
        let uniquePresented = diagPresentCount - diagInterpPresentCount  // 표시된 고유 콘텐츠 수
        let srcIntLo = diagSrcIntMin.isFinite ? diagSrcIntMin * 1000 : 0
        let srcIntHi = diagSrcIntMax * 1000
        let drainAvg = diagDrainSamples > 0 ? Double(diagDrainDepthSum) / Double(diagDrainSamples) : 0

        var skipParts: [String] = []
        if diagSkipToggleOff > 0 { skipParts.append("off:\(diagSkipToggleOff)") }
        if diagSkipEngineNil > 0 { skipParts.append("noEng:\(diagSkipEngineNil)") }
        if diagSkipNoPrev > 0 { skipParts.append("noPrev:\(diagSkipNoPrev)") }
        if diagSkipContentFast > 0 { skipParts.append("fast:\(diagSkipContentFast)") }
        if diagSkipBigGap > 0 { skipParts.append("gap:\(diagSkipBigGap)") }
        if diagSkipDiscontinuity > 0 { skipParts.append("discont:\(diagSkipDiscontinuity)") }
        if diagSkipEngineFail > 0 { skipParts.append("engFail:\(diagSkipEngineFail)") }
        if diagSkipOther > 0 { skipParts.append("other:\(diagSkipOther)") }
        let skips = skipParts.isEmpty ? "-" : skipParts.joined(separator: ",")

        let msg = "[SCHED] src=\(diagSourceCount)(\(String(format: "%.0f", srcFps))fps) uniqOut=\(uniquePresented) dupSkip=\(diagDupSkipCount) tsRej=\(diagTsRejectCount) interpEnc=\(diagInterpEncodedCount) skip[\(skips)] present=\(diagPresentCount) (I=\(diagInterpPresentCount)) cut=\(cuts) resync=\(diagResyncCount) poolMiss=\(diagPoolExhaustCount) tl=\(timeline.count) | glass(ms): avg=\(String(format: "%.2f", avgInterval)) σ=\(String(format: "%.2f", sqrt(variance))) max=\(String(format: "%.1f", maxInterval)) | srcInt=\(String(format: "%.1f", sourceIntervalEMA * 1000))ms [\(String(format: "%.0f", srcIntLo))~\(String(format: "%.0f", srcIntHi))] | drain=\(String(format: "%.1f", drainAvg))/\(diagDrainDepthMax) | work=\(String(format: "%.0f", avgWork))/\(String(format: "%.0f", maxWork))ms e2e=\(String(format: "%.0f", avgLatency))ms | \(pattern)"
        DiagnosticLog.shared.log(msg)
        diagResyncCount = 0
        diagSkipToggleOff = 0; diagSkipEngineNil = 0; diagSkipNoPrev = 0
        diagSkipContentFast = 0; diagSkipBigGap = 0; diagSkipDiscontinuity = 0
        diagSkipEngineFail = 0; diagSkipOther = 0

        diagSourceCount = 0
        diagDupSkipCount = 0
        diagTsRejectCount = 0
        diagPresentCount = 0
        diagInterpPresentCount = 0
        diagPoolExhaustCount = 0
        diagInterpEncodedCount = 0
        diagSrcIntMin = .infinity
        diagSrcIntMax = 0
        diagDrainDepthSum = 0
        diagDrainDepthMax = 0
        diagDrainSamples = 0
    }

    private func updateStats() {
        inputFPS = performanceMonitor.inputFPS
        outputFPS = performanceMonitor.outputFPS
        latencyMs = latencySamplesMs.isEmpty ? 0 : latencySamplesMs.reduce(0, +) / Double(latencySamplesMs.count)
    }

    // MARK: - Interpolation Control

    func updateInterpolationEnabled() {
        if isCapturing {
            Task { @MainActor in
                await configurePairEngine()
            }
            return
        }
        interpolationEngine = isInterpolationEnabled ? selectedRenderMode.displayName : "Off"
    }

    func updateRenderMode() {
        if isCapturing {
            Task { @MainActor in
                await configurePairEngine()
            }
        } else {
            interpolationEngine = selectedRenderMode.displayName
        }
    }

    func updateOverlayPlacement() {
        overlayManager?.setPlacement(selectedOverlayPlacement)
    }

    private func configurePairEngine() async {
        pairEngine?.shutdown()
        pairEngine = nil

        guard isInterpolationEnabled else {
            interpolationEngine = "Off"
            return
        }

        let engine: any PairInterpolationEngine
        switch selectedRenderMode {
        case .appleFI:
            if AppleFIEngine.isSupported {
                engine = AppleFIEngine()
            } else {
                DiagnosticLog.shared.log("[ENGINE] AppleFI unsupported on this system → MetalFlow fallback")
                engine = MetalFlowEngine()
            }
        case .metalFlow:
            engine = MetalFlowEngine()
        case .blend:
            engine = LegacyPairEngine(BlendInterpolator())
        }

        do {
            try await engine.prepare(device: device)
            pairEngine = engine
            interpolationEngine = engine.name
            DiagnosticLog.shared.log("[ENGINE] ready: \(engine.name)")
        } catch {
            DiagnosticLog.shared.log("[ENGINE] \(engine.name) prepare FAILED: \(error) → Blend fallback")
            let fallback = LegacyPairEngine(BlendInterpolator())
            if (try? await fallback.prepare(device: device)) != nil {
                pairEngine = fallback
                interpolationEngine = fallback.name
            } else {
                pairEngine = nil
                interpolationEngine = "Failed"
            }
        }
    }
}

// MARK: - Window Info

public struct WindowInfo: Identifiable, Sendable {
    public let id: CGWindowID
    public let windowID: CGWindowID
    public let ownerName: String
    public let windowName: String
    public let displayName: String
    public let width: Int
    public let height: Int

    init(windowID: CGWindowID, ownerName: String, windowName: String, displayName: String, width: Int, height: Int) {
        self.id = windowID
        self.windowID = windowID
        self.ownerName = ownerName
        self.windowName = windowName
        self.displayName = displayName
        self.width = width
        self.height = height
    }
}
