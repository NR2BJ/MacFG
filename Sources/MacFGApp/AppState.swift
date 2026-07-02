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
    // 기본 엔진 = Metal Flow: 24/30/60fps 전 매트릭스에서 우위 실측
    // (144Hz 기준 — 24fps: 144fps/σ0.8 vs AppleFI 48fps/σ9; 지터 강건성 동급 이상)
    var selectedRenderMode: RenderMode = .metalFlow
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
    private var trackingTimer: Timer?

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
        // 출력 창이 다른 화면으로 이동 → 그 화면의 vsync로 페이싱 재바인딩
        self.overlayManager?.onOutputScreenChanged = { [weak self] screen in
            guard let self, self.isCapturing, self.displaySync != nil else { return }
            DiagnosticLog.shared.log("[DISPLAY] output moved to \(screen?.localizedName ?? "?") → DisplayLink 재바인딩")
            self.restartDisplayLink()
        }
        // 디스플레이 구성 변경(주사율 전환/모니터 연결 해제) 시 DisplayLink 재시작 —
        // 기존 링크는 이전 모드에 묶여 페이싱이 깨진다 (사용자가 120↔144Hz를 오가는 환경)
        NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.handleScreenParametersChange()
            }
        }
    }

    private func handleScreenParametersChange() {
        guard isCapturing, displaySync != nil else { return }
        DiagnosticLog.shared.log("[DISPLAY] screen parameters changed → DisplayLink 재시작")
        restartDisplayLink()
    }

    private func restartDisplayLink() {
        displaySync?.stop()
        let sync = DisplayLinkSync(device: device)
        displaySync = sync
        sync.start(screen: overlayManager?.outputScreen) { [weak self] timestamp, targetTimestamp in
            MainActor.assumeIsolated {
                self?.onDisplayLinkTick(timestamp: timestamp, targetTimestamp: targetTimestamp)
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
        if let cIdx = args.firstIndex(of: "--corner-radius"), cIdx + 1 < args.count,
           let r = Double(args[cIdx + 1]), r >= 0, r <= 64 {
            OverlayStyleConstants.cornerRadius = r
            DiagnosticLog.shared.log("[AUTO] cornerRadius=\(r)")
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
        guard !isCapturing else { return }
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

            // 엔진 준비를 먼저 끝낸 뒤 렌더 루프 시작 (출력 화면의 vsync에 바인딩)
            await configurePairEngine()
            restartDisplayLink()

            statsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.updateStats()
                }
            }

            // 창 추적은 30Hz면 충분 — 틱(120Hz)마다 CGWindowList를 부르면 호출당 0.5-2ms로
            // vsync 틱을 놓쳐 출력 fps 천장이 ~110으로 내려앉는다 (실측)
            trackingTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.overlayManager?.updateTracking()
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
        trackingTimer?.invalidate()
        trackingTimer = nil

        isCapturing = false
        captureMethod = "None"
        trackingMethod = "None"
        resetScheduler()
        logger.info("Capture stopped")
        DiagnosticLog.shared.log("Capture stopped")
    }

    /// 캡처 스트림만 재시작 (오버레이/DisplayLink/엔진 유지) — 창 리사이즈 대응
    private func restartCaptureStream(reason: String) async {
        guard let windowID = selectedWindowID, isCapturing, !isRestartingCapture else { return }
        isRestartingCapture = true
        defer { isRestartingCapture = false }
        DiagnosticLog.shared.log("[CAPTURE] stream restart: \(reason)")
        await captureManager.stopCapture()
        do {
            try await captureManager.startCapture(windowID: windowID, device: device)
            resetScheduler()
            pairEngine?.reset()
        } catch {
            DiagnosticLog.shared.log("[CAPTURE] stream restart FAILED: \(error) → 캡처 종료")
            await stopCapture()
        }
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
        resetSnapState()
        lastVsyncTarget = 0
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
    /// 케이던스 스냅 상태 — 캡처 ts는 디스플레이 vsync 그리드(144Hz 등)에 양자화되고
    /// 실콘텐츠(브라우저)는 PTS가 [5~99ms]로 튄다. 중앙값 간격 + 앵커 그리드로 락을 유지하고
    /// 이탈 3연속일 때만 재동기 — 즉발 재동기는 그리드 정렬 위상을 흔들어 σ를 키운다 (실측).
    private var snapTsRing: [CFTimeInterval] = []
    private var snapMissStreak = 0
    private var snappedLastTimestamp: CFTimeInterval = 0
    private var diagResyncCount = 0
    private var lastPresentedTimestamp: CFTimeInterval = 0
    private var lastPresentedTexture: (any MTLTexture)?
    private var sourceIntervalEMA: Double = 0
    private var hasReceivedFirstFrame: Bool = false
    /// 캡처 창 리사이즈 감지 (연속 감지 횟수 — 드래그 중 재시작 연발 방지)
    private var resizeMismatchCount = 0
    private var isRestartingCapture = false
    private var presentedTimes: [CFTimeInterval] = []
    private var latencySamplesMs: [Double] = []
    /// 최근 vsync 목표 시각 — 보간 위상을 디스플레이 그리드에 정렬하기 위한 기준
    private var lastVsyncTarget: CFTimeInterval = 0

    /// 출력 지연: 콘텐츠 시각을 이만큼 과거로 조준한다.
    /// 보간 프레임 I(A,B)가 B 도착 + GPU/ANE 완료 후 표시 슬롯에 준비되어 있으려면
    /// 소스 간격의 ~1.25배 + 워크 마진이 필요. (60fps 소스 기준 ~25ms)
    private var latencyOffset: Double {
        let interval = sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0
        // + 반 슬롯: SCK 배달이 소스 vsync에 양자화되어 최대 반 간격 늦게 오는데,
        // 그 마진이 없으면 늦은 쌍의 보간 프레임이 표시 시한을 놓쳐 stale-drop
        // (보간 188장 생성 → 113장 표시 실측 — present 110/s의 주범)
        let displayHalfSlot = 0.5 / max(displaySync?.refreshRate ?? 120, 60)
        return min(max(interval, 1.0 / 120.0), 1.0 / 24.0) * 1.25 + 0.004 + displayHalfSlot
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
        lastVsyncTarget = targetTimestamp

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

        // 캡처 창 리사이즈 감지 — SCK 스트림은 시작 시 해상도 고정이라 리사이즈하면
        // 늘어나거나 흐려진다. 새 크기가 ~1초 유지되면 스트림만 재시작 (오버레이/엔진 유지).
        if diagTick % 60 == 0, isCapturing, !isRestartingCapture, stablePoolWidth > 0,
           let src = overlayManager?.sourcePixelSize {
            let mismatch = abs(src.width - stablePoolWidth) > 8 || abs(src.height - stablePoolHeight) > 8
            if mismatch {
                resizeMismatchCount += 1
                if resizeMismatchCount >= 2 {
                    resizeMismatchCount = 0
                    Task { await restartCaptureStream(reason: "resize → \(src.width)x\(src.height)") }
                }
            } else {
                resizeMismatchCount = 0
            }
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
        // stale 한계: 평시 2.5 간격(늦은 묶음도 순서대로 표시 — 구멍보다 +1vsync 지연이 낫다),
        // 큐가 깊어지면(≥5) 1.2로 조여 백로그를 서서히 배출 — 생성≈소비 균형에서 큐가
        // 고여 e2e가 +40ms 눌러앉는 것 방지 (실측 75-84ms → 목표 ~55ms)
        let staleCutoff = target - interval * (timeline.count >= 5 ? 1.2 : 2.5)
        let candidates = timeline.filter { $0.timestamp > lastPresentedTimestamp && $0.timestamp <= target + 0.002 }
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
            resetSnapState()
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
            let displayInterval = 1.0 / max(refreshRate, 30)
            // 콘텐츠가 주사율에 근접하면 보간 무의미 (갭에 표시 슬롯이 없음)
            let contentAlreadyFast = gap < displayInterval * 0.75
            if gap > 0 && gap < 0.25 && !contentAlreadyFast
                && prev.texture.width == stable.width && prev.texture.height == stable.height
                && previousAcceptedTs == prev.rawTimestamp {
                // 보간 위상을 vsync 그리드 시각에 정렬 — 균등분할(t=k/(n+1))은
                // 60fps→144Hz(쌍당 2.4슬롯)처럼 비정수 조합에서 쌍마다 2/3개를 오가며
                // 시간축이 출렁였다 (Metal Flow가 AppleFI보다 덜 부드럽던 원인).
                // 그리드 시각에 놓인 프레임은 pick 시점과 정확히 일치 → 완전 균일 모션.
                var tValues: [Float] = []
                if lastVsyncTarget > 0 {
                    let gridRef = lastVsyncTarget - latencyOffset
                    let kStart = ((prev.timestamp - gridRef) / displayInterval + 1e-6).rounded(.up)
                    var slotTime = gridRef + kStart * displayInterval
                    // 양쪽 양보 구간: A 직후 0.4슬롯 + B 직전 0.6슬롯은 원본이 차지 —
                    // 그리드 위상에 따라 쌍당 2장이 끼며 과생성(1.4장/쌍 → 큐 적체 e2e+30ms 실측)
                    // 되는 것을 차단. 60fps@120Hz에서 유효 창이 정확히 1슬롯 = 쌍당 1장 보장.
                    while slotTime < snappedTs - displayInterval * 0.6 && tValues.count < 8 {
                        let t = (slotTime - prev.timestamp) / gap
                        if slotTime > prev.timestamp + displayInterval * 0.4 && t > 0.02 {
                            tValues.append(Float(t))
                        }
                        slotTime += displayInterval
                    }
                }
                // 폴백은 큰 갭 + 불운한 그리드 위상일 때만. 작은 갭(≤1.5슬롯)은 소스 두 장이
                // 이미 인접 슬롯을 채우므로 [0.5] 폴백이 잉여 프레임 → 큐 적체(e2e +40ms 실측)
                if tValues.isEmpty && gap > displayInterval * 1.5 { tValues = [0.5] }
                if tValues.isEmpty {
                    diagSkipContentFast += 1
                }
                interpResult = tValues.isEmpty ? nil : pairEngine?.encodePair(
                    stableA: prev.texture, stableB: stable,
                    tsA: prev.timestamp, tsB: snappedTs,
                    tValues: tValues,
                    into: cb
                )
                pairStartTs = prev.timestamp
                pairGap = gap
                if let interpResult {
                    diagInterpEncodedCount += interpResult.frames.count
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

    /// 캡처 타임스탬프를 콘텐츠 케이던스에 스냅 (단일 메커니즘: 케이던스 연속).
    ///
    /// snapped_n = snapped_{n-1} + round(rawDelta/interval)·interval — 위상 기준점 없이
    /// 직전 스냅에서 정수 스텝 전진. 간격은 링 시간폭/프레임수(양자화 무편향).
    /// 앵커 위상 방식과 병행하면 서로 다른 위상으로 스냅이 섞여 갭이 8/25ms로 요동
    /// (실측) — 단일 메커니즘이 자기일관적이라 갭이 균일해진다.
    private func snapTimestamp(raw: CFTimeInterval, rawDelta: Double) -> CFTimeInterval {
        if rawDelta > 0.5 || rawDelta <= 0 {
            snapTsRing = []
        }
        snapTsRing.append(raw)
        if snapTsRing.count > 16 { snapTsRing.removeFirst() }
        guard snapTsRing.count >= 5, snappedLastTimestamp > 0 else {
            snapMissStreak = 0
            snappedLastTimestamp = raw
            return raw
        }
        let interval = (snapTsRing.last! - snapTsRing.first!) / Double(snapTsRing.count - 1)
        guard interval > 0.002 else {
            snappedLastTimestamp = max(raw, snappedLastTimestamp + 0.001)
            return snappedLastTimestamp
        }
        sourceIntervalEMA = interval

        let steps = max(1.0, (rawDelta / interval).rounded())
        var predicted = snappedLastTimestamp + steps * interval
        let err = raw - predicted

        if abs(err) <= interval * 0.6 {
            snapMissStreak = 0
            predicted += err * 0.08 // 실클럭 드리프트 추적 (천천히)
            let snapped = max(predicted, snappedLastTimestamp + 0.001)
            snappedLastTimestamp = snapped
            return snapped
        }

        // 이탈 — 이 프레임은 raw로 통과, 3연속이면 재동기
        snapMissStreak += 1
        if snapMissStreak >= 3 || rawDelta > 0.5 {
            snapMissStreak = 0
            diagResyncCount += 1
        }
        let snapped = max(raw, snappedLastTimestamp + 0.001)
        snappedLastTimestamp = snapped
        return snapped
    }

    private func resetSnapState() {
        snapTsRing = []
        snapMissStreak = 0
        snappedLastTimestamp = 0
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
        // 출력 FPS는 실제 glass 시각(presented handler의 presentedTime)으로 계산 —
        // PerformanceMonitor의 renderTimestamps는 mailbox 드레인 시각이라 틱에 뭉쳐
        // 표시값이 80-120으로 맥놀이 (실프레임은 꾸준한데 지표만 출렁, 실측)
        if presentedTimes.count >= 2,
           let first = presentedTimes.first, let last = presentedTimes.last, last > first {
            outputFPS = Double(presentedTimes.count - 1) / (last - first)
        } else {
            outputFPS = 0
        }
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
