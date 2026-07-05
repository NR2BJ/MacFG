import SwiftUI
import AppKit
import Carbon.HIToolbox
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
    case rife
    case blend

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .appleFI: "Apple FI"
        case .metalFlow: "Metal Flow"
        case .rife: "Neural"
        case .blend: "Blend 2x"
        }
    }

    /// UI에 노출할 엔진들 — Neural은 모델 파일이 있을 때만
    static var userSelectable: [RenderMode] {
        var modes: [RenderMode] = [.appleFI, .metalFlow]
        if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) {
            modes.append(.rife)
        }
        return modes
    }
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
    // 앱 설정 (엔진 무관) — 각 update*가 자체 영속. UI 언어는 관찰되어 변경 시 뷰 재구성→L() 재평가.
    var uiLanguage: String = UserDefaults.standard.string(forKey: "s.lang") ?? "system"
    var devLoggingEnabled: Bool = UserDefaults.standard.bool(forKey: "s.devlog")
    var menuBarOnly: Bool = UserDefaults.standard.bool(forKey: "s.menubaronly")
    // 기본 엔진 = Metal Flow: 24/30/60fps 전 매트릭스에서 우위 실측
    // (144Hz 기준 — 24fps: 144fps/σ0.8 vs AppleFI 48fps/σ9; 지터 강건성 동급 이상)
    var selectedRenderMode: RenderMode = .metalFlow
    var selectedOverlayPlacement: OverlayPlacement = .coverSource
    /// 업스케일 방식 — 뷰어에서 출력>소스일 때. off/ane/metalfx/aneMetalfx.
    /// 기본 Off: 흔한 케이스는 Cover+보간. 업스케일 선택 시 자동으로 Separate Window로 전환.
    var upscaleMode: UpscaleMode = .off
    /// CAS 샤프닝 on/off (업스케일과 독립, Cover 1:1 포함 어디서나)
    var casEnabled: Bool = true
    /// CAS 샤프닝 강도 0~1
    var sharpness: Double = 0.5
    /// 오클루전 방향별 워프 (실험, MetalFlow 전용) — 가림/드러남 경계에서 보이는 쪽 단방향 워프.
    /// 반복 패턴 aliasing 부작용 가능 → 실영상 A/B용. 기본 off.
    var occlusionDirectional: Bool = false
    /// 모션 부드러움 0(예리)~1(부드러움), 0.5=기본. MetalFlow 전용 취향 슬라이더 (실시간 반영).
    var motionSmoothness: Double = 0.5
    /// 경계 전환 0(crisp/저더)~1(soft/고스팅), 0.5=기본. 콘텐츠 취향(게임 crisp / 영화 soft).
    var boundarySoftness: Double = 0.5
    /// 업스케일 실동작 상태 (UI 표시용) — nil이면 미캡처/비활성
    var upscaleStatus: String?
    /// 보간 배율: 0=Auto(디스플레이 슬롯 전부 채움), 2~5=소스 fps × N 상한.
    /// 30fps 소스를 굳이 120까지 안 올리고 60(×2)에서 멈추고 싶을 때.
    var frameMultiplier: Int = 0
    /// 소스 리사이즈 프리셋(짧은 변 px, 0=끔). 캡처 전 미리 설정 → 캡처 시작 시 소스 창을 이 크기로.
    var sourcePreset: Int = 0

    // 사용자 지정 단축키 (init에서 UserDefaults 로드로 덮어씀)
    var hotCapture = HotKeyBinding(keyCode: UInt32(kVK_ANSI_U), modifiers: UInt32(controlKey | optionKey | cmdKey), label: "⌃⌥⌘U")
    /// 보간 on/off 전역 토글 — 전체화면 뷰어 안에서 동영상(보간 on)↔텍스트/인터랙티브(보간 off,
    /// 저지연)를 즉시 전환. 보간은 다음 프레임을 기다려 ~30ms 지연 + 텍스트 불연속 변화를 뭉갬.
    var hotInterp = HotKeyBinding(keyCode: UInt32(kVK_ANSI_I), modifiers: UInt32(controlKey | optionKey | cmdKey), label: "⌃⌥⌘I")

    // MARK: - Components
    let device: any MTLDevice
    /// 보간/복사 작업용 큐 — present 큐와 분리해 4-5ms 보간 작업이 present를 막지 않게 한다
    @ObservationIgnored nonisolated(unsafe) private var workQueue: (any MTLCommandQueue)?
    /// present 전용 큐 (틱당 ~0.3ms 렌더패스만)
    @ObservationIgnored nonisolated(unsafe) private var presentQueue: (any MTLCommandQueue)?
    private let captureManager = CaptureManager()
    /// 영역 캡처: 소스 창 좌상단 기준 크롭 사각형(pt). nil이면 창 전체.
    /// 설정되면 resize-reconfigure를 끈다(영역이 창-리사이즈 재구성으로 날아가지 않게).
    @ObservationIgnored nonisolated(unsafe) var captureRegion: CGRect?
    private var overlayManager: OverlayManager?
    private let performanceMonitor = PerformanceMonitor()
    @ObservationIgnored nonisolated(unsafe) private var pairEngine: (any PairInterpolationEngine)?
    private let mailbox = RenderMailbox()
    private let logger = Logger(subsystem: "com.macfg", category: "AppState")

    // MARK: - A2 렌더 스레드 (CAMetalDisplayLink)
    // 틱은 전용 렌더 스레드에서 실행 — nonisolated(unsafe) 상태들은 캡처 활성 중 렌더 스레드가
    // 소유하고, 메인은 정지 후(detach 동기 보장) 또는 명시된 락/미러를 통해서만 접근한다.
    private let renderDriver = RenderDriver()
    @ObservationIgnored nonisolated(unsafe) private var renderSurface: RenderSurface?
    /// UI 설정의 렌더용 미러 (메인이 쓰고 렌더가 읽는 racy-but-benign 단순값)
    @ObservationIgnored nonisolated(unsafe) private var mirrorInterpolationEnabled = true
    @ObservationIgnored nonisolated(unsafe) private var mirrorFrameMultiplier = 0
    @ObservationIgnored nonisolated(unsafe) private var mirrorRefreshRate: Double = 120
    /// 숨김→표시 전이 시 렌더 스레드가 자기 틱에서 스케줄러/엔진을 리셋하게 하는 신호
    @ObservationIgnored nonisolated(unsafe) private var pendingShowReset = false
    /// 캡처 색공간의 렌더→메인 전파 중복 방지
    @ObservationIgnored nonisolated(unsafe) private var lastSentColorSpace: CGColorSpace?
    /// presentedTimes/latencySamplesMs: 렌더가 쓰고 메인(updateStats)이 읽음 — 락 보호
    private let statsLock = NSLock()

    // MARK: - Stats Timer
    private var statsTimer: Timer?
    private var trackingTimer: Timer?
    private var drawableReattachTimer: Timer?

    // MARK: - Overlay Auto-Hide (단일 모니터: 소스 벗어나면 오버레이 양보)
    /// 캡처 대상 창을 소유한 앱의 PID (0 = 미확인 → 자동 숨김 비활성, 항상 표시)
    private var sourceOwnerPID: pid_t = 0
    /// 사용자가 단축키로 강제 숨김 (자동 숨김과 OR)
    private var overlayUserHidden = false
    /// 현재 오버레이가 숨김으로 적용된 상태인지 (전이 감지 + 렌더 정지 게이트)
    @ObservationIgnored nonisolated(unsafe) private var overlayHiddenState = false
    private var workspaceObserver: NSObjectProtocol?

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
            guard let self, self.isCapturing else { return }
            // CAMetalDisplayLink는 레이어의 디스플레이를 자동 추적 — 주사율 미러만 갱신
            self.mirrorRefreshRate = Double(screen?.maximumFramesPerSecond ?? 120)
            DiagnosticLog.shared.log("[DISPLAY] output moved to \(screen?.localizedName ?? "?") (refresh=\(Int(self.mirrorRefreshRate)))")
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

        // 소스 앱이 최전면을 벗어나면 오버레이를 양보(자동 숨김) — 단일 모니터에서
        // 오버레이가 다른 앱/설정 창을 계속 덮어 "갇히는" 문제 해소.
        workspaceObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.refreshOverlayVisibility()
            }
        }

        // 대상 창 닫힘 → SCK 스트림 중단 즉시 캡처 정지 (폴링 대기 없이 거의 동시)
        captureManager.onStreamStopped = { [weak self] in
            Task { @MainActor in
                guard let self, self.isCapturing, !self.isRestartingCapture else { return }
                DiagnosticLog.shared.log("[CAPTURE] source window gone (SCK stopped) → stop")
                await self.stopCapture()
            }
        }

        loadSettings()
        loadHotKeys()
    }

    /// 설정을 UserDefaults에 저장 (재시작해도 유지 — 설정 우선 앱 특성)
    /// UI 언어 적용 — **AppLanguage.current를 먼저 갱신한 뒤** uiLanguage를 세팅.
    /// onChange는 body 재평가 이후에 불려서 재구성이 옛 언어로 일어나던 버그(2026-07-05)를 회피:
    /// 커스텀 바인딩 set에서 이 메서드를 부르면 current가 관찰 트리거보다 먼저 갱신돼 즉시 반영됨.
    func setLanguage(_ raw: String) {
        AppLanguage.apply(raw: raw)
        uiLanguage = raw
    }

    /// 개발자 로그 토글 — on이면 /tmp/MacFG_diag.log 기록, off면 삭제+기록 중단.
    func updateDevLogging() { DiagnosticLog.shared.setEnabled(devLoggingEnabled) }

    /// Dock 표시 여부 — on이면 메뉴바 전용(.accessory, Dock/⌘Tab 제거), off면 일반(.regular).
    func updateMenuBarOnly() {
        NSApplication.shared.setActivationPolicy(menuBarOnly ? .accessory : .regular)
        if !menuBarOnly { NSApplication.shared.activate(ignoringOtherApps: true) }
        UserDefaults.standard.set(menuBarOnly, forKey: "s.menubaronly")
    }

    func persistSettings() {
        // 렌더 스레드 미러 동기화 (설정 변경은 전부 여길 지남)
        mirrorInterpolationEnabled = isInterpolationEnabled
        mirrorFrameMultiplier = frameMultiplier
        let d = UserDefaults.standard
        d.set(selectedRenderMode.rawValue, forKey: "s.engine")
        d.set(frameMultiplier, forKey: "s.mult")
        d.set(upscaleMode.rawValue, forKey: "s.upscale")
        d.set(casEnabled, forKey: "s.cas")
        d.set(sharpness, forKey: "s.sharp")
        d.set(sourcePreset, forKey: "s.preset")
        d.set(isInterpolationEnabled, forKey: "s.interp")
        d.set(occlusionDirectional, forKey: "s.occdir")
        d.set(motionSmoothness, forKey: "s.msmooth")
        d.set(boundarySoftness, forKey: "s.bsoft")
    }

    private func loadSettings() {
        let d = UserDefaults.standard
        if let e = d.string(forKey: "s.engine"), let m = RenderMode(rawValue: e) {
            selectedRenderMode = m; interpolationEngine = m.displayName
        }
        if d.object(forKey: "s.mult") != nil { frameMultiplier = d.integer(forKey: "s.mult") }
        if let u = d.string(forKey: "s.upscale"), let m = UpscaleMode(rawValue: u) { upscaleMode = m }
        if d.object(forKey: "s.cas") != nil { casEnabled = d.bool(forKey: "s.cas") }
        if d.object(forKey: "s.sharp") != nil { sharpness = d.double(forKey: "s.sharp") }
        if d.object(forKey: "s.preset") != nil { sourcePreset = d.integer(forKey: "s.preset") }
        if d.object(forKey: "s.interp") != nil { isInterpolationEnabled = d.bool(forKey: "s.interp") }
        if d.object(forKey: "s.occdir") != nil { occlusionDirectional = d.bool(forKey: "s.occdir") }
        MetalFlowEngine.occlusionDirectional = occlusionDirectional
        if d.object(forKey: "s.msmooth") != nil { motionSmoothness = d.double(forKey: "s.msmooth") }
        MetalFlowEngine.motionSmoothness = Float(motionSmoothness)
        if d.object(forKey: "s.bsoft") != nil { boundarySoftness = d.double(forKey: "s.bsoft") }
        MetalFlowEngine.boundarySoftness = Float(boundarySoftness)
        // 배치는 업스케일 모드에서 파생
        selectedOverlayPlacement = upscaleMode == .off ? .coverSource : .viewerWindow
    }

    /// 오클루전 방향별 워프 토글 (실험) — 정적 var를 워프가 매 쌍 읽으므로 캡처 중에도 즉시 반영.
    func updateOcclusionDirectional() {
        MetalFlowEngine.occlusionDirectional = occlusionDirectional
        DiagnosticLog.shared.log("[OCC] directional=\(occlusionDirectional)")
        persistSettings()
    }

    /// 모션 부드러움 슬라이더 — 정적 var를 워프/스무딩이 매 쌍 읽으므로 캡처 중 즉시 반영.
    func updateMotionSmoothness() {
        MetalFlowEngine.motionSmoothness = Float(motionSmoothness)
        persistSettings()
    }

    /// 경계 전환 슬라이더 — 정적 var를 워프가 매 쌍 읽으므로 캡처 중 즉시 반영.
    func updateBoundarySoftness() {
        MetalFlowEngine.boundarySoftness = Float(boundarySoftness)
        persistSettings()
    }

    private func handleScreenParametersChange() {
        guard isCapturing else { return }
        DiagnosticLog.shared.log("[DISPLAY] screen parameters changed → 렌더 링크 재부착")
        attachRenderDriver()
    }

    /// 렌더 드라이버를 현재 오버레이의 레이어에 부착 (배치 전환/화면 모드 변경 시 재호출)
    private func attachRenderDriver(watchDrawableGrowth: Bool = true) {
        guard let surface = overlayManager?.currentRenderSurface else {
            logger.warning("attachRenderDriver: no surface")
            return
        }
        renderSurface = surface
        mirrorRefreshRate = Double(overlayManager?.outputScreen?.maximumFramesPerSecond ?? 120)
        let attachW = Int(surface.metalLayer.drawableSize.width)
        renderDriver.attach(layer: surface.metalLayer) { [weak self] tick in
            self?.onDisplayLinkTick(
                timestamp: tick.timestamp,
                targetTimestamp: tick.targetPresentTimestamp,
                drawable: tick.drawable
            )
        }
        // CAMetalDisplayLink는 부착 시점의 drawableSize를 물어 그 크기의 드로어블을 vend한다.
        // 뷰어는 초기 창(960×540)으로 부착되므로, 첫 프레임이 레터박스 타깃(예: 3115×2160)으로
        // drawableSize를 키운 뒤에도 1-2초간 960×540 드로어블이 나와 고해상 업스케일 결과가
        // 다운스케일→재확대되어 흐릿하다(텍스트에서 특히 뚜렷, 실측). drawableSize가 커지면
        // 1회 재부착해 큰 드로어블을 즉시 vend하게 한다.
        if watchDrawableGrowth, selectedOverlayPlacement == .viewerWindow {
            scheduleDrawableReattach(afterWidth: attachW)
        }
    }

    /// drawableSize가 부착 시점보다 커지는 순간을 감지해 디스플레이링크를 1회 재부착 (뷰어 첫 캡처 흐림 해소)
    private func scheduleDrawableReattach(afterWidth: Int) {
        drawableReattachTimer?.invalidate()
        var ticks = 0
        drawableReattachTimer = Timer.scheduledTimer(withTimeInterval: 0.06, repeats: true) { [weak self] t in
            guard let self, self.isCapturing,
                  let surf = self.overlayManager?.currentRenderSurface else { t.invalidate(); return }
            ticks += 1
            let w = Int(surf.metalLayer.drawableSize.width)
            if w > afterWidth + 8 {
                t.invalidate()
                DiagnosticLog.shared.log("[DRIVER] drawableSize \(afterWidth)→\(w) 성장 감지 → 링크 재부착 (드로어블 크기 동기화)")
                self.attachRenderDriver(watchDrawableGrowth: false)
                self.forceRepresentTicks = 8   // 정적 콘텐츠도 재부착 후 큰 드로어블로 다시 그리게
            } else if ticks > 50 {   // ~3s 후 포기 (변화 없음)
                t.invalidate()
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
                  // layer 0 = 일반 창. floating(PiP 등, 보통 3)도 포함하되 메뉴바(24+)·데스크톱(<0)은 제외.
                  layer >= 0, layer < 24,
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
        }.filter { !Self.systemOwners.contains($0.ownerName) }
    }

    /// 캡처 대상 아님 — floating 레이어 완화로 딸려 나오는 시스템 UI 제외
    static let systemOwners: Set<String> = [
        "MacFG", "MacFGApp", "Dock", "Window Server", "WindowServer", "Control Center",
        "Notification Center", "Spotlight", "Wallpaper", "Screenshot", "SystemUIServer",
    ]

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
        if args.contains("--upscale") { upscaleMode = .aneMetalfx }
        if args.contains("--no-interp") { isInterpolationEnabled = false }
        // 영역 캡처 테스트: --capture-rect x,y,w,h (소스 창 상대 pt)
        if let rIdx = args.firstIndex(of: "--capture-rect"), rIdx + 1 < args.count {
            let parts = args[rIdx + 1].split(separator: ",").compactMap { Double($0) }
            if parts.count == 4 {
                captureRegion = CGRect(x: parts[0], y: parts[1], width: parts[2], height: parts[3])
                DiagnosticLog.shared.log("[AUTO] captureRegion=\(captureRegion!)")
            }
        }
        if let uIdx = args.firstIndex(of: "--upscale-mode"), uIdx + 1 < args.count,
           let m = UpscaleMode(rawValue: args[uIdx + 1]) {
            upscaleMode = m
            DiagnosticLog.shared.log("[AUTO] upscaleMode=\(m.rawValue)")
        }
        if let sIdx = args.firstIndex(of: "--sharpen"), sIdx + 1 < args.count,
           let v = Double(args[sIdx + 1]), (0.0...1.0).contains(v) {
            casEnabled = v > 0
            sharpness = v
            DiagnosticLog.shared.log("[AUTO] sharpen=\(v)")
        }
        if let mIdx = args.firstIndex(of: "--multiplier"), mIdx + 1 < args.count,
           let m = Int(args[mIdx + 1]), (2...5).contains(m) {
            frameMultiplier = m
            DiagnosticLog.shared.log("[AUTO] frameMultiplier=×\(m)")
        }
        if let fIdx = args.firstIndex(of: "--flow-base"), fIdx + 1 < args.count,
           let base = Double(args[fIdx + 1]), base >= 120, base <= 2048 {
            MetalFlowEngine.flowBaseLongSide = base
            DiagnosticLog.shared.log("[AUTO] flowBaseLongSide=\(Int(base))")
        }
        if args.contains("--occ-directional") {
            MetalFlowEngine.occlusionDirectional = true
            DiagnosticLog.shared.log("[AUTO] occlusionDirectional=on (실험 — 실영상 A/B용)")
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
            try await captureManager.startCapture(windowID: windowID, device: device, captureRect: captureRegion)
            captureMethod = captureManager.activeMethod.rawValue

            overlayManager?.setPlacement(selectedOverlayPlacement)
            try overlayManager?.start(windowID: windowID)
            overlayManager?.setUpscaleMode(upscaleMode)
            overlayManager?.setSharpness(casEnabled ? Float(sharpness) : 0)
            trackingMethod = overlayManager?.trackingMethod ?? "Unknown"

            // 엔진 준비를 먼저 끝낸 뒤 렌더 루프 시작 (전용 스레드 + CAMetalDisplayLink)
            await configurePairEngine()
            mirrorInterpolationEnabled = isInterpolationEnabled
            mirrorFrameMultiplier = frameMultiplier
            pendingShowReset = false
            attachRenderDriver()

            statsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.updateStats()
                }
            }

            // 창 추적은 30Hz면 충분 — 틱(120Hz)마다 CGWindowList를 부르면 호출당 0.5-2ms로
            // vsync 틱을 놓쳐 출력 fps 천장이 ~110으로 내려앉는다 (실측).
            // 뷰어 배치도 15Hz — 상대커서 매핑이 sourceFrameNS를 쓰므로, 드래그로 소스 창이
            // 움직였을 때 다음 조작 좌표가 어긋나지 않게 신선도가 필요 (2Hz는 0.5s 지연으로
            // 매핑이 헛돌았음). 렌더는 전용 스레드라 메인 CGWindowList 15Hz는 틱에 무해.
            let trackHz: Double = selectedOverlayPlacement == .coverSource ? 30.0 : 15.0
            trackingTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / trackHz, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    self.overlayManager?.updateTracking()
                    // 창 종료/리사이즈 감지 (렌더 틱에서 이관 — overlayManager는 MainActor)
                    guard self.isCapturing else { return }
                    if self.hasReceivedFirstFrame, (self.overlayManager?.trackingFailureCount ?? 0) > 30 {
                        DiagnosticLog.shared.log("[CAPTURE] target window gone (tracking) → stop")
                        await self.stopCapture()
                        return
                    }
                    let nowT = CFAbsoluteTimeGetCurrent()
                    if nowT - self.lastResizeCheck >= 0.5 {
                        self.lastResizeCheck = nowT
                        if !self.isRestartingCapture, self.captureRegion == nil, self.stablePoolWidth > 0,
                           let src = self.overlayManager?.sourcePixelSize {
                            let mismatch = abs(src.width - self.stablePoolWidth) > 8 || abs(src.height - self.stablePoolHeight) > 8
                            if mismatch {
                                self.resizeMismatchCount += 1
                                if self.resizeMismatchCount >= 2 {
                                    self.resizeMismatchCount = 0
                                    await self.resizeCaptureStream(width: src.width, height: src.height)
                                }
                            } else {
                                self.resizeMismatchCount = 0
                            }
                        }
                    }
                }
            }

            // 자동 숨김 기준용 소스 PID + 초기 표시 상태
            sourceOwnerPID = ownerPID(of: windowID)
            overlayManager?.sourcePID = sourceOwnerPID   // 뷰어 마우스 역매핑 대상
            overlayUserHidden = false
            overlayHiddenState = false
            isCapturing = true
            // 소스 앱을 최전면으로 활성화 — cover/viewer 공통.
            // 첫 캡처는 보통 MacFG 창이 최전면인 상태(옵션 설정 후 핫키/버튼)라, 소스(브라우저)가
            // 백그라운드로 밀려 렌더 품질이 떨어진다(브라우저 백그라운드 스로틀 — 첫 캡처만 저화질,
            // 정지 후 브라우저가 최전면 복귀해 2번째부턴 정상이던 증상의 원인). 소스를 활성 앱으로
            // 되돌리면 오버레이(floating/shielding 레벨)는 여전히 위에 뜨면서 소스는 풀품질 렌더.
            if sourceOwnerPID != 0 {
                NSRunningApplication(processIdentifier: sourceOwnerPID)?.activate()
            }
            logger.info("Capture started: \(self.captureMethod) + \(self.trackingMethod)")
            DiagnosticLog.shared.log("Capture started: \(captureMethod) + \(trackingMethod) mode=\(selectedRenderMode.rawValue)")

            // 미리 정한 소스 해상도 프리셋 적용 (추적/AX 준비 후). 영역 캡처 시엔 무의미 → 스킵.
            if sourcePreset != 0, captureRegion == nil {
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(500))
                    if isCapturing { resizeSourceToPreset(sourcePreset) }
                }
            }
        } catch {
            logger.error("Failed to start capture: \(error)")
            return
        }
    }

    func stopCapture() async {
        renderDriver.detach()   // 동기 — 반환 후 렌더 틱 없음 보장
        renderSurface = nil

        await captureManager.stopCapture()
        overlayManager?.stop()

        pairEngine?.shutdown()
        pairEngine = nil
        interpolationEngine = "None"

        statsTimer?.invalidate()
        statsTimer = nil
        trackingTimer?.invalidate()
        trackingTimer = nil
        drawableReattachTimer?.invalidate()   // 정지 후 재부착 방지 (detach된 링크 재생성 차단)
        drawableReattachTimer = nil
        forceRepresentTicks = 0

        isCapturing = false
        captureMethod = "None"
        trackingMethod = "None"
        sourceOwnerPID = 0
        overlayUserHidden = false
        overlayHiddenState = false
        // 스케줄러 리셋은 렌더 스레드에서 — detach로 틱이 멈췄어도, 메인에서 timeline을 직접
        // 비우면 마지막 in-flight 틱의 removeFirst(count-12)와 겹쳐 크래시났다(v1.1.0 실측:
        // "Can't remove more items than it has"). perform으로 렌더 런루프에 태워 직렬화.
        renderDriver.perform { [weak self] in self?.resetScheduler() }
        logger.info("Capture stopped")
        DiagnosticLog.shared.log("Capture stopped")
    }

    /// 무중단 리사이즈 — 스트림을 끊지 않고 출력 크기만 갱신 (SCStream.updateConfiguration).
    /// 전체 stop→start 재시작이 유발하던 수 초 붕괴(프레임 갭 + 케이던스 재락)를 없앤다.
    /// 전체화면/최대화 전환 시 present이 안 무너지는 것이 핵심 (실측: 전환 시 11~87fps 붕괴 → 제거).
    private func resizeCaptureStream(width: Int, height: Int) async {
        guard isCapturing, !isRestartingCapture else { return }
        isRestartingCapture = true
        defer { isRestartingCapture = false }
        do {
            try await captureManager.updateConfiguration(width: width, height: height)
            pendingShowReset = true   // 리셋은 렌더 스레드 틱에서 (동시 변조 방지)
            DiagnosticLog.shared.log("[CAPTURE] seamless resize → \(width)x\(height)")
        } catch {
            // updateConfiguration 미지원(IOSurface 폴백)/실패 → 기존 전체 재시작으로 폴백
            DiagnosticLog.shared.log("[CAPTURE] seamless resize failed (\(error)) → full restart")
            isRestartingCapture = false   // restartCaptureStream이 자체 플래그 관리
            await restartCaptureStream(reason: "resize → \(width)x\(height)")
        }
    }

    /// 캡처 스트림만 재시작 (오버레이/DisplayLink/엔진 유지) — 무중단 리사이즈 실패 시 폴백
    private func restartCaptureStream(reason: String) async {
        guard let windowID = selectedWindowID, isCapturing, !isRestartingCapture else { return }
        isRestartingCapture = true
        defer { isRestartingCapture = false }
        DiagnosticLog.shared.log("[CAPTURE] stream restart: \(reason)")
        await captureManager.stopCapture()
        do {
            try await captureManager.startCapture(windowID: windowID, device: device, captureRect: captureRegion)
            pendingShowReset = true
        } catch {
            DiagnosticLog.shared.log("[CAPTURE] stream restart FAILED: \(error) → 캡처 종료")
            await stopCapture()
        }
    }

    nonisolated private func resetScheduler() {
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
        // 적응 지연은 소스/세션 특성이므로 새 캡처에서 다시 학습
        extraLatencySlots = 0
        paceMissCount = 0
        paceCleanWindows = 0
        lastPaceAdjustTick = 0
        paceWarmupUntilTick = diagTick + 720   // ~6s 과도기 무시 (케이던스 락)
        pendingIngest = []
        inFlightPresents.withLock { $0 = 0 }
        _ = mailbox.drain()
    }

    /// 리사이즈 전용 경량 리셋 — 크기 의존 상태(타임라인/풀/이전 프레임)만 비우고
    /// 케이던스(스냅 링/EMA/타임스탬프)는 유지한다. 전체 리셋의 ~16프레임 재락을 회피.
    nonisolated private func softResetForResize() {
        timeline = []
        inFlightTextures = []
        stablePool = []
        stablePoolWidth = 0
        stablePoolHeight = 0
        prevStable = nil                 // 크기 바뀐 이전 프레임과 새 프레임은 페어 불가 → 1프레임 워밍업
        lastPresentedTexture = nil
        lastAcceptedFingerprint = 0
        resizeMismatchCount = 0
        _ = mailbox.drain()
        // 유지: snapTsRing / sourceIntervalEMA / snappedLastTimestamp / lastPresentedTimestamp /
        //       lastAcceptedTimestamp / hasReceivedFirstFrame → 콘텐츠 타이밍 연속성 보존
    }

    // MARK: - Scheduler State

    @ObservationIgnored nonisolated(unsafe) private var timeline: [TimelineEntry] = []
    @ObservationIgnored nonisolated(unsafe) private var stablePool: [any MTLTexture] = []
    @ObservationIgnored nonisolated(unsafe) private var stablePoolWidth = 0
    @ObservationIgnored nonisolated(unsafe) private var stablePoolHeight = 0
    @ObservationIgnored nonisolated(unsafe) private var inFlightTextures: Set<ObjectIdentifier> = []
    /// timestamp = 스냅된 콘텐츠 시각 (타임라인/보간용), rawTimestamp = SCK 원본 시각 (연속성 검사용)
    @ObservationIgnored nonisolated(unsafe) private var prevStable: (texture: any MTLTexture, timestamp: CFTimeInterval, rawTimestamp: CFTimeInterval)?
    @ObservationIgnored nonisolated(unsafe) private var lastAcceptedTimestamp: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var lastAcceptedFingerprint: UInt64 = 0
    /// 케이던스 스냅 상태 — 캡처 ts는 디스플레이 vsync 그리드(144Hz 등)에 양자화되고
    /// 실콘텐츠(브라우저)는 PTS가 [5~99ms]로 튄다. 중앙값 간격 + 앵커 그리드로 락을 유지하고
    /// 이탈 3연속일 때만 재동기 — 즉발 재동기는 그리드 정렬 위상을 흔들어 σ를 키운다 (실측).
    @ObservationIgnored nonisolated(unsafe) private var snapTsRing: [CFTimeInterval] = []
    @ObservationIgnored nonisolated(unsafe) private var snapMissStreak = 0
    @ObservationIgnored nonisolated(unsafe) private var snappedLastTimestamp: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var diagResyncCount = 0
    @ObservationIgnored nonisolated(unsafe) private var lastPresentedTimestamp: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var lastPresentedTexture: (any MTLTexture)?
    /// 링크 재부착 직후 남은 강제 재present 틱 수 — 정적 콘텐츠(새 프레임 없음)에서 흐린(작은
    /// 드로어블) 프레임을 새 드로어블 크기로 교체. 드로어블 풀(3) 순환분 커버.
    @ObservationIgnored nonisolated(unsafe) private var forceRepresentTicks = 0
    @ObservationIgnored nonisolated(unsafe) private var sourceIntervalEMA: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var hasReceivedFirstFrame: Bool = false
    /// 캡처 창 리사이즈 감지 (연속 감지 횟수 — 드래그 중 재시작 연발 방지, 메인 타이머 전용)
    @ObservationIgnored nonisolated(unsafe) private var resizeMismatchCount = 0
    private var lastResizeCheck: CFTimeInterval = 0
    /// 인제스트 이월 큐 — 버스트 틱(숨김 해제/재개 직후 최대 8장)의 인코딩 CPU가
    /// vsync 콜백을 삼키지 않게 틱당 4장 캡, 나머지는 다음 틱에서 처리
    @ObservationIgnored nonisolated(unsafe) private var pendingIngest: [FrameSlot] = []
    /// 인플라이트 present 수 (presentedHandler에서 감소 — 임의 스레드라 락 보호).
    /// 인플라이트 present 수 (presentedHandler에서 감소). 드로어블 포화 진단용 (drawBusy).
    private let inFlightPresents = OSAllocatedUnfairLock(initialState: 0)
    @ObservationIgnored nonisolated(unsafe) private var diagPresentBusy = 0
    @ObservationIgnored nonisolated(unsafe) private var isRestartingCapture = false
    @ObservationIgnored nonisolated(unsafe) private var presentedTimes: [CFTimeInterval] = []
    @ObservationIgnored nonisolated(unsafe) private var latencySamplesMs: [Double] = []
    /// 최근 vsync 목표 시각 — 보간 위상을 디스플레이 그리드에 정렬하기 위한 기준
    @ObservationIgnored nonisolated(unsafe) private var lastVsyncTarget: CFTimeInterval = 0

    /// 출력 지연: 콘텐츠 시각을 이만큼 과거로 조준한다.
    /// 보간 프레임 I(A,B)가 B 도착 + GPU/ANE 완료 후 표시 슬롯에 준비되어 있으려면
    /// 소스 간격의 ~1.25배 + 워크 마진이 필요. (60fps 소스 기준 ~25ms)
    /// + 적응분(extraLatencySlots): 소스 배달 지터(PiP/브라우저 srcInt 8~30ms 실측)로
    /// miss(staleDrop/지각 도착)가 지속되면 표시 슬롯 단위로 여유를 늘려 흡수 — LS식
    /// "여유 지연을 두고 큐를 안정 소비". 영상 시청엔 +1-2프레임 지연이 구멍보다 낫다.
    nonisolated private var latencyOffset: Double {
        let refresh = max(mirrorRefreshRate, 60)
        // 저지연 모드(보간 OFF) — 보간 프레임 페이싱 버퍼가 불필요하므로 오프셋을 최소로.
        // 인터랙티브(텍스트 드래그 등)에서 표시 지연을 확 줄인다(실측 56ms→~30ms). work(블릿+
        // 업스케일 ~5-8ms)를 커버할 만큼(~1.5슬롯)만 두고 소스 프레임을 곧장 표시.
        if !mirrorInterpolationEnabled {
            return 1.5 / refresh + 0.004 + extraLatencySlots / refresh
        }
        let interval = sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0
        // + 반 슬롯: SCK 배달이 소스 vsync에 양자화되어 최대 반 간격 늦게 오는데,
        // 그 마진이 없으면 늦은 쌍의 보간 프레임이 표시 시한을 놓쳐 stale-drop
        // (보간 188장 생성 → 113장 표시 실측 — present 110/s의 주범)
        let displayHalfSlot = 0.5 / refresh
        return min(max(interval, 1.0 / 120.0), 1.0 / 24.0) * 1.25 + 0.004 + displayHalfSlot
            + extraLatencySlots / refresh
    }

    // ── 적응형 페이싱 (AIMD): miss 지속 → +1슬롯(빠르게), 장기 무결 → -1슬롯(느리게) ──
    /// 추가 지연 (표시 슬롯 단위, 0~4). 정수 슬롯만 — 인코딩 그리드(gridRef)와 표시 타깃이
    /// 같은 격자를 유지해 위상 정렬이 깨지지 않는다 (전환 시 1회 홀드만).
    @ObservationIgnored nonisolated(unsafe) private var extraLatencySlots: Double = 0
    /// 최근 윈도 내 miss (staleDrop + 이미 기한 지난 도착)
    @ObservationIgnored nonisolated(unsafe) private var paceMissCount = 0
    @ObservationIgnored nonisolated(unsafe) private var paceCleanWindows = 0
    /// work(캡처→타임라인 등재) 지연 p90 [ms] — 적응 지연의 실측 하한 근거
    @ObservationIgnored nonisolated(unsafe) private var paceWorkP90: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var lastPaceAdjustTick = 0
    /// 이 틱까지는 miss 무시 — 캡처 시작/리셋 직후 케이던스 락 과도기의 miss로
    /// 깨끗한 소스까지 +4 램프되는 것 방지 (무지터 소스 e2e 57→91ms 낭비 실측)
    @ObservationIgnored nonisolated(unsafe) private var paceWarmupUntilTick = 0
    @ObservationIgnored nonisolated(unsafe) private var diagStaleSampleCount = 0

    /// ~2초마다: miss ≥4면 지연 +1슬롯 (최대 4), 3윈도(~6s) 연속 0이면 -1슬롯 회수.
    nonisolated private func adaptPacing() {
        guard diagTick - lastPaceAdjustTick >= 240 else { return }
        lastPaceAdjustTick = diagTick
        defer { paceMissCount = 0 }
        if adaptDisabled { extraLatencySlots = 0; return }      // A/B: 적응 지연 완전 차단
        // 저지연 모드(보간 OFF, 인터랙티브) — 적응 지연 램프 차단. 스무딩보다 반응성 우선
        // (램프하면 저지연 오프셋을 도로 상쇄해 텍스트 드래그가 다시 밀림, 실측 lat=+3).
        if !mirrorInterpolationEnabled { extraLatencySlots = 0; return }
        guard diagTick >= paceWarmupUntilTick else { return }   // 과도기 miss 폐기
        // 실측 work p90 기반 **하한** — 오프셋이 파이프라인 지연보다 얇으면 miss가 구조적으로
        // 반복되고, AIMD가 램프↔감쇠(6s 무결 -1 → 재miss → +1)를 오가며 표시 타깃이 슬롯
        // 단위로 출렁였다(톱니 = content wobble 기여, lat=+2→+3→+4 실측). 필요 슬롯을 직접
        // 계산해 즉시 올리고, 감쇠는 이 하한 아래로 내려가지 못하게 앵커한다.
        let refresh = max(mirrorRefreshRate, 60)
        let slotMs = 1000.0 / refresh
        let interval = min(max(sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0, 1.0 / 120.0), 1.0 / 24.0)
        let baseMs = (interval * 1.25 + 0.004 + 0.5 / refresh) * 1000.0
        let requiredExtra = paceWorkP90 > 0
            ? min(4.0, max(0.0, ((paceWorkP90 + 2.0 - baseMs) / slotMs).rounded(.up)))
            : 0.0
        if extraLatencySlots < requiredExtra {
            extraLatencySlots = requiredExtra
            paceCleanWindows = 0
            DiagnosticLog.shared.log("[PACE] work p90=\(String(format: "%.0f", paceWorkP90))ms → 지연 하한 \(Int(requiredExtra))슬롯")
        }
        if paceMissCount >= 4 {
            paceCleanWindows = 0
            if extraLatencySlots < 4 {
                extraLatencySlots += 1
                DiagnosticLog.shared.log("[PACE] miss \(paceMissCount)/2s → 지연 +1슬롯 (extra=\(Int(extraLatencySlots)))")
            }
        } else if paceMissCount == 0 {
            paceCleanWindows += 1
            if paceCleanWindows >= 3, extraLatencySlots > requiredExtra {
                extraLatencySlots -= 1
                paceCleanWindows = 0
                DiagnosticLog.shared.log("[PACE] 6s 무결 → 지연 -1슬롯 (extra=\(Int(extraLatencySlots)), 하한=\(Int(requiredExtra)))")
            }
        } else {
            paceCleanWindows = 0
        }
    }

    // ── 진단 ──
    @ObservationIgnored nonisolated(unsafe) private var diagTick: Int = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSourceCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagDupSkipCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTsRejectCount = 0        // 타임스탬프 비전진으로 스킵 (중복 프레임 재전송)
    @ObservationIgnored nonisolated(unsafe) private var diagPresentCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagInterpPresentCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagPoolExhaustCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagInterpEncodedCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagFrameTypes: [String] = []
    @ObservationIgnored nonisolated(unsafe) private var diagSrcIntMin: Double = .infinity  // 콘텐츠 간격 min/max (VFR 판별)
    @ObservationIgnored nonisolated(unsafe) private var diagSrcIntMax: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var diagDrainDepthSum: Int = 0         // 매 틱 drain한 프레임 수 (버스트 판별)
    @ObservationIgnored nonisolated(unsafe) private var diagDrainDepthMax: Int = 0
    @ObservationIgnored nonisolated(unsafe) private var diagDrainSamples: Int = 0
    // 보간 스킵 사유별 카운터 (interpEnc=0 재발 시 원인 특정)
    @ObservationIgnored nonisolated(unsafe) private var diagSkipToggleOff = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipEngineNil = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipNoPrev = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipContentFast = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipBigGap = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipDiscontinuity = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipEngineFail = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipOther = 0
    @ObservationIgnored nonisolated(unsafe) private var diagStaleDropCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipBackpressure = 0
    // 틱 핸들러 CPU 계측 — 8ms 초과 시 다음 vsync 콜백 스킵 = 틱 레이트 유실 (120 고정 실패 원인)
    @ObservationIgnored nonisolated(unsafe) private var diagTickCPUSum: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTickCPUMax: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTickOverruns = 0
    @ObservationIgnored nonisolated(unsafe) private var diagLastLogWall: CFTimeInterval = 0
    // 틱 갭 구조: link.timestamp 기준 vsync 스킵 감지 + 스킵 직전 틱의 CPU (범인 판별 —
    // 직전 cpu 낮은데 갭 = 핸들러 밖 메인스레드 작업(SwiftUI 등)이 콜백을 삼킨 것)
    @ObservationIgnored nonisolated(unsafe) private var diagLastTickTs: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var diagPrevTickCPU: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTickGaps = 0
    @ObservationIgnored nonisolated(unsafe) private var diagGapPrevCPUMax: Double = 0
    // 콘텐츠-시간 간격 (표시 프레임 간 콘텐츠 진행량 ms) — 균일성이 wobble의 직접 지표
    @ObservationIgnored nonisolated(unsafe) private var diagContentIntervals: [Double] = []
    // 적응형 지연 A/B용 (MACFG_NO_ADAPT=1이면 extraLatencySlots 0 고정 — 회귀 판별)
    private let adaptDisabled = ProcessInfo.processInfo.environment["MACFG_NO_ADAPT"] != nil

    // MARK: - Render Loop

    nonisolated private func onDisplayLinkTick(timestamp: CFTimeInterval, targetTimestamp: CFTimeInterval, drawable: any CAMetalDrawable) {
        let tickStart = CFAbsoluteTimeGetCurrent()
        defer {
            // 틱 핸들러 CPU 시간 계측 — 8.3ms 초과가 잦으면 다음 vsync 콜백이 스킵되어
            // 틱 레이트 자체가 117Hz로 새는(=120 고정 실패) 주범 (실사용 로그로 확증)
            let cpuMs = (CFAbsoluteTimeGetCurrent() - tickStart) * 1000
            diagTickCPUSum += cpuMs
            if cpuMs > diagTickCPUMax { diagTickCPUMax = cpuMs }
            if cpuMs > 8.0 { diagTickOverruns += 1 }
            diagPrevTickCPU = cpuMs
        }
        // vsync 스킵 감지 (link.timestamp 간격 > 1.4슬롯) — 직전 틱 CPU가 낮은데 갭이면
        // 핸들러 밖(다른 메인스레드 작업)이 콜백을 삼킨 것
        if diagLastTickTs > 0 {
            let dt = timestamp - diagLastTickTs
            if dt > 1.4 / max(mirrorRefreshRate, 60) {
                diagTickGaps += 1
                if diagPrevTickCPU > diagGapPrevCPUMax { diagGapPrevCPUMax = diagPrevTickCPU }
            }
        }
        diagLastTickTs = timestamp
        diagTick += 1
        lastVsyncTarget = targetTimestamp

        // 숨김→표시 전이: 메인이 신호만 세우고 리셋은 렌더 스레드 자신이 수행 (동시 변조 방지)
        if pendingShowReset {
            pendingShowReset = false
            resetScheduler()
            pairEngine?.reset()
        }

        // 오버레이 숨김(자동/수동) 중 — GPU 양보: 캡처/메일박스 파이프만 비우고
        // 보간·present는 생략한다 (사용자가 다른 앱으로 전환한 목적이 GPU 확보이므로).
        if overlayHiddenState {
            let (_, released, _) = mailbox.drain()
            for id in released { inFlightTextures.remove(id) }
            _ = captureManager.drainFrames()   // 파이프 적체 방지 (텍스처는 풀로 회수)
            pendingIngest = []
            return
        }

        // 1) 완료된 GPU 작업 수거 → 타임라인 등재
        let (newEntries, released, presented) = mailbox.drain()
        for id in released { inFlightTextures.remove(id) }
        if !newEntries.isEmpty {
            timeline.append(contentsOf: newEntries)
            timeline.sort { $0.timestamp < $1.timestamp }
            // 지각 도착 감지: 등재 시점에 이미 표시 기한(target)에서 1슬롯+ 지난 항목 —
            // 지연 여유(latencyOffset)가 소스 지터/워크 스파이크보다 얇다는 신호 (적응 지연 입력)
            let lateBar = targetTimestamp - latencyOffset - 1.0 / max(mirrorRefreshRate, 60)
            paceMissCount += newEntries.lazy.filter { $0.timestamp < lateBar }.count
        }
        statsLock.lock()
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
        statsLock.unlock()

        // 2) 표시 먼저 — 지연 민감 경로를 틱 선두로 (present-first).
        // 인제스트/인코딩(CPU 1-3ms+)을 먼저 하면 틱 핸들러가 간헐적으로 8.3ms를 넘겨
        // 다음 vsync 콜백이 스킵 → 틱 레이트가 117Hz로 새며 120 고정 실패 (실사용 로그 확증).
        // 이 틱에 캡처된 프레임은 어차피 GPU 완료 후 다음 틱에나 표시 가능하므로 손해 없음.
        timeline.removeAll { $0.timestamp <= lastPresentedTimestamp }
        // 방어적 클램프 — 재부착/리셋 타이밍 경합으로 count가 줄어도 "remove more than it has"
        // 크래시가 나지 않게 현재 count로 상한 (v1.1.0 크래시 실측 후 봉쇄).
        if timeline.count > 12 {
            timeline.removeFirst(min(timeline.count - 12, timeline.count))
        }
        let presentBefore = diagPresentCount
        presentDueEntry(targetTimestamp: targetTimestamp, drawable: drawable)
        // 링크 재부착 직후 강제 재present — 정적 콘텐츠는 새 프레임이 없어 presentDueEntry가
        // 아무것도 표시하지 않으므로, 재부착 전 그려둔 흐린(960×540 드로어블) 프레임이 남는다.
        // 이번 틱에 새 present가 없었으면 최신 텍스처를 새(큰) 드로어블에 다시 그려 교체한다.
        if forceRepresentTicks > 0 {
            forceRepresentTicks -= 1
            if diagPresentCount == presentBefore, let tex = lastPresentedTexture {
                let e = TimelineEntry(timestamp: lastPresentedTimestamp, texture: tex,
                                      isInterpolated: false, captureTimestamp: lastPresentedTimestamp)
                presentEntry(e, at: targetTimestamp, drawable: drawable)
            }
        }

        // 3) 캡처 프레임 drain → work 인코딩 (무거움 — 표시 이후로).
        // 버스트 캡: 숨김 해제/스트림 재개 직후 한 틱에 8장까지 몰리면 인코딩 CPU가
        // 8ms를 넘겨 다음 vsync 콜백을 삼킴 — 틱당 4장, 나머지는 다음 틱으로 이월
        // (타임스탬프 보존, 표시는 어차피 latencyOffset 뒤라 이월 8ms는 무해)
        var sawTexture = false
        var depth = 0
        pendingIngest.append(contentsOf: captureManager.drainFrames().filter { $0.texture != nil })
        let ingestNow = min(pendingIngest.count, 4)
        if ingestNow > 0 {
            for slot in pendingIngest.prefix(ingestNow) {
                sawTexture = true
                depth += 1
                ingest(slot)
            }
            pendingIngest.removeFirst(min(ingestNow, pendingIngest.count))
            diagDrainDepthSum += depth
            diagDrainSamples += 1
            if depth > diagDrainDepthMax { diagDrainDepthMax = depth }
        }

        if sawTexture {
            hasReceivedFirstFrame = true
        }
        // (창 종료/리사이즈 감지는 트래킹 타이머(메인)로 이동 — overlayManager는 MainActor)

        adaptPacing()
        maybeLogDiagnostics()
    }

    /// 표시할 항목 선택 + present — targetTimestamp에서 latencyOffset만큼 과거의 콘텐츠.
    /// 미표시 항목 중 가장 오래된 것부터 순서대로 (늦게 도착한 보간 프레임도 순서 보존).
    /// 최신-우선으로 고르면 워크 완료가 한 틱만 늦어도 보간 프레임이 영구 드랍된다.
    nonisolated private func presentDueEntry(targetTimestamp: CFTimeInterval, drawable: any CAMetalDrawable) {
        let target = targetTimestamp - latencyOffset
        let interval = min(max(sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0, 1.0 / 120.0), 1.0 / 24.0)
        // stale 한계: 평시 2.5 간격(늦은 묶음도 순서대로 표시 — 구멍보다 +1vsync 지연이 낫다),
        // 큐가 깊어지면 1.2로 조여 백로그를 서서히 배출 — 생성≈소비 균형에서 큐가
        // 고여 e2e가 +40ms 눌러앉는 것 방지 (실측 75-84ms → 목표 ~55ms).
        // '깊다' 기준은 적응 지연 슬롯만큼 상향 — extra 체제에선 tl 8-9가 정상 깊이인데
        // 이를 적체로 오판해 상시 타이트 드랍하던 것 방지 (staleDrop 8-14/2s 실측 → 완화)
        let staleCutoff = target - interval * (timeline.count >= 6 + Int(extraLatencySlots) ? 1.2 : 2.5)
        let candidates = timeline.filter { $0.timestamp > lastPresentedTimestamp && $0.timestamp <= target + 0.002 }
        var pick = candidates.first
        // 따라잡기는 틱당 최대 1장만 건너뜀 — 여러 장을 한 번에 버리면 눈에 보이는 점프.
        // 주의: staleDrop을 paceMiss로 세지 않는다 — 이 드롭은 큐 잉여 트림이라 지연을 늘려도
        // 안 사라지고(lat=+4에서도 지속 실측), miss로 세면 감쇠가 영영 막혀 e2e만 부푼다
        // (73→101ms 실측). 지연 부족(지각 도착)은 drain의 lateBar가 따로 센다.
        if let current = pick, current.timestamp < staleCutoff, candidates.count > 1 {
            pick = candidates[1]
            diagStaleDropCount += 1
            if diagStaleSampleCount < 4 {   // [진단] 드롭 원인 규명용 샘플
                diagStaleSampleCount += 1
                let ageMs = (target - current.timestamp) * 1000
                DiagnosticLog.shared.log("[STALE] age=\(String(format: "%.1f", ageMs))ms cutoff=\(String(format: "%.1f", (target - staleCutoff) * 1000))ms tl=\(timeline.count) cand=\(candidates.count) interp=\(current.isInterpolated)")
            }
        }
        if let pick {
            presentEntry(pick, at: targetTimestamp, drawable: drawable)
        }
    }

    /// 새 캡처 프레임 수용: 중복 제거 → 안정 복사 + 보간 인코딩 (비동기 GPU)
    nonisolated private func ingest(_ slot: FrameSlot) {
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

        // 색공간 전파 (변경시에만 — MainActor 홉)
        if slot.colorSpace !== lastSentColorSpace {
            lastSentColorSpace = slot.colorSpace
            let cs = slot.colorSpace
            Task { @MainActor [weak self] in self?.overlayManager?.setCaptureColorSpace(cs) }
        }

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
        let refreshRate = mirrorRefreshRate
        // 보간 스킵 사유 진단 (재현 시 원인 즉시 특정용)
        if !mirrorInterpolationEnabled { diagSkipToggleOff += 1 }
        else if pairEngine == nil { diagSkipEngineNil += 1 }
        else if prevStable == nil { diagSkipNoPrev += 1 }
        let wantInterpolation = mirrorInterpolationEnabled && pairEngine != nil
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
                if mirrorFrameMultiplier >= 2 {
                    // 정수배 모드: 출력 = 소스 fps × M 상한 (쌍당 M-1장 균등분할).
                    // 갭이 크면(드랍) 스텝 비례로 늘려 M배 케이던스 유지.
                    // M×fps가 주사율을 넘는 초과분은 표시 불가라 생성도 안 함.
                    let interval = sourceIntervalEMA > 0 ? sourceIntervalEMA : gap
                    let steps = max(1.0, (gap / interval).rounded())
                    let maxUseful = max(1, Int((interval / displayInterval).rounded()))
                    let m = min(mirrorFrameMultiplier, maxUseful)
                    let count = min(m * Int(steps) - 1, 8)
                    if count >= 1 {
                        // 균등분할. 주의: M×fps < 주사율이면 홀드가 2슬롯+ 이므로 위상 오차가
                        // 가끔 1/3슬롯 홀드로 튀는 양자화 지터(σ~3ms)는 불가피 — 표시 그리드
                        // 스냅도 개선 없음 실측 (소스 프레임이 소스 그리드에 있어 혼합 케이던스)
                        tValues = (1...count).map { Float($0) / Float(count + 1) }
                    }
                } else if gap / displayInterval > 1.5,
                          abs(gap / displayInterval - (gap / displayInterval).rounded()) < 0.12,
                          (gap / displayInterval).rounded() <= 9 {
                    // **정수 배율(스냅된 gap = 표시 슬롯의 정수배: 60→120=2, 30→120=4, 24→120=5)**
                    // 은 소스 그리드 균등분할 — 원본(S)은 소스 케이던스 그리드, 보간(I)은 vsync
                    // 그리드에 놓으면 두 클럭이 드리프트하며 표시 간격이 (슬롯±δ)로 교대해
                    // content wobble ±4ms를 만든다(실측 ±3.7). 균등분할이면 S·I 모두 소스 그리드
                    // 위 정확히 등간격 → 혼합 그리드 wobble 원천 소멸.
                    let n = Int((gap / displayInterval).rounded())
                    tValues = (1..<n).map { Float($0) / Float(n) }
                } else if lastVsyncTarget > 0 {
                    // 비정수 조합(60→144 = 쌍당 2.4슬롯 등)은 vsync 그리드 정렬 유지 —
                    // 균등분할은 쌍마다 2/3장을 오가며 시간축이 출렁인다(과거 실측).
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
                // 배압: 워크 스파이크로 큐가 깊어졌으면 생성 단계에서 줄인다 —
                // 이미 예약된 프레임을 드레인으로 버리는 것(눈에 보이는 딸꾹질)보다
                // 새 보간을 덜 만드는 쪽이 시각적으로 무해 (MetalFlow만 간헐 드랍 보고의 원인).
                // 임계값은 콘텐츠 fps에 비례 — 24fps는 쌍당 4-5장이라 tl=11이 '건강한' 깊이.
                // + 적응 지연 슬롯: extra만큼 큐가 깊어지는 건 의도(지터 흡수 버퍼)이므로
                // 배압이 이를 적체로 오판해 보간을 솎아내지 않게 문턱도 함께 올린다.
                let expectedDepth = Int((gap / displayInterval).rounded(.up)) * 2 + 3 + Int(extraLatencySlots)
                if timeline.count >= expectedDepth && tValues.count > 1 {
                    // 절반 솎아내기 (홀수 인덱스 유지)
                    tValues = tValues.enumerated().filter { $0.offset % 2 == 1 }.map(\.element)
                }
                if timeline.count >= expectedDepth + 4 {
                    tValues = []
                    diagSkipBackpressure += 1
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

    nonisolated private func presentEntry(_ entry: TimelineEntry, at targetTimestamp: CFTimeInterval, drawable: any CAMetalDrawable) {
        if inFlightPresents.withLock({ $0 }) >= 2 { diagPresentBusy += 1 }

        guard let presentQueue, let cb = presentQueue.makeCommandBuffer() else { return }
        guard let surface = renderSurface else { cb.commit(); return }
        // CAMetalDisplayLink가 배달한 드로어블에 직접 인코딩 — nextDrawable 없음
        surface.encode(texture: entry.texture, into: cb, drawable: drawable)

        // "왔다갔다"의 진짜 지표: 연속 표시 프레임의 콘텐츠-시간 간격 불균일.
        // 균일 모션이면 매 표시가 콘텐츠를 ~동일량 전진(60→120이면 ~8.3ms). 이게 출렁이면 wobble.
        // (glass σ는 표시 시각만 봐서 이 문제를 못 잡음 — 표시는 균등한데 콘텐츠가 출렁일 수 있음)
        if lastPresentedTimestamp > 0 {
            let cd = (entry.timestamp - lastPresentedTimestamp) * 1000.0
            if cd > 0 && cd < 100 { diagContentIntervals.append(cd) }
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
        let inFlightRef = inFlightPresents
        inFlightRef.withLock { $0 += 1 }
        drawable.addPresentedHandler { d in
            inFlightRef.withLock { $0 = max(0, $0 - 1) }
            mailboxRef.postPresented(at: d.presentedTime, captureTs: captureTs, isInterp: isInterp)
        }
        // CAMetalDisplayLink의 드로어블은 targetPresentTimestamp 슬롯에 이미 바인딩 —
        // plain present가 곧 그 슬롯 표시 (예전 plain-present 실험과 달리 시각이 링크에 고정됨)
        cb.present(drawable)
        cb.commit()
    }

    /// 캡처 타임스탬프를 콘텐츠 케이던스에 스냅 (단일 메커니즘: 케이던스 연속).
    ///
    /// snapped_n = snapped_{n-1} + round(rawDelta/interval)·interval — 위상 기준점 없이
    /// 직전 스냅에서 정수 스텝 전진. 간격은 링 시간폭/프레임수(양자화 무편향).
    /// 앵커 위상 방식과 병행하면 서로 다른 위상으로 스냅이 섞여 갭이 8/25ms로 요동
    /// (실측) — 단일 메커니즘이 자기일관적이라 갭이 균일해진다.
    nonisolated private func snapTimestamp(raw: CFTimeInterval, rawDelta: Double) -> CFTimeInterval {
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

    nonisolated private func resetSnapState() {
        snapTsRing = []
        snapMissStreak = 0
        snappedLastTimestamp = 0
    }

    /// 사용 중이지 않은 풀 텍스처 획득 (타임라인/직전 소스/마지막 표시/인플라이트 제외)
    nonisolated private func acquireStableTexture(width: Int, height: Int) -> (any MTLTexture)? {
        if width != stablePoolWidth || height != stablePoolHeight {
            stablePool = []
            stablePoolWidth = width
            stablePoolHeight = height
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead]
            desc.storageMode = .private
            // 8장: 적응 지연(+최대 4슬롯)으로 타임라인이 깊어져도 소스 수용이 거절되지 않게
            // (6장에서 poolMiss 1-2/2s 실측 — 거절 시 연속성 리셋 = 눈에 보이는 홀드)
            for _ in 0..<8 {
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

    nonisolated private func maybeLogDiagnostics() {
        guard diagTick % 240 == 0 else { return }  // 진단 중 2초 주기 (@120Hz)

        // 틱 레이트: 240틱의 실제 벽시계 소요 — 2.0s면 무손실 120Hz, 2.05s면 ~117Hz(틱 유실)
        let nowWall = CFAbsoluteTimeGetCurrent()
        let tickHz = diagLastLogWall > 0 ? 240.0 / (nowWall - diagLastLogWall) : 0
        diagLastLogWall = nowWall
        let tickCPUAvg = diagTickCPUSum / 240.0
        let tickStats = String(format: "tick=%.1fHz cpu=%.1f/%.1fms over=%d gap=%d(pre%.1f)", tickHz, tickCPUAvg, diagTickCPUMax, diagTickOverruns, diagTickGaps, diagGapPrevCPUMax)
        diagTickCPUSum = 0; diagTickCPUMax = 0; diagTickOverruns = 0
        diagTickGaps = 0; diagGapPrevCPUMax = 0
        // 콘텐츠 간격 통계 (wobble 지표)
        let ci = diagContentIntervals
        let ciAvg = ci.isEmpty ? 0 : ci.reduce(0, +) / Double(ci.count)
        let ciVar = ci.isEmpty ? 0 : ci.map { ($0 - ciAvg) * ($0 - ciAvg) }.reduce(0, +) / Double(ci.count)
        let ciStats = String(format: "content=%.1f±%.1fms", ciAvg, sqrt(ciVar))
        diagContentIntervals = []

        let workLats = mailbox.drainWorkLatencies()
        let avgWork = workLats.isEmpty ? 0 : workLats.reduce(0, +) / Double(workLats.count)
        let maxWork = workLats.max() ?? 0
        // work p90 추적 (적응 지연 하한의 근거) — 상승은 즉시, 감쇠는 10%/2s (스파이크 견고)
        if !workLats.isEmpty {
            let sorted = workLats.sorted()
            let p90 = sorted[min(sorted.count - 1, (sorted.count * 9) / 10)]
            paceWorkP90 = max(p90, paceWorkP90 * 0.9)
        }
        let cuts = mailbox.drainSceneCutCount()

        // presented 간격 통계 (실제 glass 시각 기반 — 스무스니스의 ground truth)
        statsLock.lock()
        let presentedSnapshot = presentedTimes
        statsLock.unlock()
        var intervals: [Double] = []
        if presentedSnapshot.count >= 2 {
            for i in 1..<presentedSnapshot.count {
                let d = (presentedSnapshot[i] - presentedSnapshot[i - 1]) * 1000.0
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
        if diagStaleDropCount > 0 { skipParts.append("staleDrop:\(diagStaleDropCount)") }
        if diagSkipBackpressure > 0 { skipParts.append("backpres:\(diagSkipBackpressure)") }
        if diagPresentBusy > 0 { skipParts.append("drawBusy:\(diagPresentBusy)") }
        let skips = skipParts.isEmpty ? "-" : skipParts.joined(separator: ",")

        let msg = "[SCHED] src=\(diagSourceCount)(\(String(format: "%.0f", srcFps))fps) uniqOut=\(uniquePresented) dupSkip=\(diagDupSkipCount) tsRej=\(diagTsRejectCount) interpEnc=\(diagInterpEncodedCount) skip[\(skips)] present=\(diagPresentCount) (I=\(diagInterpPresentCount)) lat=+\(Int(extraLatencySlots)) \(tickStats) \(ciStats) cut=\(cuts) resync=\(diagResyncCount) poolMiss=\(diagPoolExhaustCount) tl=\(timeline.count) | glass(ms): avg=\(String(format: "%.2f", avgInterval)) σ=\(String(format: "%.2f", sqrt(variance))) max=\(String(format: "%.1f", maxInterval)) | srcInt=\(String(format: "%.1f", sourceIntervalEMA * 1000))ms [\(String(format: "%.0f", srcIntLo))~\(String(format: "%.0f", srcIntHi))] | drain=\(String(format: "%.1f", drainAvg))/\(diagDrainDepthMax) | work=\(String(format: "%.0f", avgWork))/\(String(format: "%.0f", maxWork))ms e2e=\(String(format: "%.0f", avgLatency))ms | \(pattern)"
        DiagnosticLog.shared.log(msg)
        diagResyncCount = 0
        diagSkipToggleOff = 0; diagSkipEngineNil = 0; diagSkipNoPrev = 0
        diagSkipContentFast = 0; diagSkipBigGap = 0; diagSkipDiscontinuity = 0
        diagSkipEngineFail = 0; diagSkipOther = 0
        diagStaleDropCount = 0; diagSkipBackpressure = 0; diagPresentBusy = 0; diagStaleSampleCount = 0

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

    /// 설정 창이 실제로 보이는가 (일반 레벨 titled 창의 occlusion) — 전체화면 뷰어가
    /// 덮고 있으면 false. 오버레이/뷰어는 borderless라 제외됨.
    private var settingsWindowVisible: Bool {
        NSApp.windows.contains {
            $0.styleMask.contains(.titled) && $0.level == .normal && $0.occlusionState.contains(.visible)
        }
    }

    private func updateStats() {
        // 아무도 못 보는 설정 창엔 갱신 생략 — 전체화면 뷰어가 덮은 상태에서도 stats 쓰기가
        // SwiftUI 레이아웃(NSHostingView, 4K에서 10-20ms)을 돌려 vsync 콜백을 삼킴 (sample 실측).
        // 시청 모드(뷰어 전체화면)에서 SwiftUI 부하를 0으로 — 설정을 보면 자동 재개.
        guard settingsWindowVisible else { return }
        // @Observable 쓰기는 값이 실제로 바뀔 때만 — 렌더 틱과 같은 메인스레드에서
        // SwiftUI가 설정 뷰 body를 재평가(4K에서 5-15ms)해 vsync 콜백을 삼키는 것 방지.
        // 지터 콘텐츠에서 fps 소수점이 매번 달라 0.5s마다 전체 뷰 무효화 → tick 116-117Hz로
        // 새던 원인 (깨끗한 소스는 값 불변 → 재평가 없음 → 120.0 실측과 정합).
        // 반올림(fps 정수, latency 정수)로 변경 빈도 자체도 낮춘다.
        let newInput = performanceMonitor.inputFPS.rounded()
        if inputFPS != newInput { inputFPS = newInput }
        // 출력 FPS는 실제 glass 시각(presented handler의 presentedTime)으로 계산 —
        // PerformanceMonitor의 renderTimestamps는 mailbox 드레인 시각이라 틱에 뭉쳐
        // 표시값이 80-120으로 맥놀이 (실프레임은 꾸준한데 지표만 출렁, 실측)
        statsLock.lock()
        let ptSnap = presentedTimes
        let latSnap = latencySamplesMs
        statsLock.unlock()
        var newOutput: Double = 0
        if ptSnap.count >= 2,
           let first = ptSnap.first, let last = ptSnap.last, last > first {
            newOutput = (Double(ptSnap.count - 1) / (last - first)).rounded()
        }
        if outputFPS != newOutput { outputFPS = newOutput }
        let newLatency = (latSnap.isEmpty ? 0 : latSnap.reduce(0, +) / Double(latSnap.count)).rounded()
        if latencyMs != newLatency { latencyMs = newLatency }
        let newScale = overlayManager?.scaleStatus
        if upscaleStatus != newScale { upscaleStatus = newScale }
    }

    // MARK: - Interpolation Control

    func updateInterpolationEnabled() {
        persistSettings()
        if isCapturing {
            Task { @MainActor in
                await configurePairEngine()
            }
            return
        }
        interpolationEngine = isInterpolationEnabled ? selectedRenderMode.displayName : "Off"
    }

    func updateRenderMode() {
        persistSettings()
        if isCapturing {
            Task { @MainActor in
                await configurePairEngine()
            }
        } else {
            interpolationEngine = selectedRenderMode.displayName
        }
    }

    func updateUpscale() {
        persistSettings()
        overlayManager?.setUpscaleMode(upscaleMode)
        overlayManager?.setSharpness(casEnabled ? Float(sharpness) : 0)
    }

    /// 배치는 업스케일 모드에서 자동 결정 (사용자 선택 없음): 업스케일 쓰면 Separate Window(실효),
    /// 안 쓰면 Cover. 캡처 중 변경 시 오버레이 재생성.
    func autoSelectPlacementForUpscale() {
        let target: OverlayPlacement = upscaleMode == .off ? .coverSource : .viewerWindow
        guard target != selectedOverlayPlacement else { return }
        selectedOverlayPlacement = target
        if isCapturing { updateOverlayPlacement() }
    }

    /// 캡처 중인 소스 창을 프리셋(짧은 변 px)으로 리사이즈 — 종횡비 유지, 방향 자동 감지.
    /// 가로 영상: 짧은 변=세로(=preset). 세로 영상(직캠): 짧은 변=가로(=preset).
    /// 소스가 영상 네이티브 해상도에 맞을수록 1:1 렌더 → 깨끗한 캡처 → 업스케일 효과↑.
    /// 주의: AX는 창 전체(타이틀바 포함)를 리사이즈 → 타이틀바 있는 창은 영상이 그만큼 더 작음.
    func resizeSourceToPreset(_ shortSide: Int) {
        guard isCapturing, let src = overlayManager?.sourcePixelSize, src.width > 0, src.height > 0 else { return }
        let aspect = Double(src.width) / Double(src.height)
        let targetW: Int, targetH: Int
        if src.width >= src.height {   // 가로 영상: 짧은 변 = 세로
            targetH = shortSide
            targetW = Int((Double(shortSide) * aspect).rounded())
        } else {                        // 세로 영상(직캠): 짧은 변 = 가로
            targetW = shortSide
            targetH = Int((Double(shortSide) / aspect).rounded())
        }
        let ok = overlayManager?.resizeSourceWindow(toPixelWidth: targetW, height: targetH) ?? false
        DiagnosticLog.shared.log("[PRESET] resize source → \(targetW)x\(targetH) (\(ok ? "ok" : "AX 실패"))")
    }

    func updateOverlayPlacement() {
        overlayManager?.setPlacement(selectedOverlayPlacement)
        if isCapturing { attachRenderDriver() }
        // 배치 전환 시 숨김 상태 초기화 (뷰어는 자동 숨김 대상 아님)
        overlayUserHidden = false
        overlayHiddenState = false
        refreshOverlayVisibility()
    }

    // MARK: - Overlay Visibility / Hotkeys

    private func ownerPID(of windowID: CGWindowID) -> pid_t {
        guard let info = CGWindowListCopyWindowInfo([.optionIncludingWindow], windowID) as? [[String: Any]],
              let pid = info.first?[kCGWindowOwnerPID as String] as? pid_t else { return 0 }
        return pid
    }

    /// 소스 최전면 여부 + 수동 숨김을 종합해 오버레이 표시/숨김을 적용.
    /// 숨김→표시 전이 시 스케줄러/엔진을 리셋해 끊긴 A→B 연속성을 정리한다.
    func refreshOverlayVisibility() {
        guard isCapturing else { return }
        // 뷰어 배치는 자동 숨김 대상 아님 (사용자가 직접 제어하는 일반 창)
        guard selectedOverlayPlacement == .coverSource else {
            overlayHiddenState = false
            return
        }
        // MacFG 자신이 최전면일 땐 숨기지 않음 — 설정 창에서 FPS를 보는 동안 보간이 멈추는
        // "관찰자 효과" 방지. 자동 숨김의 목적(다른 앱 가림 해소)은 제3앱일 때만 유효.
        let frontPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let sourceFront = sourceOwnerPID == 0
            || frontPID == sourceOwnerPID
            || frontPID == ProcessInfo.processInfo.processIdentifier
        let shouldHide = overlayUserHidden || !sourceFront
        guard shouldHide != overlayHiddenState else { return }
        overlayHiddenState = shouldHide
        overlayManager?.setOverlayHidden(shouldHide)
        if shouldHide {
            DiagnosticLog.shared.log("[OVERLAY] hidden (front≠source or manual)")
        } else {
            // 숨김 동안 프레임을 버려 연속성이 끊김 — 리셋은 렌더 스레드가 자기 틱에서 수행
            pendingShowReset = true
            DiagnosticLog.shared.log("[OVERLAY] shown → scheduler reset (render-thread)")
        }
    }


    /// 현재 최전면 앱의 최상단 일반 창 (MacFG 제외). ⌃⌥⌘U 원샷 캡처용.
    private func frontmostWindow() -> (id: CGWindowID, name: String)? {
        guard let frontPID = NSWorkspace.shared.frontmostApplication?.processIdentifier else { return nil }
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return nil }
        // 목록은 앞→뒤 순서 — frontmost 앱의 첫 유효 창을 고른다
        for info in list {
            guard let pid = info[kCGWindowOwnerPID as String] as? pid_t, pid == frontPID,
                  let layer = info[kCGWindowLayer as String] as? Int, layer >= 0, layer < 24,
                  let owner = info[kCGWindowOwnerName as String] as? String, !Self.systemOwners.contains(owner),
                  let wid = info[kCGWindowNumber as String] as? CGWindowID,
                  let b = info[kCGWindowBounds as String] as? [String: CGFloat],
                  (b["Width"] ?? 0) > 50, (b["Height"] ?? 0) > 50 else { continue }
            let name = info[kCGWindowName as String] as? String ?? ""
            return (wid, name.isEmpty ? owner : "\(owner) — \(name)")
        }
        return nil
    }

    /// ⌃⌥⌘U / 버튼: 포커스 창 캡처 토글 — 설정(배치/엔진/업스케일)은 앱에서 미리 정한 대로.
    /// 캡처 중이면 정지(LS식 단일 토글). 뷰어 배치면 전체화면으로.
    func toggleCaptureFocused() {
        Task { @MainActor in
            if isCapturing { await stopCapture(); return }
            guard let target = frontmostWindow() else {
                DiagnosticLog.shared.log("[HOTKEY] no focused window to capture")
                return
            }
            selectedWindowID = target.id
            selectedWindowName = target.name
            await startCapture()
        }
    }

    /// 전역 단축키 등록 (앱 시작 시 + 변경 시). 포커스 창 캡처 토글 하나.
    /// keyCode 0 = 미설정 → 등록 생략.
    func registerHotKeys() {
        var bindings: [HotKeyCenter.Binding] = []
        if hotCapture.keyCode != 0 {
            bindings.append(.init(id: 3, keyCode: hotCapture.keyCode, modifiers: hotCapture.modifiers) { [weak self] in
                self?.toggleCaptureFocused()
            })
        }
        if hotInterp.keyCode != 0 {
            bindings.append(.init(id: 4, keyCode: hotInterp.keyCode, modifiers: hotInterp.modifiers) { [weak self] in
                self?.toggleInterpolationHotkey()
            })
        }
        HotKeyCenter.shared.register(bindings)
    }

    /// 보간 on/off 전역 토글 (라이브 반영) — 동영상↔텍스트 즉시 전환
    func toggleInterpolationHotkey() {
        isInterpolationEnabled.toggle()
        updateInterpolationEnabled()
        DiagnosticLog.shared.log("[HOTKEY] 보간 \(isInterpolationEnabled ? "ON(동영상)" : "OFF(저지연)")")
    }

    /// 단축키 변경 시: UserDefaults 저장 + 재등록
    func updateHotKeys() {
        if let data = try? JSONEncoder().encode(hotCapture) { UserDefaults.standard.set(data, forKey: "hk.capture") }
        if let data = try? JSONEncoder().encode(hotInterp) { UserDefaults.standard.set(data, forKey: "hk.interp") }
        registerHotKeys()
    }

    private func loadHotKeys() {
        if let d = UserDefaults.standard.data(forKey: "hk.capture"),
           let b = try? JSONDecoder().decode(HotKeyBinding.self, from: d) { hotCapture = b }
        if let d = UserDefaults.standard.data(forKey: "hk.interp"),
           let b = try? JSONDecoder().decode(HotKeyBinding.self, from: d) { hotInterp = b }
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
        case .rife:
            if RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) {
                engine = RIFEEngine()
            } else {
                DiagnosticLog.shared.log("[ENGINE] RIFE model missing → MetalFlow fallback")
                engine = MetalFlowEngine()
            }
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
