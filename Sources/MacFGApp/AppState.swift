import SwiftUI
import AppKit
import Carbon.HIToolbox
@preconcurrency import Metal
import QuartzCore
import MetalPerformanceShaders
import CaptureKit
import Overlay
import FramePacing
import Interpolation
import Monitoring
import os
import ImageIO
import UniformTypeIdentifiers
import Vision

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
    /// 엔진 출력 링 슬롯의 세대 도장 (0 = 소스/검증 불요). 표시 직전 engine.isFrameLive로
    /// "후속 warp가 이 텍스처를 덮지 않았나" 확인 — burst 오버런 시 깨끗한 드롭용.
    var stamp: UInt64 = 0
}

/// GPU 완료/presented 핸들러(백그라운드 스레드) → 렌더 틱(MainActor) 전달함
private final class RenderMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [TimelineEntry] = []
    private var releasedTextures: [ObjectIdentifier] = []
    private var presentedRecords: [(presentedAt: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool)] = []

    func postCompleted(entries newEntries: [TimelineEntry], released: ObjectIdentifier?, workLatencyMs: Double, sceneCut: Bool) {
        lock.lock()
        entries.append(contentsOf: newEntries)
        if let released { releasedTextures.append(released) }
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
    /// Cover 모드에서 다른 앱으로 전환해도 오버레이를 계속 표시 (기본 ON — 시청 중 멀티태스킹이
    /// 핵심 사용이라). off면 제3앱 최전면 시 오버레이 숨김+보간 정지(GPU 양보, 단일모니터 전체화면
    /// Cover 트랩 회피용). 창 소스는 소스 영역만 덮으니 켜둬도 안전; 트랩 시 보간/오버레이 단축키로 escape.
    var coverKeepVisible: Bool = true
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
    /// 정보 오버레이 토글 — 뷰어 좌상단에 소스/보간/업스케일 정보 표시.
    var hotInfo = HotKeyBinding(keyCode: UInt32(kVK_ANSI_K), modifiers: UInt32(controlKey | optionKey | cmdKey), label: "⌃⌥⌘K")
    /// 정보 오버레이 표시 상태 (세션 상태 — 비영속)
    @ObservationIgnored nonisolated(unsafe) var infoOverlayVisible = false
    /// 메뉴바 팝오버 열림 여부 (MenuBarExtra 콘텐츠 onAppear/onDisappear) — stats 갱신 게이트
    @ObservationIgnored nonisolated(unsafe) var popoverVisible = false
    /// Quit 버튼 → AppDelegate. 델리게이트가 "사용자 요청"으로 표시해야 종료가 허용된다
    /// (시스템發 종료 요청은 거부해 메뉴바 상주를 지킨다 — MacFGApp.swift 주석 참조).
    @ObservationIgnored var onQuitRequested: (() -> Void)?

    // MARK: - Components
    let device: any MTLDevice
    /// 보간/복사 작업용 큐 — present 큐와 분리해 4-5ms 보간 작업이 present를 막지 않게 한다
    @ObservationIgnored nonisolated(unsafe) private var workQueue: (any MTLCommandQueue)?
    /// present 전용 큐 (틱당 ~0.3ms 렌더패스만)
    @ObservationIgnored nonisolated(unsafe) private var presentQueue: (any MTLCommandQueue)?
    /// cb1(stable blit + UI 검출) 전용 copy 큐 (O1-3) — workQueue에서 분리하면 cb2(warp)의
    /// predict 이벤트 대기가 다음 프레임 blit을 head-of-line 블로킹하던 것을 없애 파이프라인
    /// 중첩(predict↔warp)을 복원. MACFG_SPLITQ=1로 활성(검증 전 기본 OFF — 동시성 변경).
    @ObservationIgnored nonisolated(unsafe) private var copyQueue: (any MTLCommandQueue)?
    /// cb1(blit)을 별도 copy 큐로 분리할지 — **엔진별로 다르다.**
    /// 이 분리가 노리는 병목은 "cb2가 RIFE predict(ANE) 이벤트를 기다리며 workQueue를 점유해
    /// 다음 프레임 blit이 head-of-line 블로킹되는 것"이다. MetalFlow는 predict 대기 자체가 없어
    /// 이득 경로가 존재하지 않는다 — 실측도 그랬다(N=20: MetalFlow σ 0.78→0.84·tick 119.3→117.4 손해,
    /// RIFE σ 1.49→1.21·편차 ±1.18→±0.74 개선). 그래서 RIFE에서만 켠다. MACFG_SPLITQ로 수동 오버라이드.
    @ObservationIgnored nonisolated(unsafe) private var splitQueueEnabled = false
    @ObservationIgnored private let splitQueueOverride: Bool? = ProcessInfo.processInfo.environment["MACFG_SPLITQ"].map { $0 == "1" }

    /// 단계별 지연 분해 계측 (MACFG_STAGEDBG=1). 라이브 work(50~80ms)가 큐 대기인지 GPU 실행인지
    /// 가른다: capIngest(캡처→인제스트 = SCK/큐 대기), cb1(blit+검출 GPU), cb2(warp GPU),
    /// work(캡처→cb2완료 총). GPU 시간은 cb.gpuStart/EndTime(대기 제외 순수 실행). 부하와 무관하게
    /// **비율**이 병목을 드러낸다. 완료 핸들러가 임의 스레드라 stageLock으로 누적.
    /// **개발 로그 토글에 묶는다.** 예전엔 MACFG_STAGEDBG=1 환경변수 전용이었는데, 환경변수는
    /// Finder에서 더블클릭으로 켠 .app에는 전달되지 않는다 — 즉 이 계측이 **실사용에서 구조적으로
    /// 도달 불가**였고, "e2e의 70ms가 어디서 오는가"라는 질문에 답할 유일한 도구가 죽어 있었다.
    /// 로그 파일에만 쓰고 렌더 경로에 분기 하나를 더할 뿐이라 켜져 있어도 비용이 없다.
    @ObservationIgnored nonisolated(unsafe) private var stageDbg =
        ProcessInfo.processInfo.environment["MACFG_STAGEDBG"] == "1"
        || UserDefaults.standard.bool(forKey: "s.devlog")
    @ObservationIgnored nonisolated(unsafe) private var stgCapIngest = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCb1Gpu = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCb2Gpu = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgWork = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCount = 0
    @ObservationIgnored private let stageLock = NSLock()
    /// 직전 cb1 GPU 시간 — 스파이크 한 프레임을 단계별로 찍기 위해 값 자체를 들고 있는다
    /// (누적합만으론 어느 단계가 튀었는지 알 수 없다).
    @ObservationIgnored nonisolated(unsafe) private var stgLastCb1Gpu = 0.0
    /// work의 완만한 EMA — 스파이크 판정 기준선. 고정 문턱을 쓰면 4K/1080p·엔진마다 의미가 달라진다.
    @ObservationIgnored nonisolated(unsafe) private var stgWorkEMA = 0.0
    /// 스파이크 로그 최소 간격 (초) — 폭주 방지
    @ObservationIgnored nonisolated(unsafe) private var stgLastSpikeLog = 0.0
    /// UI 정적 검출 스트라이드 카운터 — 매 6프레임만 4K 검출 갱신 (백로그 증폭 방지)
    @ObservationIgnored nonisolated(unsafe) private var uiDetectFrame = 0
    /// 보간 엔진(cb2)의 GPU 실행시간 EMA [ms] — 자동 flow 스케일러 입력
    @ObservationIgnored nonisolated(unsafe) private var engineGpuMsEMA: Double = 0
    /// 기기 시딩 대기 — 소스 해상도가 정해지는 첫 프레임에서 1회 수행
    @ObservationIgnored nonisolated(unsafe) private var seedPending = true

    private let captureManager = CaptureManager()
    // U2 전체화면 재타깃: 사용자가 고른 원 창 / 현재 실제 캡처 중인 창(전체화면 시 전환).
    private var originalCaptureWindowID: CGWindowID = 0
    private var currentTargetWindowID: CGWindowID = 0
    private var retargetInFlight = false

    // 전체화면 자동 뷰어: 소스가 전체화면이면 Cover→Viewer 자동(수동 토글 불필요), 창 복귀 시 원복.
    @ObservationIgnored private var autoFsViewer = false
    /// 마지막으로 소스 프레임이 도착한 시각 — 좀비 오버레이(얼어붙은 화면) 판정의 직접 신호.
    @ObservationIgnored nonisolated(unsafe) var lastFrameArrivalAt: CFAbsoluteTime = 0

    /// GitHub 릴리즈 업데이트 확인 (알림만, 설치는 사용자가)
    let updateChecker = UpdateChecker()

    /// 부하 거버너 — 성능이 예산을 못 맞추면 화질 다이얼을 단계적으로 낮춘다 (O2-1)
    let loadGovernor = LoadGovernor()
    /// MetalFlow flow 해상도 자동 조절 — 기기별로 목표 프레임을 맞출 때까지 실측 수렴
    let autoFlowScaler = AutoFlowScaler()

    /// 거버너 레벨을 실제 다이얼에 반영. 레벨이 바뀐 순간에만 실질 작업이 일어난다.
    func applyGovernorDials() {
        // ① MetalFlow flow 해상도 — 자동 스케일러가 기기 성능에 맞춰 정한 값(수동 지정 시 그 값)에
        //    거버너 상한을 씌운 것. 거버너는 상한만 내리고, 그 안에서 스케일러가 움직인다.
        //    엔진 자율 사다리(RIFE)는 건드리지 않는다 (같은 다이얼 이중 조작 = 발진).
        let desired = autoFlowScaler.manualOverride ? userFlowBase : autoFlowScaler.current
        let base = min(desired, loadGovernor.flowBaseCap ?? desired)
        if MetalFlowEngine.flowBaseLongSide != base {
            MetalFlowEngine.flowBaseLongSide = base
            DiagnosticLog.shared.log("[GOV] flowBase → \(Int(base))")
        }
        // RIFE 워프 해상도 배율 (LSFG식) — RIFE의 실질 중간 강등 다이얼. env 수동 오버라이드가
        // 있으면 그걸 존중(측정용), 없으면 거버너가 설정.
        if ProcessInfo.processInfo.environment["MACFG_WARPSCALE"] == nil {
            let wscale = loadGovernor.warpScale
            if abs(RIFEEngine.warpScale - wscale) > 0.001 {
                RIFEEngine.warpScale = wscale
                DiagnosticLog.shared.log("[GOV] RIFE warpScale → \(String(format: "%.2f", wscale))")
            }
        }
        // ①-b RIFE flow(predict) 상한 — RIFE의 **진짜** 중간 강등 다이얼. 단계 계측으로 확정:
        //    4K work의 지배 비용은 predict(288p=7.6ms)지 워프(1.5ms)가 아니다. flow 해상도를
        //    낮추면 predict가 해상도²로 준다(240=6.2/216=5.0/180=3.1ms). 사다리는 이 상한 아래에서
        //    자율 동작(이중 조작 아님 — 거버너는 상한만 내리고 사다리가 그 안에서 움직인다).
        let flowCap = loadGovernor.rifeFlowCap ?? 1_000_000
        if RIFEEngine.flowCapShort != flowCap {
            RIFEEngine.flowCapShort = flowCap
            DiagnosticLog.shared.log("[GOV] RIFE flowCap → \(flowCap >= 1_000_000 ? "none" : "\(flowCap)p")")
        }
        // ③ 보간 바이패스 — 원본 패스스루. 단, **RIFE는 바이패스하지 않는다**: 180p flow로
        //    predict를 3.1ms까지 낮춰 보간을 유지한다(원본 패스스루보다 항상 낫다 — 사용자 피드백
        //    "원본이 더 나을 정도"의 정면 해소). sub-model 단이 없는 MetalFlow/AppleFI만 바이패스.
        //    엔진은 살려둬 복귀가 즉시 되게 한다(엔진 내리면 configurePairEngine 수백 ms + 재락 ~1s).
        let rifeKeepsInterp = selectedRenderMode == .rife && RIFEEngine.modelAvailable(short: 180)
        // RIFE가 bypass에서도 보간을 유지하려면 배율/t 상한도 bypass값(1/0 = 보간 없음)이 아니라
        // heavy값(2/1)이어야 한다 — 안 그러면 t가 0개라 encodePair가 빈 tValues로 nil을 뱉어
        // (engFail) 보간이 실질적으로 꺼진다. flowCap이 이미 부하를 흡수하므로 heavy로 충분.
        let bypassButRife = rifeKeepsInterp && loadGovernor.level == .bypass
        // ② 보간 배율 상한 — 렌더 스레드 미러에 반영 (t 생성 단계에서 소비)
        let effMultCap = bypassButRife ? 2 : loadGovernor.multiplierCap
        let mult = min(frameMultiplier, effMultCap ?? frameMultiplier)
        if mirrorFrameMultiplier != mult { mirrorFrameMultiplier = mult }
        let wantInterp = isInterpolationEnabled && !(loadGovernor.bypassInterpolation && !rifeKeepsInterp)
        if mirrorInterpolationEnabled != wantInterp { mirrorInterpolationEnabled = wantInterp }
        // ④ 업스케일 체인 — MetalFX(GPU)만 끄고 ANE 2x(GPU-free, 1.68ms 실측)는 유지
        overlayManager?.setUpscaleAllowsMetalFX(loadGovernor.allowsMetalFX)
        // ⑤ 갭 확장 / t 개수 상한 — 렌더 스레드 미러로 전달.
        // 갭 확장은 거버너가 정상(L0)이어도, 소스가 버스트(srcInt 지터 큼)면 억제한다:
        // 버스트로 벌어진 갭은 여유가 아니라 몰려온 프레임 사이의 빈 구간이라, 거기 t를 채우면
        // 절반이 폐기되며 저더(σ↑)만 만든다(4K 버스트 실측). 균일 소스는 지터가 작아 영향 없음.
        let srcSpreadMs = (diagSrcIntMax > 0 && diagSrcIntMin.isFinite && diagSrcIntMin >= 0)
            ? (diagSrcIntMax - diagSrcIntMin) * 1000 : 0
        gapExpansionAllowed = loadGovernor.allowsGapExpansion && srcSpreadMs < 20
        tCountCap = bypassButRife ? 1 : loadGovernor.tCountCap
    }

    /// 거버너 미러 (렌더 스레드에서 읽음) — 갭 확장 허용 / t 개수 상한
    @ObservationIgnored nonisolated(unsafe) private var gapExpansionAllowed = true
    @ObservationIgnored nonisolated(unsafe) private var tCountCap: Int?

    /// 사용자가 고른 flow 해상도 (거버너 상한 계산의 기준값)
    @ObservationIgnored private var userFlowBase: Double = MetalFlowEngine.flowBaseLongSide

    /// 거버너에 이번 창의 부하 신호를 먹인다 — 전부 기존 계측이라 추가 비용 없음.
    /// 거버너는 화질 다이얼만 만지고 페이싱(extraLatencySlots)은 건드리지 않는다.
    nonisolated private func feedLoadGovernor() {
        let refresh = max(mirrorRefreshRate, 60)
        let interval = min(max(sourceIntervalEMA > 0 ? sourceIntervalEMA : 1.0 / 60.0, 1.0 / 120.0), 1.0 / 24.0)
        // compute 과부하 신호 — RIFE일 때만 유효(비-RIFE는 0 → 거버너가 기존 presentRatio 판정 사용).
        let rife = pairEngine as? RIFEEngine
        let signals = LoadGovernor.Signals(
            workP90Ms: paceWorkP90,
            workAvgMs: paceWorkAvg,
            sourceIntervalMs: interval * 1000.0,
            tickHz: lastTickHz,
            refreshHz: refresh,
            missCount: paceMissCount,
            presentRatio: pacePresentRatio,
            predictP90Ms: rife?.recentPredictP90Ms ?? 0,
            slotExhaustFrac: rife?.recentExhaustFrac ?? 0)
        // 자동 flow 스케일러 입력 — 달성도는 "틱이 주사율을 내는가"와 "낸 프레임을 지키는가" 중
        // 나쁜 쪽(둘 다 목표 프레임 미달의 증상). 엔진 GPU 비중이 근거로 함께 들어간다.
        let tickRatio = refresh > 0 && lastTickHz > 1 ? min(1.0, lastTickHz / refresh) : 1.0
        // 합성 규칙과 그 근거는 AutoFlowScaler.combinedAchieved 참조 (단위 테스트로 고정돼 있다).
        let achieved = AutoFlowScaler.combinedAchieved(tickRatio: tickRatio, keepRatio: pacePresentRatio)
        let engineMs = engineGpuMsEMA
        let budgetMs = interval * 1000.0
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.loadGovernor.update(signals)
            self.autoFlowScaler.update(achievedRatio: achieved, engineMs: engineMs, budgetMs: budgetMs)
            self.applyGovernorDials()
        }
    }

    /// GPU 코어 수 추정 — 정확한 API가 없어 이름으로 등급만 가른다 (시딩용, 오차 허용).
    /// 틀려도 피해는 "처음 몇 초 보수적 화질"이고 실측 신호가 곧 정정한다.
    nonisolated static func gpuCoreCountEstimate(_ device: any MTLDevice) -> Int {
        let n = device.name
        if n.contains("Ultra") { return 48 }
        if n.contains("Max")   { return 32 }
        if n.contains("Pro")   { return 16 }
        return 10   // 베이스 M 시리즈 (M1 8, M4 10 — 저사양 판정 경계에 걸치게)
    }

    /// 소스가 Neural(RIFE) 예산을 넘길 만큼 큰지 — 4K급(≈7MP 이상). UI 경고용.
    /// 실측(M4): 3840×2160에서 Neural 85ms/프레임(소스 60fps→24fps로 붕괴), MetalFlow 35ms, AppleFI 10ms.
    var sourceIsLargeForNeural: Bool {
        let w = stablePoolWidth, h = stablePoolHeight
        guard w > 0, h > 0 else { return false }
        return w * h >= 7_000_000
    }
    @ObservationIgnored private var fsSample = false
    /// 현재 전체화면 상태가 유지되기 시작한 시각 (0=미측정). 시간 기반 디바운스용.
    @ObservationIgnored private var fsStableSince: CFAbsoluteTime = 0
    /// 영역 캡처: 소스 창 좌상단 기준 크롭 사각형(pt). nil이면 창 전체.
    /// 설정되면 resize-reconfigure를 끈다(영역이 창-리사이즈 재구성으로 날아가지 않게).
    @ObservationIgnored nonisolated(unsafe) var captureRegion: CGRect?
    private var overlayManager: OverlayManager?
    private let performanceMonitor = PerformanceMonitor()
    @ObservationIgnored nonisolated(unsafe) private var pairEngine: (any PairInterpolationEngine)?
    /// 시간축 정지-UI 검출기 (채팅/HUD 프리즈) — 렌더 스레드 소유(ingest에서 갱신·조회).
    @ObservationIgnored nonisolated(unsafe) private var uiDetector: UIStaticDetector?
    /// Vision 텍스트 검출 (흐린 채팅 보완) — ~2초 주기, 백그라운드 큐. 스냅샷 shared 텍스처 재사용.
    @ObservationIgnored nonisolated(unsafe) private var visionSnapshotTex: (any MTLTexture)?
    /// Vision 스냅샷 축소용 스케일러 — 매 호출 생성은 낭비라 캐시한다.
    @ObservationIgnored nonisolated(unsafe) private var visionScaler: MPSImageBilinearScale?
    @ObservationIgnored nonisolated(unsafe) private var visionInFlight = false
    @ObservationIgnored nonisolated(unsafe) private var lastVisionAt: CFTimeInterval = 0
    private let visionQueue = DispatchQueue(label: "macfg.vision", qos: .utility)
    private let mailbox = RenderMailbox()
    private let logger = Logger(subsystem: "com.macfg", category: "AppState")

    // MARK: - A2 렌더 스레드 (CAMetalDisplayLink)
    // 틱은 전용 렌더 스레드에서 실행 — nonisolated(unsafe) 상태들은 캡처 활성 중 렌더 스레드가
    // 소유하고, 메인은 정지 후(detach 동기 보장) 또는 명시된 락/미러를 통해서만 접근한다.
    private let renderDriver = RenderDriver()
    @ObservationIgnored nonisolated(unsafe) private var renderSurface: RenderSurface?
    /// UI 설정의 렌더용 미러 (메인이 쓰고 렌더가 읽는 racy-but-benign 단순값)
    @ObservationIgnored nonisolated(unsafe) private var mirrorInterpolationEnabled = true
    /// isCapturing의 렌더 스레드용 미러 — 캡처 콜백이 정지 직후에도 인제스트하지 않도록.
    /// (isCapturing은 MainActor라 렌더/캡처 스레드에서 못 읽는다)
    @ObservationIgnored nonisolated(unsafe) private var isCapturingMirror = false
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
        self.copyQueue = device.makeCommandQueue()
        self.overlayManager = OverlayManager(device: device)
        self.interpolationEngine = selectedRenderMode.displayName
        // 정지-UI 프리즈 토글 (A/B·회귀 확인용). 기본 on.
        if ProcessInfo.processInfo.environment["MACFG_NO_UISTATIC"] != nil { UIStaticDetector.enabled = false }
        // 새 릴리즈 확인 (기본 on, 6시간 주기). 설치는 하지 않고 알리기만 한다.
        self.updateChecker.startPeriodicCheck()
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
    func updateDevLogging() {
        DiagnosticLog.shared.setEnabled(devLoggingEnabled)
        stageDbg = devLoggingEnabled || ProcessInfo.processInfo.environment["MACFG_STAGEDBG"] == "1"
        registerHotKeys()   // 개발 덤프 단축키(⌃⌥⌘D/O)를 devLogging 상태에 맞춰 등록/해제
    }

    /// Dock 표시 여부 — on이면 메뉴바 전용(.accessory, Dock/⌘Tab 제거), off면 일반(.regular).
    func updateMenuBarOnly() {
        NSApplication.shared.setActivationPolicy(menuBarOnly ? .accessory : .regular)
        if !menuBarOnly { NSApplication.shared.activate(ignoringOtherApps: true) }
        UserDefaults.standard.set(menuBarOnly, forKey: "s.menubaronly")
    }

    /// 설정을 별도 창으로 분리 (팝오버가 딴 곳 클릭 시 닫히는 게 불편한 경우). 한 번 만들어 재사용.
    @ObservationIgnored private var detachedWindow: NSWindow?
    func openSettingsWindow() {
        if let w = detachedWindow {
            // 최소화 복원이 먼저다 — makeKeyAndOrderFront는 축소된 창을 되살리지 않는다.
            // Dock 아이콘이 없는 앱이라 ⌘M 한 번이면 이 창도, 채택 실패 시 자동으로 띄우는
            // 폴백 경로도 전부 조용한 무동작이 된다(= 탈출구 상실).
            if w.isMiniaturized { w.deminiaturize(nil) }
            w.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true); return
        }
        let host = NSHostingController(rootView: WindowPickerView(appState: self))
        let w = NSWindow(contentViewController: host)
        w.title = "MacFG"
        w.styleMask = [.titled, .closable, .miniaturizable]
        w.isReleasedWhenClosed = false          // 닫아도 재사용 (accessory라 앱은 안 꺼짐)
        w.center()
        detachedWindow = w
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
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
        d.set(coverKeepVisible, forKey: "s.coverkeep")
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
        if d.object(forKey: "s.coverkeep") != nil { coverKeepVisible = d.bool(forKey: "s.coverkeep") }
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
        overlayManager?.ensureViewerOnScreen()   // 디스플레이 분리/해상도 축소 시 화면 밖 잔류 방지
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
            userFlowBase = base   // 거버너 기준값도 갱신 — 안 하면 기본값으로 되돌림
            autoFlowScaler.manualOverride = true   // 명시 지정 = 자동 조절 중지 (측정/실험용)
            DiagnosticLog.shared.log("[AUTO] flowBaseLongSide=\(Int(base)) (자동 스케일 OFF)")
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
                // 자체검증: MACFG_AUTODUMP 설정 시 캡처 안정화 후 프레임 덤프 자동 무장
                if ProcessInfo.processInfo.environment["MACFG_AUTODUMP"] != nil {
                    Task { @MainActor in
                        try? await Task.sleep(for: .seconds(4))
                        self.startFrameDump()
                    }
                }
                if ProcessInfo.processInfo.environment["MACFG_AUTOINFO"] != nil {
                    Task { @MainActor in try? await Task.sleep(for: .seconds(5)); self.infoOverlayVisible = true; self.refreshInfoOverlay() }
                }
                if let od = ProcessInfo.processInfo.environment["MACFG_AUTOOUTDUMP"] {
                    // 값이 숫자면 지연(초) — 사다리 승격 전환창(~1s 소스-온리)을 피해
                    // 정착 후를 측정할 때 사용 (예: MACFG_AUTOOUTDUMP=25). 그 외엔 6초.
                    let delay = Double(od) ?? 6
                    Task { @MainActor in
                        try? await Task.sleep(for: .seconds(delay > 0 ? delay : 6))
                        self.startOutputDump()
                    }
                }
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        DiagnosticLog.shared.log("[AUTO] target window not found: \(titleSub)")
    }

    // MARK: - Capture Control

    /// startCapture 진행 중 플래그 — isCapturing만으로는 재진입을 못 막는다(아래 주석 참조).
    @ObservationIgnored private var isStartingCapture = false

    func startCapture() async {
        guard !isCapturing else { return }
        // **재진입 가드.** isCapturing은 이 함수 *끝*에서야 true가 되는데, 그 전에
        // `await captureManager.startCapture`와 `await configurePairEngine()`(RIFE면 수백 ms)가
        // 있다. 그래서 단축키를 빠르게 두 번 누르면 두 Task가 모두 위 guard를 통과한다. 결과:
        //  · 같은 pendingSlots에 프레임을 밀어넣는 SCStream이 두 개
        //  · 아무도 닫을 수 없는 고아 shielding 레벨 오버레이
        //  · statsTimer/trackingTimer가 invalidate 없이 덮어써져, 먼저 만든 타이머가
        //    RunLoop.main에 붙들린 채 **정지 후에도 15~30Hz로 메인 스레드를 계속 두드린다**
        //    — 표시 간격 σ가 전부인 앱에서 이건 그냥 상시 지터원이다.
        // defer는 반드시 guard **뒤에** 둔다. 앞에 두면 세 번째 탭에서 구멍이 다시 열린다.
        guard !isStartingCapture else {
            DiagnosticLog.shared.log("[CAPTURE] 시작이 이미 진행 중 — 중복 요청 무시")
            return
        }
        isStartingCapture = true
        defer { isStartingCapture = false }

        guard let windowID = selectedWindowID else {
            logger.warning("No window selected")
            return
        }

        // O1-1 콜백 구동 인제스트: 프레임이 도착하면 렌더 틱을 기다리지 말고 즉시 렌더 스레드
        // 런루프에 인제스트를 태운다. 틱을 기다리면 평균 ½틱(~4.2ms)이 그냥 버려지고, 그 지연이
        // work·e2e에 그대로 실릴 뿐 아니라 paceWorkP90을 올려 적응지연 하한까지 밀어올린다.
        // performAsync는 틱과 같은 런루프라 절대 겹치지 않음 → 무락 전제 보존.
        // MACFG_CBINGEST=0이면 기존(틱 drain 전용) 경로로 폴백.
        if ProcessInfo.processInfo.environment["MACFG_CBINGEST"] != "0" {
            captureManager.onFrameAvailable = { [weak self] in
                guard let self else { return }
                self.renderDriver.performAsync { [weak self] in
                    guard let self, self.isCapturingMirror else { return }
                    self.drainAndIngest(maxCount: 4)
                }
            }
        } else {
            captureManager.onFrameAvailable = nil
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
            // 최초 시작: 엔진 준비(수백 ms) 동안 SCK가 쌓은 백로그를 폐기하고 스케줄러를
            // 워밍업 포함 깨끗한 상태로 리셋 — 안 하면 첫 틱이 오래된 프레임을 버스트로
            // 먹고 그 miss로 적응 지연이 +4까지 불필요하게 램프 (리뷰 지적, 로그 확인).
            _ = captureManager.drainFrames()
            // **resetScheduler는 렌더 스레드에서 돌려야 한다 — 메인에서 직접 부르면 크래시난다.**
            // 이 함수는 timeline · inFlightTextures · stablePool · prevStable ·
            // lastPresentedTexture를 통째로 비운다. 전부 렌더 스레드가 매 프레임 읽고 쓰는
            // 구조다. 캡처 시작은 마우스 이탈 시 소스 재활성(OverlayWindow의 activate) →
            // 전체화면 재타깃 → 재시작 경로로도 들어오므로, 렌더 스레드가 살아 있는 채로
            // 여기 도달할 수 있다. 그때 배열 버퍼가 통째로 교체되면 순회 중이던 렌더 스레드가
            // 죽은 참조를 retain해 **SIGTRAP**으로 죽는다.
            // 실측 크래시 2건(2026-08-01, 마우스 반복 진입/이탈 중):
            //   ① MacFG.Render: acquireStableTexture → Sequence.first(where:) →
            //      Array.subscript → swift_unknownObjectRetain
            //   ② SCK 전달 큐: outlined consume of FrameSlot? → swift_unknownObjectRelease
            //      (①이 힙을 깨뜨린 뒤 엉뚱한 곳에서 표출된 것)
            // 코드베이스는 이미 올바른 규약을 갖고 있다 — 다른 두 호출부는 renderDriver.perform
            // 안에서 부르거나(:1011) 렌더 틱 자신이 부른다(pendingShowReset 경로). 여기만 예외였다.
            // 드라이버가 안 돌 때는 이 구조를 만지는 스레드가 우리뿐이라 직접 호출이 안전하다
            // (perform은 런루프가 죽어 있으면 조용한 no-op이라 리셋이 통째로 유실된다).
            if renderDriver.isRunning {
                renderDriver.perform { [weak self] in self?.resetScheduler() }
            } else {
                resetScheduler()
            }
            // 거버너/스케일러 시딩은 소스 해상도를 아는 첫 프레임 시점(acquireStableTexture)에서
            // 한다 — 여기선 stablePool이 아직 없어 크기가 0이라 "4K 무거움" 판정이 불가능하다.
            loadGovernor.reset()
            autoFlowScaler.softReset()
            engineGpuMsEMA = 0
            seedPending = true
            applyGovernorDials()
            pendingShowReset = false
            attachRenderDriver()

            // 재진입 가드가 있어도 남은 타이머는 확실히 끊는다 — RunLoop가 강참조로 붙들어
            // 덮어쓰기만으론 죽지 않는다(정지 후에도 계속 도는 유령 타이머의 원인).
            statsTimer?.invalidate()
            trackingTimer?.invalidate()
            statsTimer = addCommonTimer(0.5) { [weak self] _ in
                Task { @MainActor in
                    self?.updateStats()
                    self?.detectFullscreenRetarget()
                }
            }

            // 창 추적은 30Hz면 충분 — 틱(120Hz)마다 CGWindowList를 부르면 호출당 0.5-2ms로
            // vsync 틱을 놓쳐 출력 fps 천장이 ~110으로 내려앉는다 (실측).
            // 뷰어 배치도 15Hz — 상대커서 매핑이 sourceFrameNS를 쓰므로, 드래그로 소스 창이
            // 움직였을 때 다음 조작 좌표가 어긋나지 않게 신선도가 필요 (2Hz는 0.5s 지연으로
            // 매핑이 헛돌았음). 렌더는 전용 스레드라 메인 CGWindowList 15Hz는 틱에 무해.
            let trackHz: Double = selectedOverlayPlacement == .coverSource ? 30.0 : 15.0
            trackingTimer = makeTrackingTimer(hz: trackHz)

            // 자동 숨김 기준용 소스 PID + 초기 표시 상태
            sourceOwnerPID = ownerPID(of: windowID)
        originalCaptureWindowID = windowID
        currentTargetWindowID = windowID
            overlayManager?.sourcePID = sourceOwnerPID   // 뷰어 마우스 역매핑 대상
            overlayUserHidden = false
            overlayHiddenState = false
            isCapturing = true
            isCapturingMirror = true
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
            // 롤백 — 실패 지점이 어디든 이미 열린 자원을 되감는다. 안 하면 isCapturing=false인데
            // SCK 스트림·오버레이·타이머는 계속 도는 유령 상태가 되고, 재시도 시 옛 스트림과
            // 새 스트림이 이중으로 프레임을 공급한다 (리뷰 확정).
            logger.error("Failed to start capture: \(error)")
            DiagnosticLog.shared.log("[CAPTURE] start failed → rollback: \(error)")
            configureEpoch += 1              // 진행 중 엔진 준비 무효화
            renderDriver.detach()
            renderSurface = nil
            await captureManager.stopCapture()
            overlayManager?.stop()
            pairEngine?.shutdown()
            pairEngine = nil
            interpolationEngine = "None"
            statsTimer?.invalidate();          statsTimer = nil
            trackingTimer?.invalidate();       trackingTimer = nil
            drawableReattachTimer?.invalidate(); drawableReattachTimer = nil
            isCapturing = false
            isCapturingMirror = false
            captureMethod = "None"
            trackingMethod = "None"
            sourceOwnerPID = 0
            return
        }
    }

    /// 창 추적 타이머 (재)생성 — 주기가 배치에 의존하므로 캡처 중 배치가 바뀌면 다시 만들어야
    /// 한다. 안 하면 시작 시점 배치의 주기(cover 30Hz / 뷰어 15Hz)에 영구 고정 (리뷰 확정).
    private func restartTrackingTimer() {
        guard isCapturing else { return }
        let trackHz: Double = selectedOverlayPlacement == .coverSource ? 30.0 : 15.0
        trackingTimer?.invalidate()
        trackingTimer = makeTrackingTimer(hz: trackHz)
    }

    private func makeTrackingTimer(hz: Double) -> Timer {
        addCommonTimer(1.0 / hz) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.overlayManager?.updateTracking()
                // 창 종료/리사이즈 감지 (렌더 틱에서 이관 — overlayManager는 MainActor)
                guard self.isCapturing else { return }
                // 최소화/Space 이동 감지 — 워크스페이스 알림만으로는 안 잡혀 좀비 오버레이가 남는다
                self.refreshOverlayVisibility()
                if self.hasReceivedFirstFrame, (self.overlayManager?.trackingFailureCount ?? 0) > 30 {
                    DiagnosticLog.shared.log("[CAPTURE] target window gone (tracking) → stop")
                    await self.stopCapture()
                    return
                }
                // 전체화면 감지는 매 틱 — lastSourceFrame 재사용이라 추가 비용 0이고,
                // 0.5s 게이트 안에 두면 전환이 1s 넘게 늦어 눈에 띄게 버벅인다(실측).
                // 리사이즈 검사보다 **먼저** 돌려서, 전체화면행이면 아래 창-캡처 리사이즈를
                // 건너뛴다 — 어차피 디스플레이 캡처로 갈 건데 창 캡처를 4K로 재설정했다 버리는
                // 중간 단계가 전환 버벅임의 주범이었다.
                self.detectFullscreenAutoViewer()

                let nowT = CFAbsoluteTimeGetCurrent()
                if nowT - self.lastResizeCheck >= 0.5 {
                    self.lastResizeCheck = nowT
                    let headingFullscreen = self.overlayManager?.sourceIsFullscreen ?? false
                    if !headingFullscreen,
                       !self.isRestartingCapture, !self.retargetInFlight, self.captureRegion == nil, self.stablePoolWidth > 0,
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
                    } else if headingFullscreen {
                        self.resizeMismatchCount = 0
                    }
                }
            }
        }
    }

    func stopCapture() async {
        // 플래그를 await 이전에 먼저 내림 — 아래 stopCapture await 중 화면 파라미터 변경이
        // handleScreenParametersChange(399행, isCapturing 가드)로 detach된 링크를 재부착해
        // 좀비 렌더 루프를 만들던 것 차단 (리뷰 확정). 진행 중 configurePairEngine도 무효화.
        isCapturing = false
        isCapturingMirror = false
        captureManager.onFrameAvailable = nil   // 콜백 인제스트 즉시 차단
        configureEpoch += 1
        // 자동 전체화면 뷰어는 이번 캡처 한정 상태 — 원복 안 하면 다음 캡처가 전체화면 뷰어로
        // 잘못 시작한다 (리뷰 확정). 사용자 설정(upscaleMode)에서 배치를 다시 유도.
        if autoFsViewer {
            autoFsViewer = false
            selectedOverlayPlacement = (upscaleMode == .off) ? .coverSource : .viewerWindow
        }
        fsSample = false
        fsStableSince = 0
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
            DiagnosticLog.shared.log("[CAPTURE] seamless resize → \(width)x\(height) (pool=\(stablePoolWidth)x\(stablePoolHeight) disp=\(captureManager.isDisplayCapture))")
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

    /// Vision 텍스트 검출 스케줄 (렌더 스레드, ingest 내) — ~2초 주기로 소스 스냅샷을 shared
    /// 텍스처에 blit하고, cb 완료 후 백그라운드 큐에서 VNRecognizeTextRequest(.fast) 실행 →
    /// 박스를 uiDetector에 제출. 일관성 신호가 놓치는 흐린 채팅 라인 보완 (오프라인 GT 검증).
    nonisolated private func scheduleVisionTextDetection(source: any MTLTexture, cb: any MTLCommandBuffer) {
        guard UIStaticDetector.enabled, let det = uiDetector, !visionInFlight else { return }
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastVisionAt > 2.0 else { return }
        lastVisionAt = now
        // ── 스냅샷은 **축소해서** 뜬다 (긴 변 1280 상한).
        // 예전엔 소스 해상도 그대로 복사했는데, 4K에선 GPU blit 33MB + CPU getBytes 33MB가
        // 2초마다 걸렸다. 그 메모리 대역폭과 뒤이은 Vision 추론(ANE)이 동시에 도는 4K 워프와
        // 경합해, [SPIKE] 실측에서 cb2gpu가 6.1 → 18~45ms로 튀고 ANE 대기도 11.6 → 25~29ms로
        // 동반 상승했다. 그 외란이 RIFE 사다리 승격 여유(0.9ms)를 삼켜 발진의 방아쇠가 됐다.
        // Vision의 boundingBox는 **정규화 좌표(0~1)** 라 입력 해상도를 낮춰도 결과가 그대로다.
        // 4K → 1280 기준이면 전송량이 33MB → 3.7MB로 약 9배 준다.
        let longSide = max(source.width, source.height)
        let vScale = longSide > 1280 ? 1280.0 / Double(longSide) : 1.0
        let dw = max(16, Int(Double(source.width) * vScale) & ~1)
        let dh = max(16, Int(Double(source.height) * vScale) & ~1)
        if visionSnapshotTex == nil || visionSnapshotTex!.width != dw || visionSnapshotTex!.height != dh {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: dw, height: dh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .shared
            visionSnapshotTex = device.makeTexture(descriptor: d)
        }
        guard let snap = visionSnapshotTex else { return }
        if dw == source.width && dh == source.height {
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            blit.copy(from: source, sourceSlice: 0, sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
                      to: snap, destinationSlice: 0, destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        } else {
            if visionScaler == nil { visionScaler = MPSImageBilinearScale(device: device) }
            guard let scaler = visionScaler else { return }
            var xf = MPSScaleTransform(scaleX: Double(dw) / Double(source.width),
                                       scaleY: Double(dh) / Double(source.height),
                                       translateX: 0, translateY: 0)
            withUnsafePointer(to: &xf) { scaler.scaleTransform = $0 }
            scaler.encode(commandBuffer: cb, sourceTexture: source, destinationTexture: snap)
        }
        visionInFlight = true
        let queue = visionQueue
        nonisolated(unsafe) let selfRef = self   // visionInFlight 플래그 리셋용 (unsafe 필드)
        cb.addCompletedHandler { [weak det] _ in
            queue.async {
                defer { selfRef.visionInFlight = false }
                guard let det else { return }
                let w = snap.width, h = snap.height
                var bytes = [UInt8](repeating: 0, count: w * h * 4)
                snap.getBytes(&bytes, bytesPerRow: w * 4,
                              from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                              | CGBitmapInfo.byteOrder32Little.rawValue),
                      let cg = ctx.makeImage() else { return }
                let req = VNRecognizeTextRequest()
                req.recognitionLevel = .fast
                req.usesLanguageCorrection = false
                try? VNImageRequestHandler(cgImage: cg).perform([req])
                // Vision은 좌하 원점 → 좌상 원점 정규화로 변환
                let rects: [CGRect] = (req.results ?? []).map { o in
                    let b = o.boundingBox
                    return CGRect(x: b.origin.x, y: 1 - b.origin.y - b.height, width: b.width, height: b.height)
                }
                det.submitTextBoxes(rects)
                DiagnosticLog.shared.log("[UISTATIC] vision \(rects.count)개 텍스트 박스")
            }
        }
    }

    nonisolated private func resetScheduler() {
        uiDetector?.reset()   // 불연속(재시작/리사이즈) — 정지-UI 누적도 리셋
        timeline = []
        inFlightTextures = [:]
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
        // Vision 스냅샷은 소스 해상도 shared 텍스처(4K ≈ 33MB) — 캡처가 끝나도 붙들고 있으면
        // 정지 상태에서 계속 상주한다 (리뷰 확정). 리셋 시 놓아주고 다음 캡처에서 재생성.
        visionSnapshotTex = nil
        // statsLock 규약(224행): presentedTimes는 렌더가 쓰고 메인(updateStats)이 읽음.
        // 여기만 락 없이 재할당하면 메인의 스냅샷 read와 CoW 버퍼 해제가 겹쳐 UAF 가능 (리뷰 확정).
        statsLock.lock()
        presentedTimes = []
        latencySamplesMs = []
        statsLock.unlock()
        // 적응 지연은 소스/세션 특성이므로 새 캡처에서 다시 학습
        extraLatencySlots = 0
        paceMissCount = 0
        paceCleanWindows = 0
        lastPaceAdjustTick = 0
        paceWarmupUntilTick = diagTick + 720   // ~6s 과도기 무시 (케이던스 락)
        paceWorkP90 = 0   // 이전(무거운) 세션의 work p90 하한이 새 캡처의 extra를 재부풀리지 않게 (감사 확정)
        pendingIngest = []
        inFlightPresents.withLock { $0 = 0 }
        _ = mailbox.drain()
    }

    /// 리사이즈 전용 경량 리셋 — 크기 의존 상태(타임라인/풀/이전 프레임)만 비우고
    /// 케이던스(스냅 링/EMA/타임스탬프)는 유지한다. 전체 리셋의 ~16프레임 재락을 회피.
    nonisolated private func softResetForResize() {
        timeline = []
        inFlightTextures = [:]
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
    /// 표시 파이프라인에 떠 있는 스테일 텍스처 — **참조를 붙잡는다.**
    ///
    /// 예전엔 `Set<ObjectIdentifier>`였다. 식별자만 들고 있으면 텍스처를 살려두지 못한다:
    /// 이 텍스처들을 붙잡는 건 stablePool뿐인데, 소스 크기가 바뀌면 acquireStableTexture가
    /// `stablePool = []`로 8장을 통째로 놓는다. 그러면 timeline/prevStable에 걸리지 않은
    /// **인플라이트 텍스처가 GPU가 아직 읽는 중에 해제된다** — GPU 측 use-after-free다.
    /// 힙이 깨지면 트랩은 엉뚱한 곳에서 난다. 실측 크래시 4건이 전부
    /// `acquireStableTexture → first(where:) → swift_unknownObjectRetain`인 이유가 이것이다
    /// (방금 깨진 그 풀을 바로 다음에 순회하니까). 마우스 진입/이탈이 트리거인 것도
    /// 소스 재활성 → 창 레이아웃 변경 → 캡처 크기 변경 → 풀 재생성 경로로 설명된다.
    /// 죽은 주소의 ObjectIdentifier가 남아 나중 텍스처와 충돌하던 문제도 함께 사라진다.
    @ObservationIgnored nonisolated(unsafe) private var inFlightTextures: [ObjectIdentifier: any MTLTexture] = [:]
    // stable 준비 이벤트 — blit(cb1) 완료를 GPU 이벤트로 알림. RIFE pack(별도 큐)이 이걸
    // 기다려 '아직 안 쓰인 stableB를 읽는' 크로스큐 레이스를 차단 (인앱 flow 폭주의 원인).
    @ObservationIgnored nonisolated(unsafe) private var stableReadyEvent: (any MTLSharedEvent)?
    @ObservationIgnored nonisolated(unsafe) private var stableReadyCounter: UInt64 = 0
    /// timestamp = 스냅된 콘텐츠 시각 (타임라인/보간용), rawTimestamp = SCK 원본 시각 (연속성 검사용)
    @ObservationIgnored nonisolated(unsafe) private var prevStable: (texture: any MTLTexture, timestamp: CFTimeInterval, rawTimestamp: CFTimeInterval)?

    // 실프레임 덤프 (⌃⌥⌘D) — 연속 고유 소스 프레임을 PNG로 저장. 오프라인 보간 화질 측정용
    // (실콘텐츠 삼중항 A/GT/B → InterpBench --triplets). 합성-실전 갭 해소 도구.
    @ObservationIgnored nonisolated(unsafe) private var frameDumpRemaining = 0
    @ObservationIgnored nonisolated(unsafe) private var frameDumpIndex = 0
    @ObservationIgnored nonisolated(unsafe) private var frameDumpDir: URL?
    // 표시 프레임 덤프 (⌃⌥⌘O) — 실제 표시된 시퀀스(S/I 순서·시각 포함)를 저장, 변위 분석으로
    // '눈이 보는 부들거림'을 수치화 (프레임별 모션 변위의 균일성)
    @ObservationIgnored nonisolated(unsafe) private var outDumpRemaining = 0
    @ObservationIgnored nonisolated(unsafe) private var outDumpIndex = 0
    @ObservationIgnored nonisolated(unsafe) private var outDumpDir: URL?
    private let frameDumpFileQueue = DispatchQueue(label: "com.macfg.framedump", qos: .utility)
    @ObservationIgnored nonisolated(unsafe) private var lastAcceptedTimestamp: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var lastAcceptedFingerprint: UInt64 = 0
    /// 케이던스 스냅 상태 — 캡처 ts는 디스플레이 vsync 그리드(144Hz 등)에 양자화되고
    /// 실콘텐츠(브라우저)는 PTS가 [5~99ms]로 튄다. 중앙값 간격 + 앵커 그리드로 락을 유지하고
    /// 이탈 3연속일 때만 재동기 — 즉발 재동기는 그리드 정렬 위상을 흔들어 σ를 키운다 (실측).
    @ObservationIgnored nonisolated(unsafe) private var snapTsRing: [CFTimeInterval] = []
    @ObservationIgnored nonisolated(unsafe) private var snapMissStreak = 0
    @ObservationIgnored nonisolated(unsafe) private var snappedLastTimestamp: CFTimeInterval = 0
    /// 격자 원점 — snappedLastTimestamp(반환값, 단조증가)와 분리. 이탈 프레임은 raw로
    /// 통과시키되 원점은 3연속 이탈(진짜 불연속)에만 이동 — 버스트 한두 방에 원점이
    /// 끌려가면 이후 모든 예측이 어긋나 srcInt가 60↔10ms로 뒤집힌다 (리뷰 지적, 실측 일치).
    @ObservationIgnored nonisolated(unsafe) private var snapAnchor: CFTimeInterval = 0
    /// interval 히스테리시스 — 락 대비 ±25% 넘는 추정치의 연속 지속 카운트
    @ObservationIgnored nonisolated(unsafe) private var snapIntervalDeviateStreak = 0
    @ObservationIgnored nonisolated(unsafe) private var diagResyncCount = 0
    /// 케이던스 격자 이탈 관측치 — B1(버스트 창) 착수 여부를 실측으로 결정하기 위한 것.
    /// 인제스트 소요 — 렌더 런루프를 점유해 링크 콜백을 버리게 만드는 후보 1순위.
    @ObservationIgnored nonisolated(unsafe) private var diagIngestSum = 0.0
    @ObservationIgnored nonisolated(unsafe) private var diagIngestSamples = 0
    @ObservationIgnored nonisolated(unsafe) private var diagIngestMax = 0.0
    @ObservationIgnored nonisolated(unsafe) private var diagIngestOver = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSnapMissCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSnapPullableCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSnapPullLagMax: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var lastPresentedTimestamp: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var lastPresentedTexture: (any MTLTexture)?
    /// 마지막 표시 텍스처의 세대 도장 — 강제 재표시(링크 재부착) 시 그 사이 덮였는지 검증
    @ObservationIgnored nonisolated(unsafe) private var lastPresentedStamp: UInt64 = 0
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
    /// 최근 진단 창의 실제 틱 레이트 — 거버너의 "렌더 루프 굶주림" 신호
    @ObservationIgnored nonisolated(unsafe) private var lastTickHz: Double = 0
    /// 이번 창의 work 평균 — 거버너 복귀 판정용 (감쇠 없는 즉응 신호)
    @ObservationIgnored nonisolated(unsafe) private var paceWorkAvg: Double = 0
    /// 이번 창의 present 처리량 / 이론 상한 (1.0 = 목표 달성). 거버너의 주신호 —
    /// work는 파이프라인 "지연"이라 15ms여도 240프레임을 다 뽑을 수 있어 과부하 판정에 부적합.
    @ObservationIgnored nonisolated(unsafe) private var pacePresentRatio: Double = 1.0
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
        // 거버너 급전은 **아래 조기 반환들보다 먼저** — 특히 바이패스(L3)에선
        // mirrorInterpolationEnabled가 false라 아래 게이트에서 반환되는데, 그러면 거버너가
        // 신호를 못 받아 영영 L3에 갇힌다(실측: 부하 종료 후 60초간 복귀 실패).
        feedLoadGovernor()
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
        // 적응 지연 상한 — 측정용 MACFG_MAXLAT로 낮춰 "지연↓ 드롭↑" 트레이드 확인 (기본 4).
        let maxSlots = ProcessInfo.processInfo.environment["MACFG_MAXLAT"].flatMap { Double($0) } ?? 4.0
        let requiredExtra = paceWorkP90 > 0
            ? min(maxSlots, max(0.0, ((paceWorkP90 + 2.0 - baseMs) / slotMs).rounded(.up)))
            : 0.0

        if extraLatencySlots < requiredExtra {
            extraLatencySlots = requiredExtra
            paceCleanWindows = 0
            DiagnosticLog.shared.log("[PACE] work p90=\(String(format: "%.0f", paceWorkP90))ms → 지연 하한 \(Int(requiredExtra))슬롯")
        }
        if paceMissCount >= 4 {
            paceCleanWindows = 0
            if extraLatencySlots < maxSlots {
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
    /// 타임라인 하드캡(12) 트림이 버린 표시 대기 프레임 수 — 배압 사문화 감시용 (감사 확정)
    @ObservationIgnored nonisolated(unsafe) private var diagCapDropCount = 0
    /// 세대 검증(ring lease)이 덮인 텍스처로 판정해 드롭한 보간 프레임 수 — burst 오버런 감시용
    @ObservationIgnored nonisolated(unsafe) private var diagLeaseDropCount = 0
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
        // vsync 스킵 감지 (link.timestamp 간격 > 1.4슬롯) — 콜백이 버려진 횟수.
        //
        // **"우리 핸들러 밖" ≠ "프로세스 밖".** 이 구분을 두 번 틀렸다.
        // cpu=/over=는 onDisplayLinkTick **안**만 잰다. 그런데 캡처 인제스트(drainAndIngest)는
        // performAsync로 **같은 런루프**에 올라가므로 이 카운터들에서 통째로 빠진다. 즉
        // "틱 CPU가 낮은데 갭이 있다"는 관측은 외란을 가리키는 게 아니라, **재지 않은 구간이
        // 있다**는 뜻이었다. 링크는 큐잉을 안 해서 늦은 콜백은 버려지고, 그래서 갭은 언제나
        // "CPU 비용 0"으로 보인다 — 자기 원인을 숨기는 지표다.
        // 실측(2026-08-01, 609창): r(gap, capIngest) = +0.729, 다른 후보는 전부 |r| < 0.4.
        // 아래 ing= 카운터가 그 구간을 직접 잰다.
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
            for id in released { inFlightTextures.removeValue(forKey: id) }
            _ = captureManager.drainFrames()   // 파이프 적체 방지 (텍스처는 풀로 회수)
            pendingIngest = []
            return
        }

        // 1) 완료된 GPU 작업 수거 → 타임라인 등재
        let (newEntries, released, presented) = mailbox.drain()
        for id in released { inFlightTextures.removeValue(forKey: id) }
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
            // **표시되지 못한 드로어블은 presentedTime이 0이다.** 그 0을 그대로 버퍼에 넣으면
            // 인덱스 0에 앉아 `last - first`가 머신 부팅 이후 시간(수십만 초)이 되고,
            // count·구간 길이 가드를 **둘 다 통과**한 채 fps가 0.0007 → 반올림 0으로 표시된다.
            // 실제로는 120fps로 잘 돌고 있는데 오버레이에 0 fps가 뜨던 잔여 원인(사용자 제보).
            // 버퍼 중간에 섞이면 반대로 count만 늘려 fps를 부풀린다. 아예 받지 않는다.
            guard record.presentedAt > 0 else { continue }
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
            diagCapDropCount += timeline.count - 12   // 표시 직전 프레임 강제 드롭 — [SCHED]에 노출
            timeline.removeFirst(min(timeline.count - 12, timeline.count))
        }
        // 세대 검증 (ring lease) — 표시 대기 중 후속 warp가 링 슬롯을 덮은 보간 엔트리 제거.
        // present는 아래에서, ingest(step3)는 그 뒤라, 이 프루닝은 '이전 틱까지의 오버런'을
        // 반영해 덮인 텍스처가 화면에 나가는 걸 막는다(잘못된 픽셀 대신 직전 프레임 유지).
        // 정상상태는 링 미포화라 no-op — burst(숨김해제/드랍복구)에서만 발동.
        if let eng = pairEngine {
            let before = timeline.count
            timeline.removeAll { $0.stamp != 0 && !eng.isFrameLive($0.stamp) }
            diagLeaseDropCount += before - timeline.count
        }
        let presentBefore = diagPresentCount
        presentDueEntry(targetTimestamp: targetTimestamp, drawable: drawable)
        // 링크 재부착 직후 강제 재present — 정적 콘텐츠는 새 프레임이 없어 presentDueEntry가
        // 아무것도 표시하지 않으므로, 재부착 전 그려둔 흐린(960×540 드로어블) 프레임이 남는다.
        // 이번 틱에 새 present가 없었으면 최신 텍스처를 새(큰) 드로어블에 다시 그려 교체한다.
        if forceRepresentTicks > 0 {
            forceRepresentTicks -= 1
            // 재표시 텍스처가 그 사이 링에서 덮였으면(보간 프레임) 스킵 — 화면은 이전 상태 유지
            let stillLive = lastPresentedStamp == 0 || (pairEngine?.isFrameLive(lastPresentedStamp) ?? false)
            if diagPresentCount == presentBefore, stillLive, let tex = lastPresentedTexture {
                let e = TimelineEntry(timestamp: lastPresentedTimestamp, texture: tex,
                                      isInterpolated: false, captureTimestamp: lastPresentedTimestamp)
                presentEntry(e, at: targetTimestamp, drawable: drawable)
            }
        }

        // 3) 캡처 프레임 drain → work 인코딩 (무거움 — 표시 이후로).
        // 버스트 캡: 숨김 해제/스트림 재개 직후 한 틱에 8장까지 몰리면 인코딩 CPU가
        // 8ms를 넘겨 다음 vsync 콜백을 삼킴 — 틱당 4장, 나머지는 다음 틱으로 이월
        // (타임스탬프 보존, 표시는 어차피 latencyOffset 뒤라 이월 8ms는 무해)
        // 콜백 구동 인제스트가 켜져 있으면 대개 여기 올 프레임이 이미 소진돼 no-op이다.
        // 그래도 남겨두는 이유: 런루프 미기동/블록 유실 시의 폴백이자, 콜백이 없는
        // IOSurface 폴백 소스의 유일한 경로.
        drainAndIngest(maxCount: 4)
        // (창 종료/리사이즈 감지는 트래킹 타이머(메인)로 이동 — overlayManager는 MainActor)

        adaptPacing()
        maybeLogDiagnostics()
    }

    /// 대기 중인 캡처 프레임을 인제스트한다. **렌더 스레드 전용** — 틱과 캡처 콜백(performAsync)
    /// 양쪽에서 불리지만 둘 다 같은 런루프라 직렬화되므로 무락 전제가 유지된다.
    ///
    /// maxCount: 한 번에 처리할 최대 장수. 버스트(숨김 해제/스트림 재개 직후 8장 몰림)에서
    /// 인코딩 CPU가 한 번에 8ms를 넘겨 다음 vsync 콜백을 삼키는 것을 막는다. 나머지는
    /// pendingIngest에 남아 다음 호출로 이월 (타임스탬프 보존, 표시는 latencyOffset 뒤라 무해).
    nonisolated private func drainAndIngest(maxCount: Int) {
        // **인제스트 시간을 잰다 — 지금까지 아무도 안 쟀던 구간.**
        // CAMetalDisplayLink는 렌더 스레드 런루프의 소스이고, 이 인제스트는 performAsync로
        // **같은 런루프**에 올라간다. 링크는 큐잉을 하지 않아서, 다음 vsync 전에 서비스되지
        // 못한 콜백은 **버려진다 — 핸들러 CPU 비용 0으로.** 그게 로그의 gap이다.
        // 그런데 cpu=/over=는 틱 핸들러 안만 재므로 이 구간이 통째로 빠져 있었고, 그 탓에
        // "우리는 한가한데 콜백이 사라진다 → 외란은 프로세스 밖"이라는 **틀린 결론**이 나왔다.
        // 실측 상관: r(gap, capIngest) = +0.729 (다른 후보는 전부 |r| < 0.4).
        let ingestT0 = CACurrentMediaTime()
        defer {
            let ms = (CACurrentMediaTime() - ingestT0) * 1000.0
            diagIngestSum += ms
            diagIngestSamples += 1
            if ms > diagIngestMax { diagIngestMax = ms }
            // 한 슬롯(vsync 간격)을 넘게 잡으면 그 사이 링크 콜백이 버려질 수 있다.
            if ms > 1000.0 / max(mirrorRefreshRate, 60) { diagIngestOver += 1 }
        }
        pendingIngest.append(contentsOf: captureManager.drainFrames().filter { $0.texture != nil })
        let n = min(pendingIngest.count, maxCount)
        guard n > 0 else { return }
        var depth = 0
        for slot in pendingIngest.prefix(n) {
            depth += 1
            ingest(slot)
        }
        pendingIngest.removeFirst(min(n, pendingIngest.count))
        diagDrainDepthSum += depth
        diagDrainSamples += 1
        if depth > diagDrainDepthMax { diagDrainDepthMax = depth }
        hasReceivedFirstFrame = true
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
        // 주의: 배출 판정을 "동시에 기한 지난 항목 수(candidates>=2)"로 바꿔봤으나 버스트에서
        // 정상적으로 몰려 도착한 프레임까지 적체로 오판해 매 틱 드롭 → staleDrop 39, σ 12.65로
        // 악화(실측, 기존 대비 4배). 정상 큐 깊이는 latencyOffset×주사율(lat=+4에서 ~7.4)이라
        // tl 6-7은 적체가 아니라 의도된 버퍼다. 원래 조건 유지.
        let staleCutoff = target - interval * (timeline.count >= 6 + Int(extraLatencySlots) ? 1.2 : 2.5)
        let candidates = timeline.filter { $0.timestamp > lastPresentedTimestamp && $0.timestamp <= target + 0.002 }
        var pick = candidates.first
        // 따라잡기는 틱당 최대 1장만 건너뜀 — 여러 장을 한 번에 버리면 눈에 보이는 점프.
        // 주의: staleDrop을 paceMiss로 세지 않는다 — 이 드롭은 큐 잉여 트림이라 지연을 늘려도
        // 안 사라지고(lat=+4에서도 지속 실측), miss로 세면 감쇠가 영영 막혀 e2e만 부푼다
        // (73→101ms 실측). 지연 부족(지각 도착)은 drain의 lateBar가 따로 센다.
        // stale-skip은 **진짜 백로그(깊은 큐)일 때만**. 얕은데(tl 2~6) 늦게 온 프레임을 버리면
        // 다음 틱에 보여줄 게 없어 구멍(max-hold 83ms 실측)이 된다 — work(45~87ms)가 지연 버퍼를
        // 초과해 프레임이 66~91ms 늦게 도착하는 버스트 소스에서 stale-drop이 오히려 스타베이션을
        // 키웠다([STALE] 실측: tl=cand=2~4로 얕은데 age 76ms 드롭). 늦은 프레임은 버리기보다
        // +1vsync 늦게라도 표시하는 게 구멍보다 낫다(주석 1437 원래 의도 복원). 깊을 때(의도된 버퍼
        // 깊이 3+적응슬롯 초과)만 한 장 건너뛰어 백로그 배출.
        let deepBacklog = timeline.count > 3 + Int(extraLatencySlots)
        if let current = pick, current.timestamp < staleCutoff, candidates.count > 1, deepBacklog {
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
            // sourceIntervalEMA는 PLL(snapTimestamp)이 단독 소유 — 여기서 raw delta로
            // 덮어쓰면 히스테리시스의 락 기준 자체가 매 프레임 오염된다 (리뷰 지적).
            if delta < diagSrcIntMin { diagSrcIntMin = delta }
            if delta > diagSrcIntMax { diagSrcIntMax = delta }
        }
        let previousAcceptedTs = lastAcceptedTimestamp
        let previousAcceptedFingerprint = lastAcceptedFingerprint
        lastFrameArrivalAt = CFAbsoluteTimeGetCurrent()   // 좀비 오버레이 판정용 (프레임 공급 생존 신호)
        lastAcceptedTimestamp = slot.timestamp
        lastAcceptedFingerprint = slot.contentFingerprint
        performanceMonitor.recordFrameArrival()
        diagSourceCount += 1

        // 타임스탬프를 콘텐츠 케이던스 그리드에 스냅 (양자화 지터 제거)
        let snappedTs = snapTimestamp(raw: slot.timestamp, rawDelta: delta)

        // 단계 계측 진입 시각 — capIngest(캡처→인제스트 큐 대기) 산출용. slot.timestamp는 SCK
        // 호스트 클럭(mach)이라 CACurrentMediaTime과 동일 기준.
        let tStageEnter = stageDbg ? CACurrentMediaTime() : 0
        let rawCaptureTs = slot.timestamp

        // cb1(blit+검출)은 splitQ면 copy 큐, 아니면 workQueue(기존). cb2(warp)는 항상 workQueue.
        let cb1Queue = (splitQueueEnabled ? copyQueue : nil) ?? workQueue
        guard let workQueue, let cb1Queue,
              let stable = acquireStableTexture(width: sourceTexture.width, height: sourceTexture.height),
              let cb = cb1Queue.makeCommandBuffer() else {
            diagPoolExhaustCount += 1
            // 이 프레임은 수용 실패 — 상태를 되돌려 다음 프레임이 정상 쌍(연속성 유지)을 만들게 함.
            // 지문도 함께 롤백 — 안 하면 드롭된 콘텐츠와 동일한 후속 프레임이 '중복'으로
            // 거절돼 실제 콘텐츠 변화가 다음 변화까지 표시 지연 (감사 확정)
            lastAcceptedTimestamp = previousAcceptedTs
            lastAcceptedFingerprint = previousAcceptedFingerprint
            resetSnapState()
            return
        }

        // 안정 복사 (SCK IOSurface 재활용에서 분리) — **별도 cb로 즉시 커밋 + ready signal**.
        // 이전엔 blit이 큰 cb(마지막 커밋)에 있어, RIFE pack(자체 큐, 즉시 커밋)이 blit보다
        // 먼저 실행되며 '아직 안 쓰인 stableB(옛 풀 내용)'를 읽음 → 쌍이 (A, 수 프레임 전)이
        // 되어 flow 폭주(등속 팬에서 |flow| 1.6↔24.9px 요동, I프레임 위치 랜덤 = 부들부들 실측).
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(
                from: sourceTexture, sourceSlice: 0, sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: sourceTexture.width, height: sourceTexture.height, depth: 1),
                to: stable, destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
        } else {
            // 인코더 생성 실패를 삼키면 **한 번도 쓰이지 않은 텍스처가 그대로 표시된다**
            // (풀은 .private이고 클리어하지 않는다 = 초기화 안 된 GPU 메모리). 조용히 넘기지 않는다.
            DiagnosticLog.shared.log("[INGEST] ⚠︎ blit 인코더 생성 실패 — 이 프레임 폐기")
            lastAcceptedTimestamp = previousAcceptedTs
            lastAcceptedFingerprint = previousAcceptedFingerprint
            return
        }

        // 시간축 정지-UI 검출 갱신 — blit 직후 같은 cb1(소스 준비됨, 순서 보장). 누적 마스크는
        // cb1 완료(=stableReady) 후 유효 → cb2 워프가 stableReady 대기 후 읽으므로 안전.
        // **스트라이드(매 6프레임)**: 이 4K 갱신이 cb1 GPU +3ms인데, workQueue 백로그가 그걸
        // +11ms work로 증폭한다(진단 확정). UI 정적 영역은 초 단위로 지속하므로 매 프레임 갱신은
        // 낭비 — 갱신 안 하면 직전 누적 마스크가 그대로 유효(maskTex 미재기록). MetalFlow 4K에서
        // work 40→목표↓의 최대 단일 레버.
        uiDetectFrame &+= 1
        if uiDetector != nil, uiDetectFrame % 6 == 0 {
            uiDetector?.update(source: stable, into: cb)
        }
        // Vision 텍스트 검출 (~2초 주기) — cb1에 스냅샷 blit, 완료 후 백그라운드에서 검출
        scheduleVisionTextDetection(source: stable, cb: cb)

        // 실프레임 덤프 — 무장(⌃⌥⌘D)돼 있으면 이 고유 프레임을 PNG로. blit과 같은 cb1(순서 보장).
        if frameDumpRemaining > 0, let dir = frameDumpDir {
            dumpStableFrame(stable, cb: cb, dir: dir, index: frameDumpIndex)
            frameDumpIndex += 1
            frameDumpRemaining -= 1
            if frameDumpRemaining == 0 {
                DiagnosticLog.shared.log("[FRAMEDUMP] \(frameDumpIndex)장 완료 → \(dir.path)")
            }
        }

        // cb1 커밋: blit 완료를 이벤트로 알리고, 보간 인코딩은 새 cb2로 (같은 큐라 순서 보장)
        if stableReadyEvent == nil { stableReadyEvent = device.makeSharedEvent() }
        stableReadyCounter += 1
        let readyValue = stableReadyCounter
        if let ev = stableReadyEvent {
            cb.encodeSignalEvent(ev, value: readyValue)
        }
        if stageDbg {
            cb.addCompletedHandler { [weak self] b in
                guard let self else { return }
                let g = (b.gpuEndTime - b.gpuStartTime) * 1000.0
                self.stageLock.lock(); self.stgCb1Gpu += g; self.stgLastCb1Gpu = g; self.stageLock.unlock()
            }
        }
        cb.commit()
        guard let cb2 = workQueue.makeCommandBuffer() else {
            lastAcceptedTimestamp = previousAcceptedTs
            lastAcceptedFingerprint = previousAcceptedFingerprint   // 위와 동일 — 지문 롤백
            resetSnapState()
            return
        }
        if let ev = stableReadyEvent {
            (pairEngine as? RIFEEngine)?.noteInputReady(event: ev, value: readyValue)
            // **항상 기다린다.** 예전엔 splitQ일 때만 걸었다 — "미분리면 같은 큐 in-order라 불필요"
            // 라는 이유였는데, 그 전제는 **cb2가 비어 있을 때 깨진다.**
            //
            // cb2에 인코딩되는 건 encodePair뿐이라, 보간이 꺼지면 cb2는 완료 핸들러만 달린 빈
            // 커맨드 버퍼가 된다. 빈 버퍼는 GPU 실행 없이 회수될 수 있어 cb1의 blit보다 먼저
            // 완료될 수 있는데, 그 핸들러가 "이 텍스처에 프레임이 들어있다"를 알리는 **유일한**
            // 신호다. 게다가 present는 또 다른 큐(presentQueue)에서 GPU 동기화 없이 그 텍스처를
            // 읽는다. 풀 텍스처는 .private에 클리어도 안 하므로, 결과는 초기화 안 된 GPU 메모리 —
            // 실측 2026-07-26: 첫 캡처를 보간 OFF로 시작하면 화면 전체가 색 노이즈가 됐고,
            // 보간을 한 번 켰다 끄면 "고쳐졌다". 후자는 풀이 실제 프레임으로 채워져 최악이
            // "한 프레임 낡음"으로 바뀐 것뿐이라, 고친 게 아니라 가린 것이었다.
            //
            // 같은 큐인 경우엔 이미 순서가 보장되므로 이 대기는 실질 비용이 없다. cb1은 이 시점에
            // 이미 커밋돼 있어 교착도 불가능하다.
            cb2.encodeWaitForEvent(ev, value: readyValue)
        }
        // 이하 보간/핸들러는 cb2에 인코딩

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
            // 갭이 디스플레이 한 프레임보다 작으면 그 사이에 표시 슬롯이 없어 보간 무의미 —
            // 게다가 브라우저 버스트 배달(30fps인데 2프레임이 7ms로 붙어 옴)에서 이 퇴화 쌍을
            // 억지로 보간하면 ANE 과부하로 engFail·아티팩트가 난다(실측). 슬롯 없는 갭은 스킵해도
            // 구멍이 안 생기므로(보여줄 자리가 없음) 문턱을 한 프레임으로 올려 버스트를 걸러낸다.
            let contentAlreadyFast = gap < displayInterval
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
                    var steps = max(1.0, (gap / interval).rounded())
                    // 갭 확장 억제: steps>1은 "소스가 느리다"가 아니라 **프레임을 잃었다**는 뜻이다
                    // (소스가 진짜 느리면 EMA가 따라가 steps=1이 된다). 여유가 있을 때 구멍을 메우는
                    // 건 옳지만, 부하로 잃은 것이면 잃은 만큼 더 만들어 부하를 키우는 자기강화가 된다:
                    // 드롭 → 갭 2배 → 3장 생성 → 4K 워프 3회 + ANE 3회 → work 50-65ms → 더 드롭.
                    // 실측(4K 디스플레이 캡처): t×3 anchors=3, present 124/240, staleDrop 103.
                    // 그래서 거버너가 개입 중이면 확장을 접고 기본 배율만 만든다.
                    if !gapExpansionAllowed { steps = 1 }
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
                    // Auto 배율(frameMultiplier=0)이 타는 경로. 갭이 커질수록 t가 비례해 늘어나므로
                    // 위 멀티플라이어 경로와 같은 자기강화 위험이 있다 — 부하 중엔 갭이 드롭 때문에
                    // 커진 것이라 더 만들면 더 잃는다. 확장 억제 시 표시 슬롯 2개분(=1장)만.
                    let nRaw = Int((gap / displayInterval).rounded())
                    let n = gapExpansionAllowed ? nRaw : min(nRaw, 2)
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
                    // 확장 억제 시 이 경로도 상한을 조인다 — 4K 버스트에서 소스가 불균일해 위
                    // 정수배율 조건을 못 맞추고 이 분기로 빠지면, 벌어진 갭을 슬롯마다 채워 t×6까지
                    // 나온다. 그게 절반 폐기되며 σ 13ms 저더의 원인(실측). 억제 중엔 2장까지만.
                    let vsyncCap = gapExpansionAllowed ? 8 : 2
                    while slotTime < snappedTs - displayInterval * 0.6 && tValues.count < vsyncCap {
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
                // 거버너 t 상한 — **세 생성 경로 공통**. 경로별로 걸면 Auto 배율(=0)처럼
                // 다른 분기를 타는 설정에서 그냥 새어나간다(실측: 캡을 첫 분기에만 걸었더니
                // 강등 후에도 t×5·t×6이 계속 나옴). 균등 간격으로 솎아 케이던스는 보존.
                if let cap = tCountCap, tValues.count > cap {
                    if cap <= 0 {
                        tValues = []
                    } else {
                        let stride = Double(tValues.count) / Double(cap)
                        tValues = (0..<cap).map { tValues[min(Int(Double($0) * stride), tValues.count - 1)] }
                    }
                }
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
                // 문턱을 타임라인 하드캡(12, 틱 스텝2 트림) 아래로 클램프 — 안 하면 저fps
                // 소스(24fps@120Hz 등)에서 expectedDepth≥13이라 배압이 영구 사문화되고,
                // 대신 캡 트림이 '표시 직전' 프레임을 무진단 대량 드롭 (감사 확정).
                // 60fps 경로는 expectedDepth≤11이라 거동 불변.
                if timeline.count >= min(expectedDepth, 10) && tValues.count > 1 {
                    // 절반 솎아내기 (홀수 인덱스 유지)
                    tValues = tValues.enumerated().filter { $0.offset % 2 == 1 }.map(\.element)
                }
                if timeline.count >= min(expectedDepth + 4, 12) {
                    tValues = []
                    diagSkipBackpressure += 1
                }
                pairEngine?.setUIMask(uiDetector?.mask)   // 정지-UI 프리즈 마스크 (없으면 nil)
                interpResult = tValues.isEmpty ? nil : pairEngine?.encodePair(
                    stableA: prev.texture, stableB: stable,
                    tsA: prev.timestamp, tsB: snappedTs,
                    tValues: tValues,
                    into: cb2
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

        // 반납은 '직전' 텍스처 — stable(이번)의 마지막 GPU 리더는 **다음 쌍의 pack**(A로 읽음)
        // 이라 지금 반납하면 풀 재사용 blit이 pack과 레이스. prev의 마지막 리더(이번 pack/보간)는
        // 이 cb2가 기다리므로 cb2 완료 시 반납이 안전.
        let releasePrevID = prevStable.map { ObjectIdentifier($0.texture) }
        prevStable = (stable, snappedTs, slot.timestamp)
        inFlightTextures[ObjectIdentifier(stable)] = stable

        let entryTs = snappedTs
        let mailboxRef = mailbox
        let stableRef: any MTLTexture = stable
        let interpFrames = interpResult?.frames ?? []
        let cutEvaluator = interpResult?.sceneCutEvaluator
        let startTs = pairStartTs
        let gapRef = pairGap
        let stageEnterRef = tStageEnter
        let rawCaptureTsRef = rawCaptureTs
        cb2.addCompletedHandler { [weak self] cb2Buf in
            // 엔진 GPU 비용 EMA — 자동 flow 스케일러의 입력(우리 몫이 예산의 몇 %인지).
            // 완료 핸들러에서 double 2개 읽기라 상시 켜도 비용 없음.
            if let self {
                let g = (cb2Buf.gpuEndTime - cb2Buf.gpuStartTime) * 1000.0
                if g > 0, g < 200 { self.engineGpuMsEMA = self.engineGpuMsEMA <= 0 ? g : self.engineGpuMsEMA * 0.9 + g * 0.1 }
            }
            if let self, self.stageDbg {
                let cb2Gpu = (cb2Buf.gpuEndTime - cb2Buf.gpuStartTime) * 1000.0
                let capIngest = (stageEnterRef - rawCaptureTsRef) * 1000.0
                let workNow = (CACurrentMediaTime() - stageEnterRef) * 1000.0

                // ── 스파이크 프레임 단독 기록.
                // [STAGE]는 120프레임 평균이라 꼬리가 묻힌다. 그런데 지연을 정하는 건 평균이 아니라
                // work **p90**이다(requiredExtra = ceil((paceWorkP90 + 2 − base)/slot)). 실측: 60fps
                // 구간에서 work 평균 21ms인데 창의 89%가 최대 40ms를 넘고 p90이 67ms — 그 꼬리 때문에
                // lat이 +3~4까지 올라가 e2e에 25~33ms가 상시로 실린다. 어느 단계가 튀는지 알아야
                // 꼬리만 잘라낼 수 있으므로, 기준선의 2.5배를 넘는 프레임 하나를 통째로 찍는다.
                let base = self.stgWorkEMA
                self.stgWorkEMA = base <= 0 ? workNow : base * 0.95 + workNow * 0.05
                let now = CACurrentMediaTime()
                if base > 0, workNow > max(25.0, base * 2.5), now - self.stgLastSpikeLog > 1.0 {
                    self.stgLastSpikeLog = now
                    let c1 = self.stgLastCb1Gpu
                    DiagnosticLog.shared.log(String(format:
                        "[SPIKE] work=%.1fms (기준 %.1f) capIngest=%.1f cb1gpu=%.1f cb2gpu=%.1f 대기=%.1f t×%d",
                        workNow, base, capIngest, c1, cb2Gpu, max(0, workNow - capIngest - c1 - cb2Gpu),
                        interpFrames.count))
                }

                self.stageLock.lock()
                self.stgCapIngest += capIngest
                self.stgCb2Gpu += cb2Gpu
                self.stgWork += workNow
                self.stgCount += 1
                let n = self.stgCount
                if n >= 120 {
                    let ci = self.stgCapIngest / Double(n), c1 = self.stgCb1Gpu / Double(n)
                    let c2 = self.stgCb2Gpu / Double(n), wk = self.stgWork / Double(n)
                    self.stgCapIngest = 0; self.stgCb1Gpu = 0; self.stgCb2Gpu = 0; self.stgWork = 0; self.stgCount = 0
                    self.stageLock.unlock()
                    DiagnosticLog.shared.log(String(format:
                        "[STAGE] capIngest=%.1f cb1gpu=%.1f cb2gpu=%.1f work=%.1f (대기=%.1f) ms/frame (n=120)",
                        ci, c1, c2, wk, max(0, wk - ci - c1 - c2)))
                } else {
                    self.stageLock.unlock()
                }
            }
            // 장면 전환이면 보간 프레임 폐기 — 무관한 두 샷 사이의 모핑 프레임 방지
            let isSceneCut = cutEvaluator?() ?? false
            var entries: [TimelineEntry] = []
            // captureTimestamp에는 **스냅 전 원본 캡처 시각**을 싣는다(entryTs는 스냅된 값).
            // e2e = presentedAt - captureTimestamp인데 스냅된 값을 기준으로 재면, 스케줄러가
            // 지연을 더할수록 보고되는 e2e가 오히려 **줄어든다** — 스케줄러가 자기 채점표를
            // 쥐고 있는 셈이라 B1(버스트 창) 같은 변경의 비용을 원리적으로 볼 수 없다.
            // timestamp(표시 슬롯)는 스냅된 값 그대로 — 그건 페이싱의 입력이라 건드리지 않는다.
            if !isSceneCut {
                for frame in interpFrames {
                    let ts = startTs + gapRef * Double(frame.t)
                    entries.append(TimelineEntry(timestamp: ts, texture: frame.texture, isInterpolated: true, captureTimestamp: rawCaptureTsRef, stamp: frame.stamp))
                }
            }
            entries.append(TimelineEntry(timestamp: entryTs, texture: stableRef, isInterpolated: false, captureTimestamp: rawCaptureTsRef))
            // 캡처 시각 → 타임라인 등재까지의 파이프라인 지연 (스케줄러 offset 튜닝 지표)
            let workLatency = (CACurrentMediaTime() - entryTs) * 1000.0
            mailboxRef.postCompleted(entries: entries, released: releasePrevID, workLatencyMs: workLatency, sceneCut: isSceneCut)
        }
        cb2.commit()
    }

    nonisolated private func presentEntry(_ entry: TimelineEntry, at targetTimestamp: CFTimeInterval, drawable: any CAMetalDrawable) {
        if inFlightPresents.withLock({ $0 }) >= 2 { diagPresentBusy += 1 }

        guard let presentQueue, let cb = presentQueue.makeCommandBuffer() else { return }
        guard let surface = renderSurface else { cb.commit(); return }
        // 표시 프레임 덤프 — 무장 시 이 표시분을 PNG로 (S/I·콘텐츠 ts를 파일명에)
        if outDumpRemaining > 0, let dir = outDumpDir {
            outDumpRemaining -= 1
            let idx = outDumpIndex
            outDumpIndex += 1
            let kind = entry.isInterpolated ? "I" : "S"
            let tsMs = Int((entry.timestamp.truncatingRemainder(dividingBy: 100)) * 1000)
            let path = dir.appendingPathComponent(String(format: "out_%03d_%@_%06d.png", idx, kind, tsMs)).path
            let tex = entry.texture
            let w = tex.width, h = tex.height
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
            desc.storageMode = .shared
            desc.usage = [.shaderRead]
            if let shared = device.makeTexture(descriptor: desc), let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
                          sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                          sourceSize: MTLSize(width: w, height: h, depth: 1),
                          to: shared, destinationSlice: 0, destinationLevel: 0,
                          destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                blit.endEncoding()
                let queue = frameDumpFileQueue
                cb.addCompletedHandler { _ in
                    queue.async { AppState.writeTexturePNG(shared, width: w, height: h, path: path) }
                }
            }
            if outDumpRemaining == 0 {
                DiagnosticLog.shared.log("[OUTDUMP] \(outDumpIndex)장 완료 → \(dir.path)")
            }
        }
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
        lastPresentedStamp = entry.stamp
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
            snapAnchor = raw
            return raw
        }
        // 기본 주기(fundamental) 추정 — 시간폭/프레임수(평균 도착률)는 중복/드랍 갭(2×주기)이
        // 섞이면 가짜 격자를 만든다(실측: 60fps+중복20% → 21ms 격자 → 스냅 위상 뒤죽박죽,
        // 표시 패턴 IISS 뭉침 = "앞뒤로 왔다갔다"). 델타 중앙값에서 시작해 배수 접기(33.4→÷2)로
        // 정련하면 VFR/중복 섞임에도 진짜 주기(16.7)에 락된다.
        var deltas: [Double] = []
        for i in 1..<snapTsRing.count {
            let d = snapTsRing[i] - snapTsRing[i - 1]
            if d > 0.002, d < 0.5 { deltas.append(d) }
        }
        guard deltas.count >= 3 else {
            snappedLastTimestamp = max(raw, snappedLastTimestamp + 0.001)
            return snappedLastTimestamp
        }
        func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        var candidate = median(deltas)
        // 버스트 감지 — 짧은 델타(압축 배달)가 있으면 접기가 오답: k==0으로 버려지고 긴
        // 델타가 주기로 오인된다(8/58ms를 58 주기로 락, 리뷰 지적). 기준은 도착률 평균 —
        // 중앙값 기준이면 3장 이상 묶음(짧은 델타가 과반 → 중앙값 자체가 짧은 쪽)에서
        // 미탐되어 EMA가 버스트 내부 간격(~7ms)으로 붕괴한다(리뷰 2차 지적). 평균은 묶음
        // 크기와 무관하게 진짜 케이던스 근방이라 짧은/긴 구분 기준으로 안전하다.
        // 드랍만 있는 소스(모든 델타 ≥ 기본주기 ≈ 0.8×평균)는 짧은 델타가 없어 미발동.
        let arrivalMean = deltas.reduce(0, +) / Double(deltas.count)
        let shortCount = deltas.filter { $0 < arrivalMean * 0.5 }.count
        if shortCount >= 2 {
            // 버스트의 각 델타는 드랍이 아니라 '한 콘텐츠 슬롯의 압축 배달' — 도착률
            // 평균(시간폭/개수)이 진짜 케이던스(33ms).
            candidate = arrivalMean
            // 드랍 혼입 가드 — 버스트 윈도에 드랍이 섞이면 평균이 과대추정된다(리뷰 지적:
            // [7,93] 지속 = 33 소스인데 평균 50). 갭만으로는 슬롯 수를 셀 수 없으므로
            // (버스트 압축이 갭 경계를 ±주기만큼 흔든다), 락된 주기의 정수비(×1.5/×2/×3)
            // 근방이면 드랍으로 얇아진 윈도로 보고 락을 유지한다.
            if sourceIntervalEMA > 0 {
                let ratio = candidate / sourceIntervalEMA
                for m in [1.5, 2.0, 3.0] where abs(ratio - m) < m * 0.12 {
                    candidate = sourceIntervalEMA
                    break
                }
            }
        } else {
        for _ in 0..<2 {   // 배수 접기 정련: 각 델타를 최근접 정수배로 나눠 기본 주기 후보로 환원
            let folded = deltas.compactMap { d -> Double? in
                let k = (d / candidate).rounded()
                return k >= 1 ? d / k : nil
            }
            if folded.count >= 3 { candidate = median(folded) }
        }
        // 최종 = 시간폭 ÷ (접기로 센 슬롯 수) — 타이머 지터는 합산에서 상쇄(늦음-편향 중앙값의
        // 과대추정 회피)되고, 중복/드랍 갭은 k=2+로 정규화. 깨끗한 소스에선 기존 시간폭-평균과 일치.
        var slotCount = 0.0
        var slotSpan = 0.0
        for d in deltas {
            let k = (d / candidate).rounded()
            if k >= 1 { slotCount += k; slotSpan += d }
        }
        if slotCount >= 3 { candidate = slotSpan / slotCount }
        }
        let interval = candidate
        guard interval > 0.002 else {
            snappedLastTimestamp = max(raw, snappedLastTimestamp + 0.001)
            return snappedLastTimestamp
        }
        // interval 히스테리시스 — 버스트가 링을 오염시키면 추정 주기가 뒤집혀(60↔10ms 실측)
        // latencyOffset·생성 t 개수가 함께 출렁인다. 락 대비 ±25% 넘는 변화는 3윈도 연속
        // 지속(진짜 케이던스 변화)일 때만 채택, 순간 오염은 락 유지. 작은 변화는 천천히 블렌드.
        if sourceIntervalEMA > 0, abs(interval - sourceIntervalEMA) > sourceIntervalEMA * 0.25 {
            snapIntervalDeviateStreak += 1
            if snapIntervalDeviateStreak >= 3 {
                snapIntervalDeviateStreak = 0
                sourceIntervalEMA = interval
            }
        } else {
            snapIntervalDeviateStreak = 0
            sourceIntervalEMA = sourceIntervalEMA > 0
                ? sourceIntervalEMA + (interval - sourceIntervalEMA) * 0.2
                : interval
        }
        let usedInterval = sourceIntervalEMA

        // 예측은 앵커(격자 원점) 기준 정수 스텝 투영 — 성공 시 앵커=스냅이라 깨끗한
        // 경로는 기존 rawDelta 방식과 동일, 이탈 후에는 보호된 원점에서 다시 투영.
        if snapAnchor <= 0 { snapAnchor = snappedLastTimestamp }
        let steps = max(1.0, ((raw - snapAnchor) / usedInterval).rounded())
        var predicted = snapAnchor + steps * usedInterval
        let err = raw - predicted

        if abs(err) <= usedInterval * 0.6 {
            snapMissStreak = 0
            predicted += err * 0.08 // 실클럭 드리프트 추적 (천천히)
            let snapped = max(predicted, snappedLastTimestamp + 0.001)
            snappedLastTimestamp = snapped
            // 앵커는 클램프 미반영 예측값 — 이탈 통과(raw) 직후 예측이 단조 클램프에 걸리면
            // 클램프값을 원점 삼는 순간 격자가 last+1ms 지점으로 끌려가 간격이 요동한다 (리뷰 지적).
            snapAnchor = predicted
            return snapped
        }

        // 이탈 — 이 프레임은 raw로 통과. 앵커는 보호, 3연속(진짜 불연속)만 재동기.
        //
        // **관측치 (B1 판단 근거).** B1(버스트 창 pull)은 PLL 코어를 건드리는 위험한 변경인데,
        // 지금까지 "실제 소스가 버스트를 내긴 하는가, 얼마나 자주인가"를 아무도 세어본 적이 없다.
        // 세 숫자를 [SCHED]에 노출해 그 질문에 먼저 답한다:
        //   snapMiss     — 격자 이탈 횟수
        //   snapPullable — 그중 "당길 수 있었던" 것 (일찍 온 프레임이고 지연이 한 간격 이내)
        //                  = B1을 구현했을 때 실제로 건질 수 있는 프레임 수
        //   pullLagMax   — 거절된 지연의 최대치 (버스트 창을 얼마나 넓혀야 하는지)
        // 한 릴리즈 주기 동안 snapPullable이 0이면 B1은 영구히 닫는다.
        diagSnapMissCount += 1
        if err < 0 {
            let lag = predicted - raw
            if lag <= usedInterval { diagSnapPullableCount += 1 }
            diagSnapPullLagMax = max(diagSnapPullLagMax, lag)
        }
        snapMissStreak += 1
        if snapMissStreak >= 3 || rawDelta > 0.5 {
            snapMissStreak = 0
            snapAnchor = raw
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
        snapAnchor = 0
        snapIntervalDeviateStreak = 0
    }

    /// 사용 중이지 않은 풀 텍스처 획득 (타임라인/직전 소스/마지막 표시/인플라이트 제외)
    nonisolated private func acquireStableTexture(width: Int, height: Int) -> (any MTLTexture)? {
        if width != stablePoolWidth || height != stablePoolHeight {
            stablePool = []
            stablePoolWidth = width
            stablePoolHeight = height
            // 기기 시딩은 **여기서** — 캡처 시작 시점엔 풀이 아직 없어 소스 크기가 0이라
            // 거버너·스케일러가 "4K 무거운 소스" 판정을 아예 못 했다(기존 버그: 로그 "소스 0MP").
            // 첫 프레임에서 실제 해상도가 정해지는 이 지점이 유일하게 정확한 시딩 시점.
            if seedPending {
                seedPending = false
                let px = width * height
                let cores = Self.gpuCoreCountEstimate(device)
                let memGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.loadGovernor.seedForDevice(gpuCoreCount: cores, memoryGB: memGB, sourcePixels: px)
                    self.autoFlowScaler.seed(gpuCoreCount: cores, sourcePixels: px)
                    self.applyGovernorDials()
                }
            }
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
        busy.formUnion(inFlightTextures.keys)

        return stablePool.first { !busy.contains(ObjectIdentifier($0)) }
    }

    // MARK: - Diagnostics

    nonisolated private func maybeLogDiagnostics() {
        guard diagTick % 240 == 0 else { return }  // 진단 중 2초 주기 (@120Hz)

        // 틱 레이트: 240틱의 실제 벽시계 소요 — 2.0s면 무손실 120Hz, 2.05s면 ~117Hz(틱 유실)
        let nowWall = CFAbsoluteTimeGetCurrent()
        let wallSpan = diagLastLogWall > 0 ? nowWall - diagLastLogWall : 0
        let tickHz = wallSpan > 0 ? 240.0 / wallSpan : 0
        lastTickHz = tickHz   // 거버너 신호용 (다음 창에서 읽음)
        diagLastLogWall = nowWall
        let tickCPUAvg = diagTickCPUSum / 240.0
        let pointerEvents = PointerTapStats.drain()
        let ingAvg = diagIngestSamples > 0 ? diagIngestSum / Double(diagIngestSamples) : 0
        let tickStats = String(format: "tick=%.1fHz cpu=%.1f/%.1fms over=%d gap=%d(pre%.1f) mouse=%llu ing=%.2f/%.1fms ingOver=%d",
                               tickHz, tickCPUAvg, diagTickCPUMax, diagTickOverruns, diagTickGaps, diagGapPrevCPUMax,
                               pointerEvents, ingAvg, diagIngestMax, diagIngestOver)
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
            paceWorkAvg = avgWork   // 거버너 복귀용 즉응 신호 (p90은 감쇠 10%/2s라 복귀가 30s+)
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
        if diagCapDropCount > 0 { skipParts.append("capDrop:\(diagCapDropCount)") }
        if diagLeaseDropCount > 0 { skipParts.append("leaseDrop:\(diagLeaseDropCount)") }
        if diagSkipBackpressure > 0 { skipParts.append("backpres:\(diagSkipBackpressure)") }
        if diagPresentBusy > 0 { skipParts.append("drawBusy:\(diagPresentBusy)") }
        let skips = skipParts.isEmpty ? "-" : skipParts.joined(separator: ",")

        let msg = "[SCHED] src=\(diagSourceCount)(\(String(format: "%.0f", srcFps))fps) uniqOut=\(uniquePresented) dupSkip=\(diagDupSkipCount) tsRej=\(diagTsRejectCount) interpEnc=\(diagInterpEncodedCount) skip[\(skips)] present=\(diagPresentCount) (I=\(diagInterpPresentCount)) lat=+\(Int(extraLatencySlots)) \(tickStats) \(ciStats) cut=\(cuts) resync=\(diagResyncCount) snapMiss=\(diagSnapMissCount)(pull=\(diagSnapPullableCount) lagMax=\(String(format: "%.1f", diagSnapPullLagMax * 1000))ms) poolMiss=\(diagPoolExhaustCount) tl=\(timeline.count) | glass(ms): avg=\(String(format: "%.2f", avgInterval)) σ=\(String(format: "%.2f", sqrt(variance))) max=\(String(format: "%.1f", maxInterval)) | srcInt=\(String(format: "%.1f", sourceIntervalEMA * 1000))ms [\(String(format: "%.0f", srcIntLo))~\(String(format: "%.0f", srcIntHi))] | drain=\(String(format: "%.1f", drainAvg))/\(diagDrainDepthMax) | work=\(String(format: "%.0f", avgWork))/\(String(format: "%.0f", maxWork))ms e2e=\(String(format: "%.0f", avgLatency))ms | \(pattern)"
        DiagnosticLog.shared.log(msg)

        // 거버너 과부하 비율 — reset 직전, 카운터가 아직 살아있을 때 계산.
        // "만든 보간 프레임 중 표시 기한을 못 넘겨 버려진 비율". 소스 fps 추정과 무관해,
        // present 총량 방식이 잘 돌던 L2를 저평가해 L3까지 불필요 강등하던 문제(실측)를 없앤다.
        //
        // drawBusy는 손실이 아니다: present 슬롯이 이미 찬 정상 상태(120Hz에 프레임이 다 준비됨)를
        // 뜻하며, 정상 동작에서도 100~240으로 크다. 이걸 손실로 세면 ratio가 0으로 붕괴한다(실측).
        // staleDrop(기한 초과 폐기) + capDrop(타임라인 오버플로)만이 진짜 손실이다.
        // 과부하 = "만든 보간 프레임 중 표시 기한을 못 넘겨 폐기된 비율".
        //
        // 버스트 소스를 예외 처리했었으나 4K에서 역효과였다: 소스가 버스트면 work가 60ms를
        // 넘어도 강등을 면제받아, t×6까지 만들어 절반을 버리고 glass σ가 13ms까지 튀었다(실측
        // 4K 디스플레이 캡처). "프레임 수는 90인데 원본보다 안 부드럽다"의 정체다. 버스트라도
        // 실제로 처리량이 무너지면 강등해야 한다 — 예외를 제거한다.
        let govDropped = diagStaleDropCount + diagCapDropCount
        let govProduced = diagInterpEncodedCount + diagStaleDropCount + diagCapDropCount
        pacePresentRatio = govProduced > 8
            ? max(0, 1.0 - Double(govDropped) / Double(govProduced))
            : 1.0

        diagResyncCount = 0
        diagSnapMissCount = 0; diagSnapPullableCount = 0; diagSnapPullLagMax = 0
        diagIngestSum = 0; diagIngestSamples = 0; diagIngestMax = 0; diagIngestOver = 0
        diagSkipToggleOff = 0; diagSkipEngineNil = 0; diagSkipNoPrev = 0
        diagSkipContentFast = 0; diagSkipBigGap = 0; diagSkipDiscontinuity = 0
        diagSkipEngineFail = 0; diagSkipOther = 0
        diagStaleDropCount = 0; diagCapDropCount = 0; diagLeaseDropCount = 0; diagSkipBackpressure = 0; diagPresentBusy = 0; diagStaleSampleCount = 0

        diagSourceCount = 0
        diagDupSkipCount = 0
        diagTsRejectCount = 0
        _ = wallSpan
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

    /// .common 모드로 반복 타이머 등록 — 메뉴바 팝오버/메뉴 트래킹(.eventTracking) 중에도
    /// 계속 발화하게. scheduledTimer는 .default 전용이라 팝오버를 열면 통계/추적이 멈췄다
    /// (렌더는 전용 스레드라 무관 — 카운터 UI만 얼었던 원인).
    private func addCommonTimer(_ interval: TimeInterval, _ block: @escaping @Sendable (Timer) -> Void) -> Timer {
        let t = Timer(timeInterval: interval, repeats: true, block: block)
        RunLoop.main.add(t, forMode: .common)
        return t
    }

    private func updateStats() {
        // 아무도 못 보는 상태에선 갱신 생략 — 뷰어가 덮은 상태에서 stats 쓰기가 SwiftUI 레이아웃을
        // 돌려 vsync 콜백을 삼킴(실측). 팝오버/분리창(라이브 섹션) 또는 정보 오버레이가 보일 때만 갱신.
        guard popoverVisible || settingsWindowVisible || infoOverlayVisible else { return }
        // @Observable 쓰기는 값이 실제로 바뀔 때만 — 렌더 틱과 같은 메인스레드에서
        // SwiftUI가 설정 뷰 body를 재평가(4K에서 5-15ms)해 vsync 콜백을 삼키는 것 방지.
        // 지터 콘텐츠에서 fps 소수점이 매번 달라 0.5s마다 전체 뷰 무효화 → tick 116-117Hz로
        // 새던 원인 (깨끗한 소스는 값 불변 → 재평가 없음 → 120.0 실측과 정합).
        // 반올림(fps 정수, latency 정수)로 변경 빈도 자체도 낮춘다.
        // 소스 fps: 평활 EMA(1/sourceIntervalEMA) 사용 — performanceMonitor.inputFPS는 최근 60프레임
        // 창이라 순간 stall에 창이 길어지면 8fps 등으로 뚝 떨어져 오표시됐다(실측: 실제 30fps인데
        // 오버레이 8 표기). [SCHED]가 쓰는 EMA와 동일해 안정적으로 실제 소스레이트를 보여준다.
        let emaFps = sourceIntervalEMA > 0 ? (1.0 / sourceIntervalEMA) : performanceMonitor.inputFPS
        let newInput = emaFps.rounded()
        if inputFPS != newInput { inputFPS = newInput }
        // 출력 FPS는 실제 glass 시각(presented handler의 presentedTime)으로 계산 —
        // PerformanceMonitor의 renderTimestamps는 mailbox 드레인 시각이라 틱에 뭉쳐
        // 표시값이 80-120으로 맥놀이 (실프레임은 꾸준한데 지표만 출렁, 실측)
        statsLock.lock()
        let ptSnap = presentedTimes
        let latSnap = latencySamplesMs
        statsLock.unlock()
        // 표본이 부족하면 **이전 값을 유지한다.** 예전엔 0으로 떨어뜨렸는데, 표본 부족은
        // "프레임이 안 나온다"가 아니라 "이 0.5초 창에 presented 기록이 아직 안 모였다"는 뜻이라
        // 실제로는 120fps로 잘 돌고 있는데 오버레이에 0 fps가 뜨는 오표시가 났다(사용자 제보).
        // 같은 이유로 산출 구간이 비정상적으로 짧으면(<0.1s) 표본 잡음이 커 신뢰하지 않는다.
        if ptSnap.count >= 8, let first = ptSnap.first, let last = ptSnap.last, last - first > 0.1 {
            let newOutput = (Double(ptSnap.count - 1) / (last - first)).rounded()
            if outputFPS != newOutput { outputFPS = newOutput }
        }
        let newLatency = (latSnap.isEmpty ? 0 : latSnap.reduce(0, +) / Double(latSnap.count)).rounded()
        if latencyMs != newLatency { latencyMs = newLatency }
        let newScale = overlayManager?.scaleStatus
        if upscaleStatus != newScale { upscaleStatus = newScale }
        refreshInfoOverlay()   // 표시 중이면 라이브 갱신
    }

    /// 정보 오버레이 갱신 — 표시 중이고 캡처 중이면 소스/보간/업스케일 정보를 뷰어에 그린다.
    func refreshInfoOverlay() {
        guard infoOverlayVisible, isCapturing else { overlayManager?.setInfoOverlay(nil); return }
        let engine = mirrorInterpolationEnabled ? interpolationEngine
            : L("Interpolation off", "보간 꺼짐", "補間オフ")
        var lines = ["MacFG"]
        lines.append(L("Source", "소스", "ソース") + ": \(Int(inputFPS)) fps")
        lines.append(L("Output", "출력", "出力") + ": \(Int(outputFPS)) fps · \(engine)")
        lines.append(L("Latency", "지연", "遅延") + ": \(Int(latencyMs)) ms")
        lines.append(L("Upscale", "업스케일", "アップスケール") + ": " + (upscaleStatus ?? L("Off", "끔", "オフ")))
        // flow 해상도 — 두 엔진 다 부하에 따라 스스로 움직이므로(MetalFlow=자동 스케일러,
        // RIFE=자체 사다리) 지금 어디에 있는지 보여준다. 화질이 변한 이유가 보이게.
        if mirrorInterpolationEnabled {
            if selectedRenderMode == .metalFlow {
                let auto = autoFlowScaler.manualOverride
                    ? L("manual", "수동", "手動") : L("auto", "자동", "自動")
                lines.append("Flow: \(Int(MetalFlowEngine.flowBaseLongSide))p (\(auto))")
            } else if selectedRenderMode == .rife, let r = pairEngine as? RIFEEngine, r.currentFlowShort > 0 {
                lines.append("Flow: \(r.currentFlowShort)p (\(L("auto", "자동", "自動")))")
            }
        }
        if let q = qualityToggleStatus { lines.append("⚙︎ " + q) }
        // 거버너가 개입 중이면 알린다 — 화질이 낮아진 이유를 사용자가 알 수 있어야 한다
        if let gov = loadGovernor.statusText {
            lines.append("⚠︎ " + gov)
        }
        overlayManager?.setInfoOverlay(lines.joined(separator: "\n"))
    }

    /// 2026-07-25 MetalFlow 화질 변경(모션비례 신뢰도 / 역방향 결합 / 정적 문턱)을 통째로 껐다 켠다.
    /// 실사용 아티팩트가 이 변경 탓인지 같은 장면에서 즉시 A/B 하기 위한 진단용 토글.
    /// MetalFlow 7/25 화질 변경 A/B 단계 (비트: 1=conf 계열, 2=static 계열).
    /// 셋을 묶어 ON/OFF만 하면 "부드러움↑ / 텍스트 흔들림↑"이 동시에 움직여 범인을 못 가린다.
    /// **기본값 1 (conf만)** — 4K 실프레임 3세트 측정 결과 static 조임은 PSNR을 거의 못 벌면서
    /// (+0.070dB) 흔들림 비용은 제일 크게(-0.442dB) 치러 되돌렸다. 자세한 근거는
    /// MetalFlowEngine.staticLo 주석 참조. 1 → 2 → 3 → 0 → 1 로 순환한다.
    @ObservationIgnored private var qualityStage = 1
    func toggleMetalFlowQualityChanges() {
        qualityStage = (qualityStage + 1) % 4
        // conf 계열 — 모션 비례 신뢰도 문턱 + 방향 혼합 상한. 가려짐/큰 변위에서 flow를 살린다.
        let conf = (qualityStage & 1) != 0
        // static 계열 — 정적 판정 문턱 조임. staticness = 1-smoothstep(statLo, statHi, 프레임차)라
        // 조이면 정적 판정이 **줄어**, 압축 노이즈로 차이가 미세하게 뜨는 정적 UI 텍스트까지
        // 워프 대상이 된다(= 텍스트 흔들림). 반대로 느슨하면 실제로 움직이는 픽셀이 원본에
        // 고정돼 부분 저더가 생긴다. 이 축이 "부드러움 ↔ 텍스트 안정"의 거래다.
        let stat = (qualityStage & 2) != 0
        MetalFlowEngine.confRel = conf ? 0.3 : 0.0
        MetalFlowEngine.confMax = conf ? 0.5 : 0.0
        MetalFlowEngine.staticLo = stat ? 0.004 : 0.008
        MetalFlowEngine.staticHi = stat ? 0.02 : 0.04
        let msg = ["0 이전 동작 (둘 다 OFF)", "1 conf만", "2 static만", "3 7/25 신규 (둘 다)"][qualityStage]
        DiagnosticLog.shared.log("[HOTKEY] MetalFlow \(msg)")
        qualityToggleStatus = msg
        refreshInfoOverlay()
        NSSound.beep()
    }
    /// 정보 오버레이에 현재 토글 상태 표시 (nil이면 미표시 = 한 번도 안 누름)
    @ObservationIgnored private var qualityToggleStatus: String?

    /// 정보 오버레이 토글 (단축키)
    func toggleInfoOverlay() {
        infoOverlayVisible.toggle()
        refreshInfoOverlay()
        DiagnosticLog.shared.log("[HOTKEY] 정보 오버레이 \(infoOverlayVisible ? "ON" : "OFF")")
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
        // 소스가 전체화면이라 뷰어를 **자동으로** 띄운 상태면 손대지 않는다. 안 그러면
        // 업스케일을 끄는 순간 배치를 cover로 바꿔 출력 창을 재생성하고, 다음 추적 틱(15~30Hz)에
        // detectFullscreenAutoViewer가 도로 viewer로 되돌려 또 재생성한다 — 무동작이어야 할
        // 설정 변경에 검은 화면 번쩍임 + 스케줄러 리셋 2회 + attachRenderDriver 2회.
        // 전체화면 이탈 경로가 살아있는 upscaleMode로 배치를 다시 유도하므로 사용자 선택은 보존된다.
        guard !autoFsViewer else { return }
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
        // 추적 주기가 배치에 의존 — 재생성 없이는 시작 시점 주기에 영구 고정 (리뷰 확정)
        restartTrackingTimer()
    }

    /// 전체화면 자동 뷰어 — 트래킹 0.5s 폴에서 호출. 2회 연속(=1s) 안정 시에만 1회 전환
    /// (전체화면 전환 애니메이션 중 떨림/썰기 방지). 업스케일로 이미 viewer면 무관.
    private func detectFullscreenAutoViewer() {
        guard isCapturing, !retargetInFlight, captureRegion == nil,
              let om = overlayManager else { return }
        let isFS = om.sourceIsFullscreen
        // 시간 기반 디바운스 — 매 틱(15~30Hz) 호출되므로 샘플 수로 세면 트래킹 주기에 따라
        // 지연이 달라진다. 전환 애니메이션 중 떨림만 걸러내면 되므로 0.2s면 충분하고,
        // 옛 "0.5s 게이트 × 2샘플 = 1s 이상"보다 체감 전환이 확연히 빠르다.
        let nowFS = CFAbsoluteTimeGetCurrent()
        if isFS != fsSample { fsSample = isFS; fsStableSince = nowFS; return }
        if fsStableSince == 0 { fsStableSince = nowFS }
        // 상시 재평가 — 이미 목표 상태면 아래 두 분기가 즉시 return이라 부하 없음.
        guard nowFS - fsStableSince >= 0.2 else { return }
        if isFS {
            syncCaptureSourceForFullscreen(true)
            guard selectedOverlayPlacement == .coverSource else { return }
            autoFsViewer = true
            selectedOverlayPlacement = .viewerWindow
            updateOverlayPlacement()
            DiagnosticLog.shared.log("[AUTOFS] 소스 전체화면 → viewer 자동 전환")
        } else {
            syncCaptureSourceForFullscreen(false)
            guard autoFsViewer else { return }
            autoFsViewer = false
            selectedOverlayPlacement = (upscaleMode == .off) ? .coverSource : .viewerWindow
            updateOverlayPlacement()
            DiagnosticLog.shared.log("[AUTOFS] 소스 창 복귀 → \(selectedOverlayPlacement == .coverSource ? "cover" : "viewer") 원복")
        }
    }

    /// 전체화면 여부에 따라 캡처 소스를 창↔디스플레이로 맞춘다.
    ///
    /// 자체 Space 전체화면에서는 창 캡처가 합성 전 창 버퍼를 주는데 그게 화면 크기와 다를 수
    /// 있다 (실측: 디스코드 전체화면 창 3840x2160인데 콘텐츠 3764x2117, 나머지 알파 0 — 창
    /// 모드에선 정상). 네이티브로는 윈도우서버가 합성하며 맞춰줘 멀쩡하지만, 캡처해 다시 그리면
    /// 여백으로 드러난다. 전체화면이면 소스가 화면 전체를 차지하므로 디스플레이를 캡처해
    /// **합성 결과 그대로** 가져온다 (추정 크롭 없이 보이는 대로).
    private func syncCaptureSourceForFullscreen(_ fullscreen: Bool) {
        guard isCapturing, !isRestartingCapture, !retargetInFlight, captureRegion == nil else { return }
        guard captureManager.isDisplayCapture != fullscreen else { return }   // 이미 원하는 모드
        guard let om = overlayManager else { return }
        retargetInFlight = true
        Task { @MainActor in
            defer { retargetInFlight = false }
            do {
                if fullscreen {
                    guard let scr = om.sourceScreen,
                          let num = scr.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber
                    else { return }
                    try await captureManager.updateToDisplayCapture(
                        displayID: CGDirectDisplayID(num.uint32Value),
                        excludingWindowIDs: om.ownWindowIDs)
                    pendingShowReset = true   // 해상도/소스가 바뀌었으니 스케줄러 재락
                } else {
                    // 역방향(디스플레이→창)은 updateContentFilter로는 스트림이 죽는다 —
                    // 필터 교체는 성공 보고하는데 이후 프레임이 한 장도 안 온다(실측: SCHED 0줄,
                    // 풀이 옛 크기에 고정돼 리사이즈가 무한 반복). 검증된 전체 재시작 경로 사용.
                    await restartCaptureStream(reason: "display → window (전체화면 이탈)")
                }
            } catch {
                DiagnosticLog.shared.log("[SCK-DISPLAY] 전환 실패(\(fullscreen ? "→디스플레이" : "→창")): \(error)")
            }
        }
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
        // 소스 창이 최소화(⌘M)되거나 다른 Space로 가면 SCK가 프레임 공급을 멈추므로, 숨기지
        // 않으면 얼어붙은 마지막 프레임이 floating 레벨로 화면에 남는다 (좀비 오버레이, 리뷰 확정).
        // coverKeepVisible과 무관하게 강제 — "포커스를 잃어도 유지"의 의도는 가려짐이지 소멸이 아님.
        //
        // 판정 신호는 **프레임 공급 중단**이다. 처음엔 kCGWindowIsOnscreen을 썼는데, 전체화면
        // 전환 중 순간적으로 false가 나오면 오버레이가 꺼지고 → 렌더 틱이 조기 반환해 인제스트까지
        // 멈추고 → 텍스처 풀이 옛 크기에 고정돼 리사이즈가 무한 반복되는 연쇄가 났다(실측).
        // "화면이 얼어붙는다"의 실제 조건은 프레임이 안 오는 것이므로 그걸 직접 본다 —
        // 프레임이 흐르는 동안엔 절대 숨지 않으니 보간이 끊길 수 없다.
        let sourceOffScreen = hasReceivedFirstFrame
            && lastFrameArrivalAt > 0
            && CFAbsoluteTimeGetCurrent() - lastFrameArrivalAt > 1.0
            && !(overlayManager?.sourceIsOnScreen ?? true)
        let shouldHide = overlayUserHidden || sourceOffScreen || (!sourceFront && !coverKeepVisible)
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


    /// U2 전체화면/PiP 재타깃: 소스 앱(PID)의 온스크린 창 중 디스플레이를 거의 덮는(≥92%)
    /// 전체화면 창이 새로 나타나면 그리로 무중단 재타깃, 사라지면 원 창 복귀. YouTube 등 HTML5
    /// 전체화면이 새 창을 만들어 원 창엔 검정+썸네일만 남는 문제 대응. statsTimer(0.5s)에서 호출.
    /// MACFG_NO_RETARGET로 비활성. 영역캡처 중엔 비활성(크롭이 원 창 기준이라).
    private func detectFullscreenRetarget() {
        // MACFG_RETARGET=1일 때만 동작 (기본 OFF — 일반 캡처에서 PiP/잔재 창 오탐으로 회귀).
        // 소스 앱이 만든 전체화면 창(f키 플레이어 전체화면)을 전 화면 후보에서 추적해 재타깃.
        guard ProcessInfo.processInfo.environment["MACFG_RETARGET"] != nil,
              isCapturing, !retargetInFlight, captureRegion == nil,
              sourceOwnerPID > 0, originalCaptureWindowID > 0 else { return }
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return }
        let regions: [CGRect] = NSScreen.screens.map(\.frame)
        var fullscreenWID: CGWindowID = 0
        for info in list {
            guard let pid = info[kCGWindowOwnerPID as String] as? pid_t, pid == sourceOwnerPID,
                  let layer = info[kCGWindowLayer as String] as? Int, layer >= 0, layer < 24,
                  let wid = info[kCGWindowNumber as String] as? CGWindowID, wid != originalCaptureWindowID,
                  let b = info[kCGWindowBounds as String] as? [String: CGFloat] else { continue }
            let wf = CGRect(x: b["X"] ?? 0, y: b["Y"] ?? 0, width: b["Width"] ?? 0, height: b["Height"] ?? 0)
            if regions.contains(where: { wf.width >= $0.width * 0.92 && wf.height >= $0.height * 0.92 && $0.contains(CGPoint(x: wf.midX, y: wf.midY)) }) {
                fullscreenWID = wid; break
            }
        }
        let desired = fullscreenWID != 0 ? fullscreenWID : originalCaptureWindowID
        guard desired != currentTargetWindowID else { return }
        retargetInFlight = true
        Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await self.captureManager.updateTargetWindow(windowID: desired)
                self.currentTargetWindowID = desired
                DiagnosticLog.shared.log("[RETARGET] \(desired == self.originalCaptureWindowID ? "원창 복귀 " : "전체화면 전환 ")wid=\(desired)")
            } catch {
                DiagnosticLog.shared.log("[RETARGET] 실패 wid=\(desired): \(error)")
            }
            self.retargetInFlight = false
        }
    }

    /// 현재 최전면 앱의 '가장 위(z-order)' 일반 창 (MacFG 제외). ⌃⌥⌘U 원샷 캡처용.
    /// 목록은 앞→뒤 순서라 첫 유효 창 = 최상단. PiP는 항상-위 창이라 최대화 브라우저보다 위에
    /// 있어 자동으로 잡힌다(사용자는 브라우저 최대화 + PiP 1080 병행). '가장 큰 창'으로 하면
    /// 최대화 브라우저를 잡아 PiP를 놓쳤다(실사용 보고). 전체화면 잔재 띠(3840×68 등)는 종횡비·
    /// 높이로 걸러 "작은 화면+검정" 캡처를 막는다.
    private func frontmostWindow() -> (id: CGWindowID, name: String)? {
        guard let frontPID = NSWorkspace.shared.frontmostApplication?.processIdentifier else { return nil }
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return nil }
        for info in list {
            guard let pid = info[kCGWindowOwnerPID as String] as? pid_t, pid == frontPID,
                  let layer = info[kCGWindowLayer as String] as? Int, layer >= 0, layer < 24,
                  let owner = info[kCGWindowOwnerName as String] as? String, !Self.systemOwners.contains(owner),
                  let wid = info[kCGWindowNumber as String] as? CGWindowID,
                  let b = info[kCGWindowBounds as String] as? [String: CGFloat],
                  let ww = b["Width"], let wh = b["Height"],
                  ww > 50, wh > 120, max(ww, wh) / min(ww, wh) < 8 else { continue }   // 잔재 띠 배제
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
        if hotCapture.isSet {
            bindings.append(.init(id: 3, keyCode: hotCapture.keyCode, modifiers: hotCapture.modifiers) { [weak self] in
                self?.toggleCaptureFocused()
            })
        }
        if hotInterp.isSet {
            bindings.append(.init(id: 4, keyCode: hotInterp.keyCode, modifiers: hotInterp.modifiers) { [weak self] in
                self?.toggleInterpolationHotkey()
            })
        }
        if hotInfo.isSet {
            bindings.append(.init(id: 7, keyCode: hotInfo.keyCode, modifiers: hotInfo.modifiers) { [weak self] in
                self?.toggleInfoOverlay()
            })
        }
        // ⌃⌥⌘M — 설정 창 열기. **절대 조건부로 두지 말 것.**
        //
        // 이 앱은 LSUIElement라 Dock 아이콘이 없고, 메뉴바 상태항목은 macOS가 채택을 거부하면
        // 안 뜬다(2026-07-25 실측). 게다가 applicationShouldTerminate가 시스템發 종료를 거부한다.
        // 그래서 이 단축키가 유일한 탈출구인데, 한때 `if hotInfo.isSet` 블록 **안에** 들어가 있었다
        // — 사용자가 정보 오버레이 단축키를 ✕로 지우면 UI도 Dock도 메뉴바도 종료도 없는 앱이 되고,
        // 그 상태가 설정에 저장돼 재실행해도 복구되지 않는다. 강제 종료 외에 방법이 없었다.
        //
        // 사용자 지정 바인딩들 **뒤에** 붙인다: HotKeyCenter가 배열 순서대로 등록하고 조합이
        // 겹치면 뒤엣것이 -9878로 실패하므로, 뒤에 둬야 사용자가 지정한 조합이 항상 이긴다.
        bindings.append(.init(id: 9, keyCode: UInt32(kVK_ANSI_M),
                              modifiers: UInt32(controlKey | optionKey | cmdKey)) { [weak self] in
            self?.openSettingsWindow()
        })
        // 개발 도구 덤프 단축키 — 개발자 로그 켜진 동안만 등록 (일반 사용자에겐 미노출).
        if devLoggingEnabled {
            bindings.append(.init(id: 5, keyCode: UInt32(kVK_ANSI_D),
                                  modifiers: UInt32(controlKey | optionKey | cmdKey)) { [weak self] in
                self?.startFrameDump()   // 실프레임 삼중항 캡처
            })
            // ⌃⌥⌘Q — MetalFlow 화질 변경(2026-07-25) 즉시 ON/OFF 토글.
            // 지각 비교는 **같은 장면에서 즉시 전환**해야 정확하다(재시작하면 장면이 달라져 흐려짐).
            // 아티팩트 원인이 이 변경인지 사용자가 직접 판정할 수 있게 한다.
            bindings.append(.init(id: 8, keyCode: UInt32(kVK_ANSI_Q),
                                  modifiers: UInt32(controlKey | optionKey | cmdKey)) { [weak self] in
                self?.toggleMetalFlowQualityChanges()
            })
            bindings.append(.init(id: 6, keyCode: UInt32(kVK_ANSI_O),
                                  modifiers: UInt32(controlKey | optionKey | cmdKey)) { [weak self] in
                self?.startOutputDump()  // 표시 시퀀스 변위 분석
            })
        }
        HotKeyCenter.shared.register(bindings)
        // 탈출구(id 9)가 실제로 등록됐는지 로그로 남긴다 — 조합 충돌은 -9878로 조용히 실패한다.
        DiagnosticLog.shared.log("[HOTKEY] 등록 id=\(bindings.map(\.id).sorted()) (9=설정창 탈출구)")
    }

    /// 보간 on/off 전역 토글 (라이브 반영) — 동영상↔텍스트 즉시 전환
    func toggleInterpolationHotkey() {
        isInterpolationEnabled.toggle()
        updateInterpolationEnabled()
        DiagnosticLog.shared.log("[HOTKEY] 보간 \(isInterpolationEnabled ? "ON(동영상)" : "OFF(저지연)")")
    }

    /// 표시 프레임 덤프 무장 (⌃⌥⌘O) — 다음 N개 '표시된' 프레임(S/I 순서 그대로)을 PNG로.
    func startOutputDump(count: Int = 36) {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = fmt.string(from: Date())
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/MacFG/bench_frames/out_\(stamp)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        outDumpDir = dir
        outDumpIndex = 0
        outDumpRemaining = count
        DiagnosticLog.shared.log("[OUTDUMP] 무장 \(count)장 → \(dir.path)")
        NSSound.beep()
    }

    /// 실프레임 덤프 무장 (⌃⌥⌘D) — 다음 N개 고유 소스 프레임을 PNG로 저장.
    func startFrameDump(count: Int = 12) {
        // 캡처 중이 아니면 무장해봐야 프레임이 안 들어와 빈 디렉터리만 남는다(실제로 겪음 —
        // 단축키는 먹고 디렉터리도 생기는데 0장이라 원인을 로그에서야 알 수 있었다).
        // 조용히 실패하지 말고 즉시 알린다.
        guard isCapturing else {
            DiagnosticLog.shared.log("[FRAMEDUMP] ⚠︎ 캡처 중이 아님 — 덤프 취소 (캡처를 먼저 시작할 것)")
            NSSound(named: "Funk")?.play()   // 실패는 다른 소리로 구분
            return
        }
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = fmt.string(from: Date())
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/MacFG/bench_frames/\(stamp)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        frameDumpDir = dir
        frameDumpIndex = 0
        frameDumpRemaining = count
        DiagnosticLog.shared.log("[FRAMEDUMP] 무장 \(count)장 → \(dir.path)")
        NSSound.beep()   // 시작 청각 피드백
    }

    /// stable(.private) → shared 복사(같은 cb) → 완료 시 PNG 파일 쓰기 (오프스레드)
    nonisolated private func dumpStableFrame(_ stable: any MTLTexture, cb: any MTLCommandBuffer, dir: URL, index: Int) {
        let w = stable.width, h = stable.height
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = [.shaderRead]
        guard let shared = device.makeTexture(descriptor: desc),
              let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: stable, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: shared, destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        let path = dir.appendingPathComponent(String(format: "frame_%03d.png", index)).path
        let queue = frameDumpFileQueue
        cb.addCompletedHandler { _ in
            queue.async { AppState.writeTexturePNG(shared, width: w, height: h, path: path) }
        }
    }

    /// bgra8 shared 텍스처 → PNG (BGRA→RGBA)
    nonisolated static func writeTexturePNG(_ tex: any MTLTexture, width w: Int, height h: Int, path: String) {
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&bytes, bytesPerRow: w * 4,
                     from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                     size: MTLSize(width: w, height: h, depth: 1)), mipmapLevel: 0)
        for i in stride(from: 0, to: bytes.count, by: 4) { bytes.swapAt(i, i + 2) }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                        UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dst, img, nil)
        CGImageDestinationFinalize(dst)
    }

    /// 단축키 변경 시: UserDefaults 저장 + 재등록
    func updateHotKeys() {
        if let data = try? JSONEncoder().encode(hotCapture) { UserDefaults.standard.set(data, forKey: "hk.capture") }
        if let data = try? JSONEncoder().encode(hotInterp) { UserDefaults.standard.set(data, forKey: "hk.interp") }
        if let data = try? JSONEncoder().encode(hotInfo) { UserDefaults.standard.set(data, forKey: "hk.info") }
        registerHotKeys()
    }

    private func loadHotKeys() {
        if let d = UserDefaults.standard.data(forKey: "hk.capture"),
           let b = try? JSONDecoder().decode(HotKeyBinding.self, from: d) { hotCapture = b }
        if let d = UserDefaults.standard.data(forKey: "hk.interp"),
           let b = try? JSONDecoder().decode(HotKeyBinding.self, from: d) { hotInterp = b }
        if let d = UserDefaults.standard.data(forKey: "hk.info"),
           let b = try? JSONDecoder().decode(HotKeyBinding.self, from: d) { hotInfo = b }
    }

    /// pairEngine 교체를 렌더 틱과 직렬화 — 틱(ingest)이 nonisolated(unsafe)로 읽는 중
    /// 메인에서 직접 shutdown/nil 하면 encodePair 도중 엔진 풀이 비워져 인덱스 트랩/레이스
    /// (감사 확정 — ⌃⌥⌘I 토글·모드 전환이 캡처 중 이 경로를 탐). perform은 틱과 같은
    /// 런루프라 절대 안 겹치고, 렌더 스레드 미기동 시엔 틱이 없어 직접 대입이 안전.
    private func swapPairEngine(_ newEngine: (any PairInterpolationEngine)?) {
        if renderDriver.isRunning {
            // perform은 동기 + 틱과 직렬 — 블록 실행 중 메인은 대기하므로 실제 동시 접근 없음
            nonisolated(unsafe) let engineRef = newEngine
            renderDriver.perform { [weak self] in self?.pairEngine = engineRef }
        } else {
            pairEngine = newEngine
        }
    }

    /// configurePairEngine 세대 — MainActor async라 prepare await 중 재진입이 교차하면
    /// "늦게 끝난 쪽"이 무조건 이겨 UI 선택과 실제 엔진이 불일치하고 패자 엔진 shutdown이
    /// 누락된다 (리뷰 확정: RIFE 로드 수백 ms 중 빠른 모드 재변경). await마다 세대 검사,
    /// stopCapture도 +1로 진행 중 설치를 무효화.
    private var configureEpoch = 0

    private func configurePairEngine() async {
        // 큐 분리는 predict 대기가 있는 엔진(RIFE)에서만 이득 — 위 선언부 주석의 실측 근거 참조
        splitQueueEnabled = splitQueueOverride ?? (selectedRenderMode == .rife && isInterpolationEnabled)
        configureEpoch += 1
        let epoch = configureEpoch
        let old = pairEngine
        swapPairEngine(nil)          // 틱이 더는 old를 못 보게 먼저 떼어낸 뒤
        old?.shutdown()              // 안전하게 해체 (렌더 스레드는 이미 nil만 봄)

        guard isInterpolationEnabled else {
            interpolationEngine = "Off"
            return
        }

        // 정지-UI 검출기 1회 준비 (엔진 무관 공유). 렌더 스레드 미기동이라 여기서 안전하게 생성.
        if uiDetector == nil {
            let det = UIStaticDetector(device: device)
            try? await det.prepare()
            guard epoch == configureEpoch else { return }   // 교차 진입 — 최신 호출이 이어감
            uiDetector = det
        }
        uiDetector?.reset()   // 새 캡처/엔진 전환 = 불연속 → 누적 리셋

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
            guard epoch == configureEpoch else { engine.shutdown(); return }
            swapPairEngine(engine)
            interpolationEngine = engine.name
            DiagnosticLog.shared.log("[ENGINE] ready: \(engine.name)")
        } catch {
            DiagnosticLog.shared.log("[ENGINE] \(engine.name) prepare FAILED: \(error) → Blend fallback")
            let fallback = LegacyPairEngine(BlendInterpolator())
            if (try? await fallback.prepare(device: device)) != nil {
                guard epoch == configureEpoch else { fallback.shutdown(); return }
                swapPairEngine(fallback)
                interpolationEngine = fallback.name
            } else if epoch == configureEpoch {
                swapPairEngine(nil)
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
