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

    /// gen: 렌더 세대. resetScheduler가 올린다 — 이전 세대의 늦은 GPU 완료가 리셋 후
    /// 새 타임라인에 스테일 프레임을 등재하는 것을 drain에서 걸러낸다(리뷰 확정: stamp 0인
    /// stable 엔트리는 lease 프루닝을 구조적으로 통과하므로 세대 검사가 유일한 방어다).
    func postCompleted(gen: UInt64, entries newEntries: [TimelineEntry], released: ObjectIdentifier?, workLatencyMs: Double, sceneCut: Bool) {
        lock.lock()
        entryGens.append(contentsOf: Array(repeating: gen, count: newEntries.count))
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
    private var entryGens: [UInt64] = []
    private var presentedGens: [UInt64] = []

    func postPresented(gen: UInt64, at time: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool) {
        lock.lock()
        presentedGens.append(gen)
        presentedRecords.append((time, captureTs, isInterp))
        if presentedRecords.count > 480 {
            presentedRecords.removeFirst(240)
            presentedGens.removeFirst(min(240, presentedGens.count))
        }
        lock.unlock()
    }

    /// current 세대와 다른 항목은 버린다. released는 세대 무관 — 이전 세대 텍스처의
    /// 보호 해제는 정리이므로 항상 통과시킨다.
    func drain(current: UInt64) -> ([TimelineEntry], [ObjectIdentifier], [(presentedAt: CFTimeInterval, captureTs: CFTimeInterval, isInterp: Bool)]) {
        lock.lock()
        defer {
            entries = []; entryGens = []
            releasedTextures = []
            presentedRecords = []; presentedGens = []
            lock.unlock()
        }
        let liveEntries = zip(entries, entryGens).filter { $0.1 == current }.map(\.0)
        let livePresented = zip(presentedRecords, presentedGens).filter { $0.1 == current }.map(\.0)
        return (liveEntries, releasedTextures, livePresented)
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
    /// 메뉴바 전용(Dock 아이콘 없음). **기본 true** — 저장된 값이 없을 때 false로 떨어지면
    /// 첫 실행이 일반 앱으로 뜬다(UserDefaults.bool은 미설정 시 false).
    var menuBarOnly: Bool = UserDefaults.standard.object(forKey: "s.menubaronly") as? Bool ?? true
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
    /// 중첩(predict↔warp)을 복원. **기본값은 엔진별 런타임 결정이다**(:4170 — RIFE ON, 그 외 OFF).
    /// MACFG_SPLITQ는 수동 오버라이드. 옛 주석은 "검증 전 기본 OFF"라 적혀 있었는데 반대였고,
    /// 그 문장이 「열린 이슈」 C6("기본 OFF를 재평가")을 만들어 냈다.
    @ObservationIgnored nonisolated(unsafe) private var copyQueue: (any MTLCommandQueue)?
    /// cb1(blit)을 별도 copy 큐로 분리할지 — **엔진별로 다르다.**
    /// 이 분리가 노리는 병목은 "cb2가 RIFE predict(ANE) 이벤트를 기다리며 workQueue를 점유해
    /// 다음 프레임 blit이 head-of-line 블로킹되는 것"이다. MetalFlow는 predict 대기 자체가 없어
    /// 이득 경로가 존재하지 않는다 — 실측도 그랬다(N=20: MetalFlow σ 0.78→0.84·tick 119.3→117.4 손해,
    /// RIFE σ 1.49→1.21·편차 ±1.18→±0.74 개선). 그래서 RIFE에서만 켠다. MACFG_SPLITQ로 수동 오버라이드.
    @ObservationIgnored nonisolated(unsafe) private var splitQueueEnabled = false
    @ObservationIgnored private let splitQueueOverride: Bool? = Knob.string("MACFG_SPLITQ").map { $0 == "1" }

    /// 단계별 지연 분해 계측 (MACFG_STAGEDBG=1). 라이브 work(50~80ms)가 큐 대기인지 GPU 실행인지
    /// 가른다: capIngest(캡처→인제스트 = SCK/큐 대기), cb1(blit+검출 GPU), cb2(warp GPU),
    /// work(캡처→cb2완료 총). GPU 시간은 cb.gpuStart/EndTime(대기 제외 순수 실행). 부하와 무관하게
    /// **비율**이 병목을 드러낸다. 완료 핸들러가 임의 스레드라 stageLock으로 누적.
    /// **개발 로그 토글에 묶는다.** 예전엔 MACFG_STAGEDBG=1 환경변수 전용이었는데, 환경변수는
    /// Finder에서 더블클릭으로 켠 .app에는 전달되지 않는다 — 즉 이 계측이 **실사용에서 구조적으로
    /// 도달 불가**였고, "e2e의 70ms가 어디서 오는가"라는 질문에 답할 유일한 도구가 죽어 있었다.
    /// 로그 파일에만 쓰고 렌더 경로에 분기 하나를 더할 뿐이라 켜져 있어도 비용이 없다.
    @ObservationIgnored nonisolated(unsafe) private var stageDbg =
        Knob.string("MACFG_STAGEDBG") == "1"
        || UserDefaults.standard.bool(forKey: "s.devlog")
    @ObservationIgnored nonisolated(unsafe) private var stgCapIngest = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCb1Gpu = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCb2Gpu = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgWork = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgCount = 0
    /// present 커맨드 버퍼 GPU 시간 — 지금까지 유일하게 안 재던 구간.
    ///
    /// **주의 — 이 계측이 처음 내놓은 결론은 철회됐다(63853b1).**
    /// 2026-08-05에 "present만 매 틱 하면 틱이 130→100Hz로 무너진다"를 관측하고 present 처리량이
    /// 천장이라고 적었으나, 그 A/B는 조건마다 앱을 재시작해 그 사이 라이브 방송 콘텐츠가 흘러갔다.
    /// 두 시기의 실제 차이는 capIngest 13.8ms vs 1.9ms였고 우리 GPU 비용은 같았다. 같은 실행 안에서
    /// 등짝으로 다시 재니 보간 ON + 매 틱 present에서 tick 132Hz / present 122·표시 97로
    /// **재현되지 않았다**. 조건별 재시작 A/B는 이 프로젝트에서 신뢰할 수 없다 —
    /// 한 실행 안 창별 상관으로 봐라(scripts/correlate.py). 관련 상관은 아래 1552/1688행.
    ///
    /// 계측 자체는 유효하니 남긴다. 측정이 틀린 게 아니라 해석이 틀렸다.
    /// 그리고 **"그러므로 present 경로는 무죄"로 읽지 마라** — present 측 손실은 별개로 살아 있는
    /// 사안이다(2246~2261행의 slip 실측 7,696프레임, 1638행 부근).
    @ObservationIgnored nonisolated(unsafe) private var stgPresentGpu = 0.0
    /// GPU 정지 시간 EMA (work − 단계합). 엔진 간 성능 차이의 핵심 항 — stageLock 보호.
    @ObservationIgnored nonisolated(unsafe) private var stgWaitEMA = 0.0
    @ObservationIgnored nonisolated(unsafe) private var stgPresentCount = 0
    /// 직전 진단 창의 벽시계 길이 [s] — 슬롯당 정규화에 필요(present 발생률 산출)
    @ObservationIgnored nonisolated(unsafe) private var diagLastLogWallSpan: Double = 0
    @ObservationIgnored private let stageLock = NSLock()
    /// 커밋 여유(목표 vsync까지 남은 ms) 분포 — 표시된 present / 버려진 present 각각 (B3).
    /// 칸: <0 / <2 / <5 / <10 / >=10 ms. stageLock 보호(완료 핸들러는 임의 스레드).
    @ObservationIgnored nonisolated(unsafe) private var diagLeadShown = [Int](repeating: 0, count: 5)
    @ObservationIgnored nonisolated(unsafe) private var diagLeadDrop = [Int](repeating: 0, count: 5)
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
    /// 캡처 소스 전환(창 ↔ 디스플레이)이 진행 중 — 겹쳐 들어오는 전환과 풀 리사이즈를 막는다.
    private var captureSwitchInFlight = false

    /// **배치 핀 — 설정/AUTOFS 파생을 모두 이긴다 (2026-08-30).**
    ///
    /// 커버가 뷰어보다 초당 20장을 잃는다는 측정이 **통제되지 않았다**: 뷰어 실측이 AUTOFS
    /// 경로였던 탓에 배치 말고도 캡처 소스(창→디스플레이), 소스 Space, 빌드(18.5시간 차),
    /// 소스 도착률(59.5 vs 54.9)까지 같이 달랐다. 게다가 같은 커버 구성에서 무너지지 않은
    /// 세션도 로그에 있다(tick 143.3 / 미표시 10.2 vs 134.9 / 48.0).
    /// 배치를 가르려면 **2×2**(소스 창모드/전체화면 × 배치 커버/뷰어)를 한 실행 안에서
    /// 채워야 하는데, 기존 `s.placement`는 loadSettings에서만 읽히고(재시작 A/B밖에 안 됨)
    /// stopCapture 원복이 그것을 조용히 무시했다.
    /// 핀은 세 경로(설정 파생, stopCapture 원복, AUTOFS)가 전부 존중하고 ⌃⌥⌘P로 즉시 순환한다.
    /// nil = 핀 없음(기존 파생 규칙).
    @ObservationIgnored nonisolated(unsafe) var placementPin: OverlayPlacement?
    /// 배치 라벨 미러 — [SCHED]는 렌더 스레드에서 찍히므로 MainActor 속성을 직접 못 읽는다.
    /// 거버너 미러(gapExpansionAllowed/tCountCap)와 같은 규약. 추적 타이머(MainActor)가 갱신한다.
    @ObservationIgnored nonisolated(unsafe) var placeTagMirror = "?" 
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
        // MACFG_FLOWBASE는 이 줄보다 우선한다. 이게 없으면 flow 해상도를 **어떤 방법으로도**
        // 고정할 수 없다: MetalFlowEngine의 MACFG_MFFLOWBASE는 이 함수가 매 창 덮어쓰고,
        // --flow-base는 CLI 전용인데 이 앱은 Finder로만 띄울 수 있고(셸 실행은 메뉴바 권한을
        // 오염시킨다), MACFG_AUTOFLOW=0은 스케일러를 현재값에 고정할 뿐 값을 못 정한다.
        // flow 해상도는 base²에 비례하는 최대 단일 다이얼이라(800→480이면 flow 비용 0.36배),
        // 측정에서 이걸 못 돌리면 남은 항목들의 기여도를 가릴 수 없다.
        // MACFG_FLOWBASE는 **절대 핀**이다 — 거버너 상한(flowBaseCap)도 무시한다.
        // min()에 같이 넣으면 부하 강등이 측정 중의 핀을 조용히 풀어 A/B가 오염된다(리뷰 확정).
        let pinned = Knob.double("MACFG_FLOWBASE")
        let desired = pinned
            ?? (autoFlowScaler.manualOverride ? userFlowBase : autoFlowScaler.current)
        let base = pinned ?? min(desired, loadGovernor.flowBaseCap ?? desired)
        if MetalFlowEngine.flowBaseLongSide != base {
            MetalFlowEngine.flowBaseLongSide = base
            DiagnosticLog.shared.log("[GOV] flowBase → \(Int(base))")
        }
        // RIFE 워프 해상도 배율 (LSFG식) — RIFE의 실질 중간 강등 다이얼. env 수동 오버라이드가
        // 있으면 그걸 존중(측정용), 없으면 거버너가 설정.
        if Knob.string("MACFG_WARPSCALE") == nil {
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
        // 엔진 비용 기반 t 상한을 여기 넣었다가 **되돌렸다**(2026-08-26).
        // RIFE 4K의 work 22ms 중 16ms가 `대기`인데, t를 3→2→1로 줄여도 대기는 15~16ms로
        // 그대로였다(실측). 원인이 t 개수가 아니라 **cb2가 ANE 추론 완료를 GPU에서 기다리는
        // 구조**이기 때문이다(RIFEEngine: packCB.commit → 워커가 predict → slot.event 시그널
        // → 호출자 cb의 encodeWaitForEvent). 그 대기는 소스 간격(16.7ms)과 거의 같다.
        // 단계별 GPU 합은 두 엔진이 사실상 동일하다: cb1+cb2+present ≈ 4.5ms.
        //   MetalFlow  cb1 0.61  cb2 3.32  present 0.54  대기 **0.32**  work  5.92
        //   RIFE       cb1 0.85  cb2 3.28  present 0.64  대기 **16.28** work 22.18
        tCountCap = bypassButRife ? 1 : loadGovernor.tCountCap
    }

    /// 거버너 미러 (렌더 스레드에서 읽음) — 갭 확장 허용 / t 개수 상한
    @ObservationIgnored nonisolated(unsafe) private var gapExpansionAllowed = true
    @ObservationIgnored nonisolated(unsafe) private var tCountCap: Int?
    /// 쌍당 보간 프레임 수 강제 상한 (측정용, MACFG_TCAP). 거버너 상한과 함께 더 작은 쪽이 이긴다.
    @ObservationIgnored nonisolated(unsafe) private let tCapOverride: Int? = Knob.int("MACFG_TCAP")
    /// 매 틱 강제 재present (측정용, MACFG_ALWAYSPRESENT) — present 레이트와 틱 굶주림의 인과 분리.
    @ObservationIgnored nonisolated(unsafe) private let alwaysRepresent = Knob.string("MACFG_ALWAYSPRESENT") == "1"
    /// N틱마다 한 번만 present (측정용, MACFG_PRESENTEVERY). 1이면 매 틱(기본).
    ///
    /// `alt`를 주면 진단 창(240틱)마다 1↔2를 **교대**한다. 조건마다 앱을 재시작하는 A/B는
    /// 라이브 소스에서 신뢰할 수 없다 — 방송 콘텐츠가 흘러가 소스 fps가 60에서 23으로 바뀌면
    /// 두 조건이 다른 세계가 된다(2026-08-07 실측으로 한 번 날림). 한 실행 안에서 교대하면
    /// 콘텐츠 드리프트가 양쪽에 동일하게 실린다.
    ///   defaults write com.macfg.MacFG env.MACFG_PRESENTEVERY -string alt
    @ObservationIgnored nonisolated(unsafe) private var presentEveryN = Knob.int("MACFG_PRESENTEVERY") ?? 1

    // ── 적응 페이싱 ──
    //
    // 확정된 사실: 표시 실패는 우리 잘못이 아니다. GPU는 100% 목표 슬롯보다 일찍 끝나고
    // (gpuLate 24167샘플 전수), 목표 슬롯 중복도 미표시 1789/s 중 5/s뿐이다. 한 실행 안
    // 40창에서 r(WindowServer CPU, 미표시) = **+0.858**, r(WS, tick) = −0.852 —
    // 컴포지터가 포화되면 우리가 제때 넘긴 프레임이 버려진다.
    //
    // 그런데 **덜 내밀면 나아지는지는 확정하지 못했다.** 같은 실행 안에서는
    // r(present/s, 미표시) = −0.323으로 오히려 음수고(공급이 원인이 아님), 설정 간 비교에서는
    // 65장/s일 때 손실 3.8% / 117장/s일 때 25%로 크게 다르다(다만 보간 on/off라 교란됨).
    // 이 세션에서 메커니즘을 여덟 번 틀렸으므로, 메커니즘을 가정하는 컨트롤러는 쓰지 않는다.
    //
    // 대신 **결과를 직접 오른다**: 목적함수는 초당 실제 표시 프레임 수다. 공급 배율을 조금씩
    // 흔들어 표시가 늘어나는 방향을 유지하고, 나빠지면 방향을 뒤집는다. 감축이 도움이 안 되면
    // 스스로 1.0으로 돌아오므로 최악의 경우가 현재 동작이다.
    // 위상 누산기로 게이팅하므로 남는 present는 **균등 간격**이 된다 — 예전에 등간격으로 낸
    // 실측(MACFG_PRESENTEVERY=3)에서 미표시가 정확히 0이었던 것이 이 형태의 근거다.
    /// **기본 꺼짐 — 이 컨트롤러는 실측에서 무효였다.**
    ///
    /// 배율을 0.75까지 내려도 present/s가 112~115로 전혀 줄지 않았다(1.1.8 실행 26창).
    /// 이유: 틱을 건너뛰어도 타임라인의 due 엔트리는 그대로 남아 **다음 틱에 그냥 나간다.**
    /// 공급을 줄인 게 아니라 미룬 것이라, 총량은 같고 간격만 더 뭉쳤다.
    /// 공급을 진짜로 줄이려면 present 게이트가 아니라 **생산(t 값 개수)** 쪽을 줄여야 한다.
    /// 켜면 다음 측정이 오염되므로 기본값을 껐다. 다시 시험하려면:
    ///   defaults write com.macfg.MacFG env.MACFG_PACEADAPT -string 1
    @ObservationIgnored nonisolated(unsafe) private var paceAdaptive = Knob.string("MACFG_PACEADAPT") == "1"
    @ObservationIgnored nonisolated(unsafe) private var paceScale: Double = 1.0
    @ObservationIgnored nonisolated(unsafe) private var pacePhase: Double = 0
    @ObservationIgnored nonisolated(unsafe) private var paceDir: Double = -0.05
    @ObservationIgnored nonisolated(unsafe) private var paceShown = 0
    @ObservationIgnored nonisolated(unsafe) private var paceWindowStart: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var paceLastRate: Double = -1
    /// 진단용 — [SCHED]에 pace=배율/표시율로 찍는다.
    @ObservationIgnored nonisolated(unsafe) private var paceLastShownRate: Double = 0
    @ObservationIgnored nonisolated(unsafe) private let presentEveryAlternates =
        Knob.string("MACFG_PRESENTEVERY") == "alt"

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
        // **소스 전달률** — 풀 고갈로 통째로 버린 소스 프레임까지 본다. 이게 없으면 붕괴 중에도
        // 틱이 주사율을 내므로 제어기가 "여유"로 오판한다(실측: 소스 40% 파괴 중 achieved=1.00).
        let srcSeen = diagSourceCount + diagPoolExhaustCount
        let deliveryRatio = srcSeen > 8 ? 1.0 - Double(diagPoolExhaustCount) / Double(srcSeen) : 1.0
        // 합성 규칙과 그 근거는 AutoFlowScaler.combinedAchieved 참조 (단위 테스트로 고정돼 있다).
        let achieved = AutoFlowScaler.combinedAchieved(tickRatio: tickRatio,
                                                       keepRatio: pacePresentRatio,
                                                       deliveryRatio: deliveryRatio)
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
    /// **실측 패널 주기 (초). 공칭 1/maximumFramesPerSecond가 아니다.**
    ///
    /// `maximumFramesPerSecond`는 정수라 144를 주지만 실측 패널은 **144.0477Hz**다
    /// (틱 6.9421ms vs 공칭 6.9444ms). 그 2.3µs 차이가 `fast` 게이트에서 부호를 가른다 —
    /// 문턱이 실제 틱보다 **항상 크므로**, 디스플레이 한 틱 간격으로 배달된 진짜 쌍이
    /// 무조건 "너무 빠름"으로 걸려 보간이 통째로 버려진다(Codex 지적, 2026-08-30 실측 확인).
    /// 링크 콜백 간격 중 스킵이 아닌 것만 모아 EMA로 추정한다. 공칭 대비 ±2% 밖이면 무시한다.
    @ObservationIgnored nonisolated(unsafe) private var measuredTickEMA: Double = 0
    /// **달성 틱 레이트 (Hz) — 놓친 콜백까지 포함한 실제 표시 슬롯 수.**
    /// 패널 주사율과 다르다: 창모드에서 패널은 144.048인데 달성은 ~139다(콜백 유실).
    /// [SCHED]가 240틱마다 계산하는 값을 미러링한다.
    @ObservationIgnored nonisolated(unsafe) private var achievedTickHz: Double = 0
    /// 표시 슬롯 계산에 쓸 주기 — 실측이 잡히면 그것, 아니면 공칭.
    nonisolated var effectiveDisplayInterval: Double {
        let nominal = 1.0 / max(mirrorRefreshRate, 30)
        guard measuredTickEMA > 0 else { return nominal }
        let r = measuredTickEMA / nominal
        return (r > 0.98 && r < 1.02) ? measuredTickEMA : nominal
    }
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
        if Knob.string("MACFG_NO_UISTATIC") != nil { UIStaticDetector.enabled = false }
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
                guard let self, !self.isRestartingCapture else { return }
                // 시작 절차 중(엔진 준비 등)이면 isCapturing이 아직 false — 무시하지 말고
                // 표시해 두고, 시작 완료 직후 정리한다(중간에 stopCapture를 끼우면 그 자체가
                // 새 레이스를 만든다).
                if self.isStartingCapture { self.streamDiedWhileStarting = true; return }
                guard self.isCapturing else { return }
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
        stageDbg = devLoggingEnabled || Knob.string("MACFG_STAGEDBG") == "1"
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
        // 배치는 업스케일 모드에서 파생 — 단, `s.placement`가 있으면 그것이 이긴다.
        // 파생 규칙만 있으면 뷰어 창 배치를 무인으로 재현할 방법이 없어서(업스케일을 켜야만
        // 뷰어가 되는데 그러면 GPU 부하가 같이 바뀌어 A/B가 교란된다) 뷰어 전용 문제
        // (마우스 진입 시 프레임 드랍 제보)를 시험할 수 없었다.
        //   defaults write com.macfg.MacFG s.placement -string viewer
        //   defaults delete com.macfg.MacFG s.placement     ← 파생 규칙으로 복귀
        if let forced = d.string(forKey: "s.placement") {
            placementPin = (forced == "viewer" || forced == "beside") ? .viewerWindow : .coverSource
        }
        selectedOverlayPlacement = placementPin ?? (upscaleMode == .off ? .coverSource : .viewerWindow)
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
        // **AppKit 접근(NSScreen)은 반드시 여기서, 메인에서.** 아래 perform 블록은 렌더 스레드에서
        // 돌므로 그 안에서 outputScreen을 읽으면 렌더 스레드가 AppKit을 만지게 된다.
        mirrorRefreshRate = Double(overlayManager?.outputScreen?.maximumFramesPerSecond ?? 120)
        let attachW = Int(surface.metalLayer.drawableSize.width)
        // **부착 조건을 남긴다.** 이게 없어서 "왜 120이 안 나오나"를 추적하는 내내 정작
        // 디스플레이가 144Hz라는 사실을 로그로 확인할 방법이 없었다. 스케줄러의 슬롯
        // 크기(1/refresh)와 드로어블 크기가 전부 여기서 정해지므로 둘 다 찍는다.
        DiagnosticLog.shared.log("[DISPLAY] 부착: \(overlayManager?.outputScreen?.localizedName ?? "?")"
            + " refresh=\(Int(mirrorRefreshRate))Hz"
            + " drawable=\(attachW)x\(Int(surface.metalLayer.drawableSize.height))"
            + " maxDrawables=\(surface.metalLayer.maximumDrawableCount)"
            + " vsync=\(surface.metalLayer.displaySyncEnabled)")
        // **renderSurface 교체는 렌더 스레드에서 한다.**
        // 이 시점엔 옛 CAMetalDisplayLink가 아직 살아 있다 — invalidate는 바로 아래
        // renderDriver.attach의 performSync 안에서야 실행된다. 그 사이 렌더 스레드는
        // presentEntry의 `guard let surface = renderSurface`로 이 필드를 retain한다.
        // 배치 전환 경로에서는 직전에 OverlayManager가 overlayWindow를 nil로 만들어
        // 이 필드가 옛 RenderSurface의 **마지막 강참조**이므로, 메인에서 대입하는 순간
        // dealloc이 시작되고 렌더 스레드가 해제 중인 객체를 retain하게 된다 —
        // 2026-08-05 크래시(-[AGXG16GFamilyTexture retain] on deallocated instance)와 같은 서명.
        // 드라이버가 안 돌 때(생애 최초 attach)는 perform이 조용한 no-op이라 직접 대입해야 한다.
        if renderDriver.isRunning {
            renderDriver.perform { [weak self] in self?.renderSurface = surface }
        } else {
            renderSurface = surface
        }
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
        // 제목은 CLI 인자 우선, 없으면 **UserDefaults**(`s.autocapturetitle`)에서 읽는다.
        // env/CLI만 지원하면 Finder 더블클릭으로 켠 앱에선 도달할 수 없다 — 그런데 셸에서
        // 인자를 주며 띄우면 macOS 26이 그 셸을 메뉴바 항목의 소유자로 기록해 허용 목록이
        // 오염된다(2026-07-25에 하루를 쓴 그 문제). defaults 키면 둘 다 피한다.
        //   defaults write com.macfg.MacFG s.autocapturetitle "치지직"
        //   defaults delete com.macfg.MacFG s.autocapturetitle     ← 끄기
        let cliTitle: String? = {
            guard let idx = args.firstIndex(of: "--auto-capture-title"), idx + 1 < args.count else { return nil }
            return args[idx + 1]
        }()
        let storedTitle = UserDefaults.standard.string(forKey: "s.autocapturetitle")
        guard let rawTitle = cliTitle ?? storedTitle, !rawTitle.isEmpty else { return }
        let titleSub = rawTitle.lowercased()
        if cliTitle == nil { DiagnosticLog.shared.log("[AUTO] 저장된 자동 캡처 제목 '\(rawTitle)'") }

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
        // 제목과 같은 이유로 defaults 폴백을 둔다 (Finder 실행 유지):
        //   defaults write com.macfg.MacFG s.autocapturerect "0,0,960,540"
        let rectSpec: String? = {
            if let rIdx = args.firstIndex(of: "--capture-rect"), rIdx + 1 < args.count { return args[rIdx + 1] }
            return UserDefaults.standard.string(forKey: "s.autocapturerect")
        }()
        if let rectSpec {
            let parts = rectSpec.split(separator: ",").compactMap { Double($0) }
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
           let m = Int(args[mIdx + 1]), (2...6).contains(m) {
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
            // **가장 큰 매치를 고른다** (first가 아니라).
            // 소스가 macOS 전체화면이면 같은 앱의 창 목록에 제목 없는 3840x68짜리 조각 창이
            // 함께 뜨는데(실측 2026-08-06, Firefox 전체화면), first는 그걸 잡아 캡처가 시작은
            // 되지만 프레임이 한 장도 안 온다 — 사용자에겐 "전체화면 소스는 보간도 안 되고
            // 정보 오버레이도 안 뜬다"로 보인다. 면적이 가장 큰 창이 언제나 사용자가 의도한 창이다.
            if let target = availableWindows
                .filter({ $0.displayName.lowercased().contains(titleSub) })
                .max(by: { $0.width * $0.height < $1.width * $1.height }) {
                selectedWindowID = target.windowID
                selectedWindowName = target.displayName
                DiagnosticLog.shared.log("[AUTO] capturing '\(target.displayName)' mode=\(selectedRenderMode.rawValue) placement=\(selectedOverlayPlacement.rawValue)")
                await startCapture()
                // 자체검증: MACFG_AUTODUMP 설정 시 캡처 안정화 후 프레임 덤프 자동 무장
                if Knob.string("MACFG_AUTODUMP") != nil {
                    Task { @MainActor in
                        try? await Task.sleep(for: .seconds(4))
                        self.startFrameDump()
                    }
                }
                if Knob.string("MACFG_AUTOINFO") != nil {
                    Task { @MainActor in try? await Task.sleep(for: .seconds(5)); self.infoOverlayVisible = true; self.refreshInfoOverlay() }
                }
                if let od = Knob.string("MACFG_AUTOOUTDUMP") {
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
    /// stopCapture 진행 중 플래그. 정지는 isCapturing을 **await 전에** 내리므로, 정지가
    /// 스트림 해제를 기다리는 수백 ms 동안 새 시작이 위 guard들을 전부 통과할 수 있다 —
    /// 그러면 옛 정지의 복귀 코드가 새 시작의 스트림 참조·오버레이·엔진을 닫는다(리뷰 확정).
    /// 정지 중 시작 요청은 무시한다(사용자가 다시 누르면 된다).
    @ObservationIgnored private var isStoppingCapture = false
    /// 시작 절차(엔진 준비 수백 ms) 중 SCK 스트림이 죽었다는 표시. 그 시점엔 isCapturing이
    /// 아직 false라 onStreamStopped 핸들러가 무시하는데, 그대로 두면 isCapturing=true의
    /// 좀비 캡처(죽은 스트림, 프레임 0장)가 되고 hasReceivedFirstFrame 기반 자동 정지도
    /// 영영 안 걸린다(리뷰 확정). 시작 완료 직후 이 플래그를 보고 정리한다.
    @ObservationIgnored private var streamDiedWhileStarting = false

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
        guard !isStoppingCapture else {
            DiagnosticLog.shared.log("[CAPTURE] 정지가 진행 중 — 시작 요청 무시 (완료 후 다시)")
            return
        }
        isStartingCapture = true
        streamDiedWhileStarting = false
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
        if Knob.string("MACFG_CBINGEST") != "0" {
            // **프레임마다 런루프를 깨우지 않는다 — 이미 예약돼 있으면 합친다.**
            //
            // 실측(2026-08-07, 실사용 마우스가 창 경계를 넘을 때):
            //   평상시  도착 172/창 → 수용 103, 중복거름 69   갭  6, tick 140Hz
            //   경계    도착 208/창 → 수용 196, 중복거름 12   갭 57, tick 111Hz
            // 마우스가 호버 하이라이트·컨트롤바를 건드려 **매 프레임이 실제로 조금씩 달라지므로**
            // 중복 필터가 40% → 6%로 무너지고, 수용 프레임이 두 배가 된다. drainAndIngest 자체는
            // 여전히 싸지만(ing 평균 0.06ms), **런루프를 깨우는 횟수가 초당 98번**이 되고 그것이
            // 같은 런루프의 vsync 콜백을 삼킨다. 실제로 이 구간에서 우리 GPU 점유는 오히려
            // 33% → 22%로 **내려간다** — GPU 포화가 아니라 런루프 경합이다.
            //
            // 합치면 대기 중인 프레임은 어차피 pendingIngest에 쌓여 있고 한 번의 drain이
            // maxCount만큼 가져가므로 지연 손해가 없다. 깨우기만 줄인다.
            captureManager.onFrameAvailable = { [weak self] in
                guard let self else { return }
                // 이미 예약된 인제스트가 있으면 그것이 처리한다 (compare-and-set)
                guard self.ingestScheduled.withLock({ was in
                    let already = was; was = true; return !already
                }) else { return }
                let queued = self.renderDriver.performAsync { [weak self] in
                    guard let self else { return }
                    self.ingestScheduled.withLock { $0 = false }
                    guard self.isCapturingMirror else { return }
                    // 오버레이 숨김 중에는 틱 경로만 GPU를 양보하고 이 도착 경로는 풀가동이었다
                    // (리뷰 확정) — 숨긴 목적이 GPU 확보이므로 여기도 같이 양보한다. 쌓인 프레임은
                    // 틱의 숨김 분기가 드레인해 텍스처를 풀로 회수한다.
                    guard !self.overlayHiddenState else { return }
                    self.drainAndIngest(maxCount: 4)
                }
                // 런루프 미기동으로 못 태웠으면 플래그를 여기서 되돌린다 — 안 하면 CAS가
                // 영구히 true로 남아 콜백 저지연 경로가 이후 계속 죽는다(틱 폴백만 남음).
                if !queued { self.ingestScheduled.withLock { $0 = false } }
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
                }
            }

            // 창 추적은 30Hz면 충분 — 틱(120Hz)마다 CGWindowList를 부르면 호출당 0.5-2ms로
            // vsync 틱을 놓쳐 출력 fps 천장이 ~110으로 내려앉는다 (실측).
            // 뷰어 배치 15Hz의 원래 근거는 상대커서 매핑의 좌표 신선도였는데, 그 기능은
            // 2026-08-20에 제거됐다. 지금 남은 용도는 소스 창 이동/리사이즈 감지(캡처 재구성)뿐이라
            // 더 낮춰도 될 가능성이 크다 — 다만 낮췄을 때 재구성이 늦어지는지 안 재봤으므로
            // 값은 그대로 둔다. 렌더는 전용 스레드라 메인 CGWindowList 15Hz는 틱에 무해하다.
            // **커버가 뷰어의 2배로 폴링한다 — 커버 모드 프레임 밀림의 용의자다 (2026-08-30).**
            // 실측(같은 4K 60fps 소스, RIFE):
            //   뷰어  slip 215/0/0/0 (100% 정시)  미표시 0.6/창
            //   커버  slip  82/98/0/0 ( 46% 정시)  미표시 **30~40/창** = 초당 20장이 버려진다
            // 이 타이머가 부르는 updateTracking → pollGeometry → CGWindowListCopyWindowInfo는
            // 메인 스레드에서 WindowServer로 가는 **동기 왕복**이고, worklog에
            // r(WindowServer CPU, 미표시) = +0.858이 이미 기록돼 있다.
            // 다만 커버는 소스 위에 겹쳐 4K 표면 두 장을 합성시키기도 하므로 용의자가 둘이다.
            // 노브로 가른다 — 15로 낮춰 미표시가 줄면 폴링, 그대로면 합성이다.
            let coverHz = Knob.double("MACFG_TRACKHZ") ?? 30.0
            let trackHz: Double = selectedOverlayPlacement == .coverSource ? coverHz : 15.0
            trackingTimer = makeTrackingTimer(hz: trackHz)

            // 자동 숨김 기준용 소스 PID + 초기 표시 상태
            sourceOwnerPID = ownerPID(of: windowID)
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
            // 시작 절차 중 스트림이 죽었으면(엔진 준비 수백 ms 사이 소스 창 닫힘 등) 여기서
            // 정리한다. 이 검사와 return 사이에 suspension이 없으므로 MainActor 직렬성이
            // 나머지 창을 닫는다 — 이후 도착하는 알림은 isCapturing=true라 정상 정지 경로를 탄다.
            if streamDiedWhileStarting {
                streamDiedWhileStarting = false
                DiagnosticLog.shared.log("[CAPTURE] 시작 중 스트림 사망 감지 → 정리 정지")
                isStartingCapture = false
                await stopCapture()
                return
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
        // **커버가 뷰어의 2배로 폴링한다 — 커버 모드 프레임 밀림의 용의자다 (2026-08-30).**
        // 실측(같은 4K 60fps 소스, RIFE):
        //   뷰어  slip 215/0/0/0 (100% 정시)  미표시 0.6/창
        //   커버  slip  82/98/0/0 ( 46% 정시)  미표시 **30~40/창** = 초당 20장이 버려진다
        // 이 타이머가 부르는 updateTracking → pollGeometry → CGWindowListCopyWindowInfo는
        // 메인 스레드에서 WindowServer로 가는 **동기 왕복**이고, worklog에
        // r(WindowServer CPU, 미표시) = +0.858이 이미 기록돼 있다.
        // 다만 커버는 소스 위에 겹쳐 4K 표면 두 장을 합성시키기도 하므로 용의자가 둘이다.
        // 노브로 가른다 — 15로 낮춰 미표시가 줄면 폴링, 그대로면 합성이다.
        let coverHz = Knob.double("MACFG_TRACKHZ") ?? 30.0
        let trackHz: Double = selectedOverlayPlacement == .coverSource ? coverHz : 15.0
        trackingTimer?.invalidate()
        trackingTimer = makeTrackingTimer(hz: trackHz)
    }

    private func makeTrackingTimer(hz: Double) -> Timer {
        addCommonTimer(1.0 / hz) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.overlayManager?.updateTracking()
                // 배치/캡처소스 라벨 미러 갱신 — [SCHED] 한 줄로 2×2를 사후에 가르기 위한 것.
                // **updateTracking 뒤에 둔다** — @fs가 읽는 sourceIsFullscreen은 lastSourceFrame
                // 기반이고 그건 updateTracking이 갱신한다. 앞에 두면 태그가 한 틱(33~66ms) 늦다.
                self.placeTagMirror = (self.selectedOverlayPlacement == .coverSource ? "cover" : "viewer")
                    + (self.placementPin != nil ? "!" : "")
                    + ((self.overlayManager?.sourceIsFullscreen ?? false) ? "@fs" : "")
                    + "/" + (self.captureManager.isDisplayCapture ? "disp" : "win")
                // 창 종료/리사이즈 감지 (렌더 틱에서 이관 — overlayManager는 MainActor)
                guard self.isCapturing else { return }
                // 최소화/Space 이동 감지 — 워크스페이스 알림만으로는 안 잡혀 좀비 오버레이가 남는다
                self.refreshOverlayVisibility()
                // **소스가 최소화되면 캡처를 정지한다 (사용자 지정, 2026-09-01).**
                // 최소화 중에도 계속 돌리려던 앞선 두 시도(추적 동결 + 오버레이 숨김)는
                // 최소화 창의 기하가 독 타일로 바뀐다는 사실과 계속 싸워야 했다 — 커버가 독
                // 아이콘 위로 옮겨가고, 캡처가 타일 크기로 재설정돼 복원 직후 화면이 찌그러졌다.
                // 정지하면 그 상태가 아예 존재하지 않는다. 복귀는 ⌃⌘Z(캡처 토글)로 사용자가 한다.
                // 기하 방어선(OverlayManager)은 AX가 최소화를 놓치는 경우를 위해 남겨둔다.
                if self.overlayManager?.sourceIsMinimized == true {
                    DiagnosticLog.shared.log("[CAPTURE] 소스 최소화 → 정지")
                    await self.stopCapture()
                    return
                }
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
                       !self.isRestartingCapture, !self.captureSwitchInFlight, self.captureRegion == nil, self.stablePoolWidth > 0,
                       let src = self.overlayManager?.sourcePixelSize {
                        // 캡처 배율(MACFG_CAPSCALE)이 걸려 있으면 배달 프레임(=풀)은 소스 픽셀의
                        // 배율 크기다 — 소스 원본과 직접 비교하면 구조적으로 영원히 불일치라서
                        // ~1초마다 재구성→전체 리셋 루프에 빠진다(리뷰 확정). 기대 크기로 비교한다.
                        let capScale = min(max(Knob.double("MACFG_CAPSCALE") ?? 1.0, 0.25), 1.0)
                        let expectedW = Int(Double(src.width) * capScale)
                        let expectedH = Int(Double(src.height) * capScale)
                        let mismatch = abs(expectedW - self.stablePoolWidth) > 8 || abs(expectedH - self.stablePoolHeight) > 8
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
        guard !isStoppingCapture else { return }
        isStoppingCapture = true
        defer { isStoppingCapture = false }
        // 플래그를 await 이전에 먼저 내림 — 아래 stopCapture await 중 화면 파라미터 변경이
        // handleScreenParametersChange(399행, isCapturing 가드)로 detach된 링크를 재부착해
        // 좀비 렌더 루프를 만들던 것 차단 (리뷰 확정). 진행 중 configurePairEngine도 무효화.
        isCapturing = false
        isCapturingMirror = false
        captureManager.onFrameAvailable = nil   // 콜백 인제스트 즉시 차단
        configureEpoch += 1
        // 전체화면 동안의 배치는 이번 캡처 한정 — 원복 안 하면 다음 캡처가 뷰어로 잘못 시작한다.
        // **조건 없이 정산한다.** 예전엔 `if autoFsViewer`로 감쌌는데, 업스케일 ON으로 시작하면
        // 배치가 처음부터 뷰어라 AUTOFS가 전환할 일이 없어 그 플래그가 false로 남았고, 그러면
        // 전체화면 중에 업스케일을 끈 경우의 배치가 정산되지 않은 채 다음 캡처로 넘어갔다.
        // 핀이 있으면 그것이 이긴다 — 없으면 사용자 설정(upscaleMode)에서 재유도.
        selectedOverlayPlacement = placementPin ?? ((upscaleMode == .off) ? .coverSource : .viewerWindow)
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
        let visionGen = uiDetector?.generation ?? 0   // 인코딩 시점 세대 — 리셋 후 제출 차단용
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
                det.submitTextBoxes(rects, generation: visionGen)
                DiagnosticLog.shared.log("[UISTATIC] vision \(rects.count)개 텍스트 박스")
            }
        }
    }


    /// 렌더 전용 상태를 만지는 곳에서 스레드를 확인한다. **트랩하지 않고 로그만 남긴다** —
    /// 크래시는 이미 나고 있고, 필요한 건 "누가 범인인가"이지 또 한 번의 크래시가 아니다.
    ///
    /// 배경: stablePool / timeline / prevStable 은 락 없이 렌더 스레드 전용이라는 전제로 쓰인다.
    /// 그 전제가 깨지면 배열 버퍼가 교체되는 사이 옛 버퍼를 인덱싱해 **이미 해제된 텍스처를
    /// retain** 하게 되고, 그게 실제 크래시로 관측됐다(2026-08-05, 좀비 확인:
    /// `-[AGXG16GFamilyTexture retain]: message sent to deallocated instance`).
    /// 크래시 지점은 acquireStableTexture와 MetalFlowEngine.encodePair 두 곳으로 서로 다른데,
    /// 이는 특정 배열의 버그가 아니라 **공유 상태에 대한 레이스**라는 신호다.
    nonisolated private func checkRenderThread(_ site: String) {
        guard raceCheckEnabled else { return }
        let name = Thread.current.name ?? ""
        guard name != "MacFG.Render" else { return }
        // **레이트 리밋.** 스택 수집(callStackSymbols)은 싸지 않다. "위반은 드물다"는 전제는
        // 당시 알려진 0.4% 비율에 맞춘 것이고, 미지의 고빈도 위반에서는 이 계측 자체가 앱을 세운다.
        // 처음 3회 + 이후 600회마다만 남긴다 (RenderDriver의 폐기 로그와 같은 공식).
        let n = raceHits.withLock { $0[site, default: 0] += 1; return $0[site] ?? 0 }
        guard n <= 3 || n % 600 == 0 else { return }
        // 호출 경로를 남긴다 — 정적 호출부는 둘 다 렌더 스레드 클로저 안이라
        // 코드만 읽어서는 이 경로를 못 찾는다. 위반은 드물어 스택 수집 비용이 무해하다.
        let stack = Thread.callStackSymbols.prefix(14).map { $0.split(separator: " ").dropFirst(3).prefix(6).joined(separator: " ") }
        DiagnosticLog.shared.log("[RACE] \(site) — 렌더 스레드가 아님: '\(name.isEmpty ? "(무명)" : name)' main=\(Thread.isMainThread)\n  " + stack.joined(separator: "\n  "))
    }
    /// 레이스 검사 활성 여부. **개발 로그 토글에도 묶는다.**
    /// 예전엔 `MACFG_DIAGBUILD` 환경변수 전용이었는데, 환경변수는 Finder 더블클릭으로 켠 .app에
    /// 전달되지 않는다 — 즉 검사기가 있는데 **실사용 빌드에서 구조적으로 도달 불가**였다.
    /// 같은 함정을 stageDbg가 이미 겪고 UserDefaults 폴백으로 고쳤다(위 주석 참조).
    /// site별 위반 횟수 — 스택 수집 레이트 리밋용
    @ObservationIgnored private let raceHits = OSAllocatedUnfairLock(initialState: [String: Int]())
    @ObservationIgnored nonisolated(unsafe) private var raceCheckEnabled =
        Knob.string("MACFG_DIAGBUILD") == "1" || UserDefaults.standard.bool(forKey: "s.devlog")

    nonisolated private func resetScheduler() {
        // 드라이버가 안 돌 때는 이 상태를 만지는 스레드가 우리뿐이라 메인 호출이 **의도된** 것이다
        // (startCapture의 else 분기). 그 정상 경로를 위반으로 찍으면 캡처를 시작할 때마다
        // [RACE]가 나와서, 진짜 위반이 섞여 들어와도 아무도 안 보게 된다 — 검사기를 켜는 것보다
        // 나쁜 결과다. 그래서 드라이버가 도는 동안에만 검사한다.
        if renderDriver.isRunning { checkRenderThread("resetScheduler") }
        uiDetector?.reset()   // 불연속(재시작/리사이즈) — 정지-UI 누적도 리셋
        timeline = []
        inFlightTextures = [:]
        presentingTextures.withLock { $0.removeAll() }
        stablePool = []
        stablePoolWidth = 0
        stablePoolHeight = 0
        prevStable = nil
        lastPresentedTimestamp = 0
        lastPresentedTexture = nil
        lastAcceptedTimestamp = 0
        stageLock.lock(); shownContentTs = 0; shownWallTs = 0; stageLock.unlock()
        acceptedTsBeforeLast = 0
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
        stageLock.lock(); stgWaitEMA = 0; stageLock.unlock()   // 세션 경계에서 대기 통계 초기화
        renderGen &+= 1   // 이전 세대의 늦은 완료를 이후 drain이 걸러낸다
        _ = mailbox.drain(current: renderGen)
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
    @ObservationIgnored nonisolated(unsafe) private var pllDumpCount = 0
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
    /// 렌더 세대 — resetScheduler마다 증가(렌더 스레드 전용 쓰기). 인코딩 시점에 캡처돼
    /// 완료 핸들러의 postCompleted/postPresented에 실리고, drain이 불일치를 버린다.
    @ObservationIgnored nonisolated(unsafe) private var renderGen: UInt64 = 0
    /// 장면 전환 감지(임의 스레드의 cb2 완료) → 렌더 틱이 소비해 엔진 prior/UI 누적을 리셋.
    private let pendingSceneCutReset = OSAllocatedUnfairLock(initialState: false)
    private var lastResizeCheck: CFTimeInterval = 0
    /// 인제스트 이월 큐 — 버스트 틱(숨김 해제/재개 직후 최대 8장)의 인코딩 CPU가
    /// vsync 콜백을 삼키지 않게 틱당 4장 캡, 나머지는 다음 틱에서 처리
    @ObservationIgnored nonisolated(unsafe) private var pendingIngest: [FrameSlot] = []
    /// 인플라이트 present 수 (presentedHandler에서 감소 — 임의 스레드라 락 보호).
    /// 인플라이트 present 수 (presentedHandler에서 감소). 드로어블 포화 진단용 (drawBusy).
    private let inFlightPresents = OSAllocatedUnfairLock(initialState: 0)
    /// **표시 커맨드 버퍼가 아직 읽고 있는 스테일 텍스처** — 완료까지 참조를 붙잡는다.
    ///
    /// 예전엔 `lastPresentedTexture` 한 장만 "사용 중"으로 지켰다. 그런데
    /// maximumDrawableCount=3이라 present는 동시에 2장 이상 떠 있고(바로 위 diagPresentBusy가
    /// 그걸 센다), 그러면 **앞선 present가 GPU에서 아직 읽는 텍스처가 busy 판정에서 빠진다.**
    /// 풀이 그걸 다시 내주면 cb1의 blit이 읽는 중인 텍스처에 새 프레임을 덮어쓴다
    /// — 화면에 프레임이 겹쳐 보이거나 멈춘 듯 보이는 증상의 원인.
    /// (인플라이트 텍스처가 조기 해제되던 버그를 고치자 이 결함이 드러났다: 예전엔 죽은
    ///  ObjectIdentifier가 주소를 재사용한 새 텍스처와 충돌해 우연히 busy로 잡히면서
    ///  이 경로를 가려주고 있었다.)
    private let presentingTextures = OSAllocatedUnfairLock(initialState: [ObjectIdentifier: (tex: any MTLTexture, count: Int)]())
    @ObservationIgnored nonisolated(unsafe) private var diagPresentBusy = 0
    @ObservationIgnored nonisolated(unsafe) private var isRestartingCapture = false
    @ObservationIgnored nonisolated(unsafe) private var presentedTimes: [CFTimeInterval] = []
    /// 표시되지 못한 present 수 (drawable.presentedTime == 0) — 창마다 리셋.
    @ObservationIgnored nonisolated(unsafe) private var diagPresentDropped = 0
    /// present 위상 계측 — 표시 시각이 링크가 준 슬롯에서 몇 슬롯 밀렸는가(0/1/2/3+).
    /// 미표시(presentedTime==0)의 원인이 "뒤 present에 추월당함"인지 가르는 지표다.
    @ObservationIgnored nonisolated(unsafe) private var diagSlipHist = [0, 0, 0, 0]
    /// 링크가 같은 표시 슬롯을 연속으로 배달한 횟수 — 크면 원인이 present가 아니라 틱 쪽이다.
    @ObservationIgnored nonisolated(unsafe) private var diagDupTargetSlot = 0
    @ObservationIgnored nonisolated(unsafe) private var diagLastTargetSlot = -1
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
        let maxSlots = Knob.string("MACFG_MAXLAT").flatMap { Double($0) } ?? 4.0
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
    /// 수용된 프레임의 변화 크기 분포 — [<0.1%, <0.5%, <2%, <10%, >=10%].
    ///
    /// 지문이 "바뀌었다"고 답한 프레임들이 실제로 **얼마나** 바뀌었는지 본다. 앞쪽 칸이 크면
    /// 영상이 아니라 UI 리페인트(캐럿·스크롤바·크롬)를 소스 프레임으로 세고 있다는 뜻이고,
    /// 그게 입력 케이던스를 모니터 주사율 쪽으로 부풀려 보간 배수를 무너뜨린다는 가설의 증거다.
    /// 판단은 데이터를 보고 — 이 세션에서 이미 여섯 개 가설이 추측만으로 죽었다.
    @ObservationIgnored nonisolated(unsafe) private var diagChangeHist = [0, 0, 0, 0, 0]
    /// 부족분 크레딧 — 쌍마다 perPair씩 쌓고 실제로 낸 장수만큼 차감한다.
    /// 정수 절단으로는 표현 못 하는 비정수 배율(60→144 = 쌍당 1.4장)을 내기 위한 것.
    @ObservationIgnored nonisolated(unsafe) private var overSupplyCredit: Double = 0
    /// 소스 케이던스 고정으로 걸러낸 프레임 수 (영상 프레임 사이에 낀 UI 갱신).
    @ObservationIgnored nonisolated(unsafe) private var diagSrcLockSkip = 0
    /// 복원 규칙으로 통과시킨 장수 (E1a) — 거부 앞항만 참인 경우.
    @ObservationIgnored nonisolated(unsafe) private var diagSrcRestore = 0
    /// 게이트 **앞** 원시 도착 간격 분포 (E1a): <4 / <8 / <13 / <20 / >=20 ms.
    @ObservationIgnored nonisolated(unsafe) private var diagRawIntHist = [Int](repeating: 0, count: 5)
    /// 마지막 수용분 **직전** 수용분의 타임스탬프 — 케이던스 게이트의 복원 판정(2슬롯 규칙)용.
    @ObservationIgnored nonisolated(unsafe) private var acceptedTsBeforeLast: CFTimeInterval = 0
    /// UI 전용 갱신(영상 미진행)으로 판정해 걸러낸 프레임 수.
    @ObservationIgnored nonisolated(unsafe) private var diagUiGateSkip = 0
    /// 연속으로 건너뛴 수 — 4장에서 강제 수용해 정지를 구조적으로 막는다.
    @ObservationIgnored nonisolated(unsafe) private var uiGateStreak = 0
    /// present용 GPU가 목표 표시 슬롯 대비 몇 칸 늦게 끝났나 — [-1칸(여유), 0칸(정시), +1, +2, +3이상].
    /// stageLock으로 보호(완료 핸들러는 임의 스레드).
    @ObservationIgnored nonisolated(unsafe) private var diagGpuLateHist = [0, 0, 0, 0, 0]
    @ObservationIgnored nonisolated(unsafe) private var diagTsRejectCount = 0        // 타임스탬프 비전진으로 스킵 (중복 프레임 재전송)
    @ObservationIgnored nonisolated(unsafe) private var diagPresentCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagInterpPresentCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagPoolExhaustCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagInterpEncodedCount = 0
    @ObservationIgnored nonisolated(unsafe) private var diagFrameTypes: [String] = []
    @ObservationIgnored nonisolated(unsafe) private var diagSrcIntMin: Double = .infinity  // 콘텐츠 간격 min/max (VFR 판별)
    @ObservationIgnored nonisolated(unsafe) private var diagSrcIntMax: Double = 0
    /// 원본 도착 간격 분포 [<8, 8~13, 13~20, 20~27, 27+]ms — 소스가 정말 균일한지 판별
    @ObservationIgnored nonisolated(unsafe) private var diagSrcIntHist = [0, 0, 0, 0, 0]
    @ObservationIgnored nonisolated(unsafe) private var diagDrainDepthSum: Int = 0         // 매 틱 drain한 프레임 수 (버스트 판별)
    @ObservationIgnored nonisolated(unsafe) private var diagDrainDepthMax: Int = 0
    @ObservationIgnored nonisolated(unsafe) private var diagDrainSamples: Int = 0
    // 보간 스킵 사유별 카운터 (interpEnc=0 재발 시 원인 특정)
    @ObservationIgnored nonisolated(unsafe) private var diagSkipToggleOff = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipEngineNil = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipNoPrev = 0
    @ObservationIgnored nonisolated(unsafe) private var diagSkipContentFast = 0
    /// 버스트 판정을 받았지만 여유가 있어 **스킵 대신 한 장으로 구제**된 쌍 수 (fastCap:N)
    @ObservationIgnored nonisolated(unsafe) private var diagFastCapped = 0
    /// 갭 확장(steps>1)이 원본 도착 간격으로도 뒷받침되는 경우 = 프레임이 실제로 빠졌다
    @ObservationIgnored nonisolated(unsafe) private var diagGapExpandReal = 0
    /// 갭 확장이 스냅에서만 나온 경우 = 제때 온 프레임을 PLL이 밀어냈을 가능성
    @ObservationIgnored nonisolated(unsafe) private var diagGapExpandSnap = 0
    /// 쿼터/상한이 의도적으로 비운 쌍 (정상 동작 — 손실 아님)
    @ObservationIgnored nonisolated(unsafe) private var diagSkipQuota = 0
    /// 어느 생성 경로도 t를 못 낸 쌍 (진짜 손실 — 여기가 0이어야 한다)
    @ObservationIgnored nonisolated(unsafe) private var diagSkipNoT = 0
    /// t 결정 진단 — "보간 0장"이 **어느 가지에서** 나왔는지 가른다.
    /// mult: 정수배가 0장을 내 비정수 경로로 넘긴 횟수 (절벽 조건에 들어왔다는 뜻)
    /// grid: 비정수 경로까지 갔는데도 0장인 횟수 (여기가 진짜 막힌 곳)
    /// ratioMin/Max: 소스간격/슬롯간격 비율 관측 범위 — 절벽(1.5 근처)에 걸리는지 직접 보여준다
    @ObservationIgnored nonisolated(unsafe) private var diagTMultFell = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTGridEmpty = 0
    @ObservationIgnored nonisolated(unsafe) private var diagTRatioMin = 999.0
    @ObservationIgnored nonisolated(unsafe) private var diagTRatioMax = 0.0
    /// 표시 상한을 넘어 생산을 줄인 횟수 (부족분 상한이 발동한 쌍 수)
    @ObservationIgnored nonisolated(unsafe) private var diagTOverSupply = 0
    /// 쌍당 1장 미만일 때 몇 쌍마다 한 장을 낼지 세는 카운터
    /// 인제스트 블록이 이미 렌더 런루프에 예약돼 있는가 — 프레임마다 깨우는 것을 합친다.
    @ObservationIgnored private let ingestScheduled = OSAllocatedUnfairLock(initialState: false)
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
    // **표시 확정 기준 콘텐츠 전진.** presentedHandler(임의 스레드)에서만 갱신 — stageLock 보호.
    // 인코딩 시점(surface.encode 직후)에 재던 예전 방식은 두 가지를 못 봤다:
    //   ① 스캔아웃되지 못한 present도 균일한 8.3ms로 집계된다 (미표시는 diagPresentDropped로
    //      따로 세질 뿐, 콘텐츠 간격 표본에는 8.3이 두 개 들어가 σ가 낙관적으로 나온다)
    //   ② 그 프레임이 화면에 **얼마나 오래 걸려 있었는지**를 아예 무시한다
    // ②가 본질이다. 눈이 읽는 양은 "표시 프레임당 콘텐츠 전진"이 아니라 **벽시계 시간당
    // 콘텐츠 전진**이다 — 홀드가 길어지면 그동안 영상이 그냥 멈춰 있는 것이기 때문이다:
    //   콘텐츠 8.3ms / 벽시계 6.9ms = 1.20 (정상)
    //   콘텐츠 8.3ms / 벽시계 27.8ms = 0.30 (모션이 4배 느려짐 = 눈에 보이는 히치)
    // 실측(2026-08-26): 홀드 ≥3틱이 초당 4.2회인데 content σ에는 한 톨도 안 잡혔다.
    @ObservationIgnored nonisolated(unsafe) private var shownContentTs: CFTimeInterval = 0
    @ObservationIgnored nonisolated(unsafe) private var shownWallTs: CFTimeInterval = 0
    /// 콘텐츠 전진 / 벽시계 전진. 1.0이 정속, <1이면 그 구간 모션이 느려진 것.
    @ObservationIgnored nonisolated(unsafe) private var diagMotionRates: [Double] = []
    /// 모션 레이트 분포. 임계값 하나로는 안 된다 — 0.6 임계는 하필 **정상 홀드-2**(8.3/13.9
    /// = 0.597)와 겹쳐 구조적 동작을 히치로 오인했다(실측 stall 18/창 = 10.8/s, 대부분 정상).
    /// 칸: <0.5 / 0.5~0.8 / 0.8~1.1 / 1.1~1.4 / ≥1.4.
    /// 120@144의 구조적 바닥은 0.60(20%)과 1.20(80%) 두 칸에만 몰리는 모양이고,
    /// 144@144가 되면 1.00 한 칸(가운데)으로 수렴한다. 칸이 퍼지면 그게 진짜 불균일이다.
    @ObservationIgnored nonisolated(unsafe) private var diagMotionHist = [0, 0, 0, 0, 0]
    /// **원시 시계열 덤프 (MACFG_TSDUMP=1 → /tmp/MacFG_ts.csv).**
    /// [SCHED]의 1.7초 평균으로는 1Hz 안팎의 맥놀이를 볼 수 없고(에일리어싱), hold 문자열은
    /// 마지막 32장(0.27초)뿐이라 주기 추정에 못 쓴다. 사용자 제보 "프레임이 몰렸다 벌어졌다
    /// 하는 사인파 같은" 것과 최적 주기(p3~p6)가 창마다 옮겨다니는 로그가 모두 맥놀이를
    /// 가리키므로, 자기상관을 돌릴 수 있는 연속 표본이 필요하다.
    /// 핸들러(임의 스레드)에서 쌓고 [SCHED] 틱(렌더 스레드)에서 비운다 — stageLock 공유.
    @ObservationIgnored nonisolated(unsafe) private let tsDumpEnabled = Knob.string("MACFG_TSDUMP") == "1"
    @ObservationIgnored nonisolated(unsafe) private var diagTsDump: [(Double, Double)] = []
    // 적응형 지연 A/B용 (MACFG_NO_ADAPT=1이면 extraLatencySlots 0 고정 — 회귀 판별)
    private let adaptDisabled = Knob.string("MACFG_NO_ADAPT") != nil

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
            } else {
                // 스킵이 아닌 간격 = 진짜 패널 주기. 공칭 근방만 받아 EMA로 다듬는다.
                let nominal = 1.0 / max(mirrorRefreshRate, 30)
                if dt > nominal * 0.9 && dt < nominal * 1.1 {
                    measuredTickEMA = measuredTickEMA > 0 ? measuredTickEMA * 0.99 + dt * 0.01 : dt
                }
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
            let (_, released, _) = mailbox.drain(current: renderGen)
            for id in released { inFlightTextures.removeValue(forKey: id) }
            _ = captureManager.drainFrames()   // 파이프 적체 방지 (텍스처는 풀로 회수)
            pendingIngest = []
            return
        }

        // 1) 완료된 GPU 작업 수거 → 타임라인 등재
        // [10] 장면 전환이 감지됐으면(cb2 완료 스레드가 표시) 시간적 prior와 UI 누적을 리셋 —
        // 컷 쌍의 쓰레기 flow가 다음 장면 첫 쌍의 시드/블렌드로 들어가는 것을 막는다(리뷰 확정).
        // 여기는 렌더 스레드라 MetalFlow.reset()의 무락 상태도 안전하다.
        if pendingSceneCutReset.withLock({ was in let v = was; was = false; return v }) {
            pairEngine?.reset()
            uiDetector?.reset()
        }
        let (newEntries, released, presented) = mailbox.drain(current: renderGen)
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
            // presentedAt == 0 = **우리가 그렸는데 화면에 한 번도 안 나간 프레임.**
            // 통계에서 빼는 것만으로는 부족하다 — 이게 몇 장인지가 곧 "왜 120이 안 나오는가"의
            // 답이다. 세지 않으면 glass 간격만 보고 "표시가 20ms 균일하다"고 읽게 되는데,
            // 실제로는 present를 10ms마다 하고 그중 절반이 버려지는 상태일 수 있다.
            guard record.presentedAt > 0 else { diagPresentDropped += 1; continue }
            paceShown += 1   // 적응 페이싱의 목적함수 — **실제로 화면에 나간** 장수만 센다
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
        // MACFG_PRESENTEVERY=N: N틱마다 한 번만 present. **측정 전용.**
        // 가르려는 것: 표시 실패(presentedTime==0)가 우리가 과잉 공급해서인가, 컴포지터의
        // 외부 천장인가. 공급을 절반으로 줄였을 때 표시 fps가 오르면 전자, 그대로면 후자다.
        // 적응 페이싱 — 표시 프레임 수를 목적함수로 공급 배율을 언덕오르기. 0.8초마다 평가한다
        // (짧으면 콘텐츠 변동을 개선으로 오독하고, 길면 반응이 굼뜨다).
        if paceAdaptive {
            let now = CACurrentMediaTime()
            if paceWindowStart == 0 { paceWindowStart = now }
            let span = now - paceWindowStart
            if span >= 0.8 {
                let rate = Double(paceShown) / span
                if paceLastRate >= 0 {
                    // 유의미한 악화일 때만 방향을 뒤집는다 — 잡음에 끌려 진동하지 않도록.
                    if rate < paceLastRate - 1.5 { paceDir = -paceDir }
                }
                paceLastRate = rate
                paceLastShownRate = rate
                paceScale = min(1.0, max(0.55, paceScale + paceDir))
                // 상/하한에 닿으면 방향을 되돌린다 (끝에 붙어 정체하지 않도록).
                if paceScale >= 1.0 || paceScale <= 0.55 { paceDir = -paceDir }
                paceShown = 0
                paceWindowStart = now
            }
        }
        var paceAllows = true
        if paceAdaptive, paceScale < 0.999 {
            // 위상 누산기 — 남기는 present가 균등 간격이 되도록. (매 N틱 스킵은 배율이
            // 정수 역수가 아닐 때 뭉친다.)
            pacePhase += paceScale
            if pacePhase >= 1.0 { pacePhase -= 1.0 } else { paceAllows = false }
        }
        if paceAllows, presentEveryN <= 1 || diagTick % presentEveryN == 0 {
            presentDueEntry(targetTimestamp: targetTimestamp, drawable: drawable)
        }
        // 링크 재부착 직후 강제 재present — 정적 콘텐츠는 새 프레임이 없어 presentDueEntry가
        // 아무것도 표시하지 않으므로, 재부착 전 그려둔 흐린(960×540 드로어블) 프레임이 남는다.
        // 이번 틱에 새 present가 없었으면 최신 텍스처를 새(큰) 드로어블에 다시 그려 교체한다.
        // MACFG_ALWAYSPRESENT=1: 새 프레임이 없어도 매 틱 재present한다. **측정 전용.**
        // 가르려는 것: 틱이 굶는 원인이 보간 GPU 작업인가, present 레이트 자체인가.
        // 바이패스(보간 0)에 이걸 켜면 GPU 부하는 그대로인데 present만 100%가 된다.
        if forceRepresentTicks > 0 || alwaysRepresent {
            if forceRepresentTicks > 0 { forceRepresentTicks -= 1 }
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
        checkRenderThread("drainAndIngest")
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

        // **소스 케이던스 고정 (MACFG_SRCFPS, 0/미설정 = 끔).**
        //
        // 세상의 영상 소스는 24/25/30/50/60뿐인데, 우리가 재는 "소스율"은 그것이 아니라
        // **창 표면이 갱신된 횟수**다. 마우스를 움직이면 소스 앱이 호버 리페인트로 영상 프레임
        // 사이에 갱신을 끼워 넣어 측정 소스율이 92~110fps로 뛴다(실측). 그 프레임들은 가짜가
        // 아니다 — 진짜로 픽셀이 바뀐다(지문 변화율로 확인: 수용분의 93.7%가 화면 10% 이상 변화).
        // 문제는 진위가 아니라 **간격**이다: 쌍 간격이 16.7ms에서 9ms로 좁아지면 표시 슬롯
        // 기준 2.4슬롯이 1.3슬롯이 되고, 원본 양보 구간을 빼면 보간을 놓을 자리가 사라진다.
        // 그래서 보간 생성이 55 → 23장/s로 반토막 나고 출력만 떨어진다.
        //
        // 영상 케이던스를 알면 그 사이에 끼는 갱신은 받지 않으면 된다. 잃는 것은 영상 한 프레임
        // 안에서의 UI 변화가 한 프레임 늦게 보이는 것뿐이고, 얻는 것은 보간이 꺼지지 않는 것이다.
        // 임계를 0.75×로 두어 실제 케이던스의 지터(SCK 배달 흔들림)는 통과시킨다.
        // **UI 전용 갱신 게이트 (MACFG_UIGATE = 변화율 임계, 0/미설정 = 끔).**
        //
        // 소스 앱은 마우스가 움직이면 영상 프레임 사이에 창 갱신을 끼워 넣는다. 그 프레임들은
        // 진짜다(픽셀이 바뀐다) — 다만 **영상이 진행하지 않는다.** 그런데 우리는 그것을 새 소스
        // 프레임으로 세므로 쌍 간격이 16.7ms에서 9ms로 좁아지고, 표시 슬롯 기준 2.4 → 1.3슬롯이
        // 되어 보간을 놓을 자리가 사라진다. 결과는 영상 모션이 덜 매끄러워지는 것이다.
        //
        // 결정적 재현(2026-08-09, TestPattern --hover-repaint: 영상은 그대로 두고 작은 영역만 갱신):
        //   마우스 정지: 인식 53fps, 소스 48.0/s, 보간 생성 **83.6/s**, 표시 128.2/s
        //   마우스 회전: 인식 106fps, 소스 76.1/s, 보간 생성 **47.1/s**, 표시 121.3/s
        //   영상 모션을 나르는 프레임(진짜 소스 48 + 보간)이 131 → 95로 28% 줄어든다.
        //
        // 그런데 둘은 **변화 크기로 깨끗하게 갈린다**. 같은 실행의 지문 변화율 분포:
        //   마우스 정지: 변화 <0.1%인 프레임 0.0%
        //   마우스 회전: 변화 <0.1%인 프레임 **56.3%** (영상 프레임은 2~10% 구간)
        // 그래서 임계 아래 갱신은 받지 않는다. 잃는 것은 그 UI 변화가 영상 한 프레임만큼
        // 늦게 보이는 것뿐이고, 얻는 것은 보간 예산이 온전히 남는 것이다.
        //
        // 주의: 실제 브라우저의 호버 리페인트가 이 재현보다 넓은 영역을 다시 그릴 수 있다.
        // 임계를 너무 높이면 진짜 영상 프레임(작은 움직임의 정지 장면)까지 버린다 — 기본 0.2%는
        // 위 분포에서 두 무리 사이의 빈 구간이다.
        // **누적으로 판정한다 — 단순 임계는 정지 화면을 만든다.**
        // changeRatio는 *직전에 배달된* 프레임 대비 값이라, 게이트로 건너뛰면 그 차이가
        // 어디에도 남지 않는다. 그러면 아주 느린 장면(작은 물체만 움직이는 영상)은 매 프레임이
        // 임계 아래여서 **영원히 수용되지 않고 화면이 멈춘다.** 이 저장소는 지문을 랜덤 384점으로
        // 뒀을 때 같은 실패를 이미 겪었다(가로 드래그 5초 정지).
        // 누적하면 두 성질이 동시에 선다: 큰 변화(영상 진행)는 즉시 통과하고, 작은 변화는
        // 쌓여서 결국 통과하므로 정지가 원천적으로 불가능하다. UI 리페인트는 그 사이 대부분
        // 걸러진다(0.02%짜리가 10장 모여야 한 번 통과 = 배달의 90%가 사라진다).
        // 임계 판정 + **연속 스킵 상한**. 누적식(스킵된 변화율을 더해 임계에 도달하면 수용)도
        // 재봤으나 임계식보다 결과가 나빴다(표시 88.9 vs 131.3, σ 3.81 vs 2.39). 다만 임계식만
        // 두면 아주 느린 장면이 영원히 임계 아래여서 화면이 멈출 수 있다 — 이 저장소가 지문을
        // 랜덤 384점으로 뒀을 때 겪은 실패다(가로 드래그 5초 정지). 연속 스킵을 4장으로 묶어
        // 정지 시간을 소스 간격 4배(60fps면 67ms) 이내로 구조적으로 못박는다.
        // **기본 꺼짐 — 실사용 로그로 기각됐다 (2026-08-20).** 결정적 재현(30px 마커)에서는
        // UI 갱신이 변화 <0.1%에 몰려 크기로 갈렸지만, 실제 브라우저의 호버 리페인트는
        // 화면의 10% 이상을 다시 그린다(부풀림 구간 수용분의 65.6%가 ≥10% 칸 — 호버
        // 하이라이트·플레이어 컨트롤은 큰 영역이다). 그 세션에서 이 게이트는 부풀림 구간에서
        // 2.4장/s밖에 못 잡고 정상 구간에서 7.1장/s를 지연시켰다 — 신호가 틀렸다.
        // 영상 프레임과 리페인트를 가르는 것은 크기가 아니라 **도착 간격**이다(아래 SRCFPS 게이트).
        // 크기로 갈리는 콘텐츠를 위해 노브는 남긴다.
        let uiGateValue = Knob.double("MACFG_UIGATE") ?? 0
        if uiGateValue > 0,
           slot.changeRatio > 0, slot.changeRatio < Float(uiGateValue),
           uiGateStreak < 4 {
            uiGateStreak += 1
            diagUiGateSkip += 1
            return
        }
        uiGateStreak = 0

        // **기본 60 — 세상의 영상 소스는 24/25/30/50/60뿐이다** (사용자 지침: "맥에서 60프레임
        // 이상 소스를 입력으로 받을 일 없다"). 60fps 영상의 프레임 간격은 16.7ms이므로, 직전
        // 수용분에서 12.5ms(0.75×) 안에 또 오는 갱신은 영상 진행이 아니라 사이에 낀 UI
        // 리페인트다. 크기 게이트(위)와 달리 이 구분은 실사용에서도 성립한다 — 리페인트가
        // 아무리 넓은 영역을 다시 그려도 영상 케이던스 그리드 밖에 도착한다는 사실은 변하지 않는다.
        // 결정적 재현 실측(소스100fps 조건): multFell 46→0, 미표시 1.1%, tick 143.9.
        // 정지 화면을 만들 수 없는 구조다: 느린 장면은 애초에 간격이 넓어 전부 수용된다.
        // 120fps를 진짜로 받아야 하는 특수 소스는 노브로 올린다(0 = 끔).
        let srcFpsCap = Knob.double("MACFG_SRCFPS") ?? 60
        // 복원 문턱 (E1c) — 1.6은 2026-08-25에 PiP FHD 사례로 정한 하드코딩 값이었다.
        // 현행 창모드 수용 평균 간격이 14.5ms라 인접쌍 합의 평균이 29.0ms이고, 1.75슬롯
        // (29.17ms)은 그 평균 바로 위에 붙는다 — 조이면 절반쯤이 갈릴 자리라 A/B가 필요하다.
        let srcRestoreSlots = Knob.double("MACFG_SRCRESTORE") ?? 1.6
        if srcFpsCap >= 1, lastAcceptedTimestamp > 0 {
            let lockedInterval = 1.0 / srcFpsCap
            let sinceLast = slot.timestamp - lastAcceptedTimestamp
            // **복원 규칙 (2026-08-25).** 간격 하나만 보는 게이트는 SCK 배달 지터에 진짜
            // 프레임을 먹는다 — 직전 프레임이 늦게 배달되면 다음 프레임과의 간격이 12.5ms
            // 아래로 압축되는데, 그 프레임은 리페인트가 아니라 케이던스 **복원**이다.
            // 실측(PiP FHD 60fps 실사용): srcLock 50/s 중 지문 중복은 3/s뿐, 진짜 프레임이
            // 초당 ~7장 거부돼 인식이 53fps로 떨어지고 2× 출력이 106에 멈췄다.
            // 판정: 마지막 수용분 직전 것부터의 간격이 1.6슬롯 이상이면 두 프레임이 합쳐
            // 2슬롯 근처를 덮는다 = 지연 배달의 복원 → 허용. 리페인트 폭풍(지속 배가)은
            // 어느 쌍을 잡아도 2슬롯을 못 채우므로 여전히 걸러진다.
            let sinceBeforeLast = acceptedTsBeforeLast > 0
                ? slot.timestamp - acceptedTsBeforeLast : .infinity
            // **원시 도착 간격 히스토그램 (E1a).** 게이트 앞에서 센다 — 무엇이 들어오는지를
            // 봐야 무엇을 버리는지 판단할 수 있다. 창 캡처와 디스플레이 캡처의 서명이 갈리는
            // 지점이 여기다(실측: /win은 8~13ms 칸에 17.6~43.0%, /disp는 <8ms 칸에 11.9~22.6%).
            let rawMs = sinceLast * 1000.0
            let rb = rawMs < 4 ? 0 : rawMs < 8 ? 1 : rawMs < 13 ? 2 : rawMs < 20 ? 3 : 4
            diagRawIntHist[rb] += 1
            if sinceLast < lockedInterval * 0.75, sinceBeforeLast < lockedInterval * srcRestoreSlots {
                diagSrcLockSkip += 1
                return
            }
            // **복원 규칙으로 살아난 장수 (E1a).** 거부 조건의 앞항만 참인 경우 = 간격은
            // 좁은데 인접쌍 합이 문턱을 넘어 통과시킨 프레임이다. 이 값이 없으면 복원이
            // "얼마나 발동하는가"를 알 수 없고, 문턱(E1c)을 조일 근거도 생기지 않는다.
            if sinceLast < lockedInterval * 0.75 { diagSrcRestore += 1 }
        }
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
            // **원본 도착 간격 히스토그램.** min/max만으로는 "가끔 한 번 튄 것"과 "상시 흔들림"이
            // 구분되지 않는다. 사용자는 소스가 균일한 60fps라고 보고했는데 우리 계측은
            // 7~30ms 범위를 보고한다 — 둘 중 하나가 틀렸고, 분포를 보면 갈린다.
            // 60fps(16.7ms) 기준 칸: <8 / 8~13 / 13~20 / 20~27 / 27+ ms
            let dms = delta * 1000
            diagSrcIntHist[dms < 8 ? 0 : dms < 13 ? 1 : dms < 20 ? 2 : dms < 27 ? 3 : 4] += 1
        }
        let previousAcceptedTs = lastAcceptedTimestamp
        let previousBeforeLast = acceptedTsBeforeLast
        let previousAcceptedFingerprint = lastAcceptedFingerprint
        lastFrameArrivalAt = CFAbsoluteTimeGetCurrent()   // 좀비 오버레이 판정용 (프레임 공급 생존 신호)
        acceptedTsBeforeLast = lastAcceptedTimestamp      // 케이던스 게이트의 복원 판정 기준
        lastAcceptedTimestamp = slot.timestamp
        lastAcceptedFingerprint = slot.contentFingerprint
        performanceMonitor.recordFrameArrival()
        diagSourceCount += 1
        // 변화 크기 분포 누적 (계측 전용 — 동작에는 아직 쓰지 않는다)
        let cr = slot.changeRatio
        diagChangeHist[cr < 0.001 ? 0 : cr < 0.005 ? 1 : cr < 0.02 ? 2 : cr < 0.10 ? 3 : 4] += 1

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
            acceptedTsBeforeLast = previousBeforeLast   // 케이던스 복원 판정 기준도 함께 롤백
            diagSourceCount -= 1                        // 실패 프레임이 전달률 분자에 남지 않게
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
            acceptedTsBeforeLast = previousBeforeLast
            diagSourceCount -= 1
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
        // **소스 버퍼를 blit이 끝날 때까지 붙잡는다.**
        // slot.texture는 CVPixelBuffer의 IOSurface를 제로카피로 감싼 것인데, Metal이 붙잡는 건
        // IOSurface뿐이고 **CVPixelBufferPool의 재활용 판정은 CVPixelBuffer의 참조수**를 본다.
        // SCK 콜백이 반환되면 버퍼가 풀로 돌아가고, 인제스트→blit 실행까지 5~15ms가 걸리는데
        // 소스는 10ms마다 오므로, 그 사이 같은 표면에 다음 프레임이 쓰이면 blit 결과에 두 프레임이
        // 섞인다 — 보간을 꺼도 보이는 "프레임 중첩"의 구조적 원인.
        // 완료 핸들러가 클로저 컨텍스트로 버퍼를 잡고 있다가 GPU가 다 읽은 뒤에 놓아준다.
        if let srcBuffer = slot.pixelBuffer {
            cb.addCompletedHandler { _ in withExtendedLifetime(srcBuffer) {} }
        }
        // cb2를 **cb1 커밋 전에** 만든다. 커밋 후에 만들다 실패하면 이미 GPU가 stable에
        // 쓰는 중인데 이 함수는 그냥 반환해 다음 프레임이 같은 stable을 재획득할 수 있다
        // (프레임 중첩의 구조적 원인과 동일 부류 — 리뷰 확정). 커밋 전 실패면 cb1은
        // 버려지고 GPU 작업이 시작되지 않으므로 stable 재사용이 안전하다.
        guard let cb2 = workQueue.makeCommandBuffer() else {
            lastAcceptedTimestamp = previousAcceptedTs
            lastAcceptedFingerprint = previousAcceptedFingerprint   // 위와 동일 — 지문 롤백
            acceptedTsBeforeLast = previousBeforeLast
            diagSourceCount -= 1
            resetSnapState()
            return
        }
        cb.commit()
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
            // 공칭(1/144)이 아니라 **실측 주기**를 쓴다 — effectiveDisplayInterval 주석 참조.
            let displayInterval = effectiveDisplayInterval
            // 갭이 디스플레이 한 프레임보다 작으면 그 사이에 표시 슬롯이 없어 보간 무의미 —
            // 게다가 브라우저 버스트 배달(30fps인데 2프레임이 7ms로 붙어 옴)에서 이 퇴화 쌍을
            // 억지로 보간하면 ANE 과부하로 engFail·아티팩트가 난다(실측). 슬롯 없는 갭은 스킵해도
            // 구멍이 안 생기므로(보여줄 자리가 없음) 문턱을 한 프레임으로 올려 버스트를 걸러낸다.
            // **판정은 스냅된 갭이 아니라 원본 간격으로 한다.**
            // 이 가드의 의도는 "콘텐츠가 정말로 표시 슬롯보다 빠르면 보간이 무의미하다"인데,
            // gap은 PLL이 스냅한 값이라 **스냅이 빗나가면 콘텐츠와 무관하게 좁아진다.**
            // 실측(2026-08-26, 깨끗한 시스템 4K 60fps): snapMiss 4.15/s와 fast 4.20/s가
            // 정확히 일치했고, 같은 창에서 원본 delta의 최소는 7.0ms로 슬롯(6.94ms)보다 항상
            // 컸다 — 즉 6.94ms 미만 갭은 소스가 만든 것이 아니라 스냅 실패의 산물이다.
            // 그 오판으로 초당 4쌍의 보간이 통째로 건너뛰어졌다(생성 55.6 vs 수용 59.7).
            // 원본 간격으로 재면 버스트 배달(30fps인데 2프레임이 7ms로 붙는 것)은 여전히
            // 걸러지고, 스냅 실패한 정상 쌍은 살아난다.
            let rawGap = slot.timestamp - prev.rawTimestamp
            //
            // **이 게이트의 대가 (2026-08-26 확정, 다중 검증).**
            // 로그 37창에서 `fast` 카운터는 `dist[0]`(도착 간격 <8ms 칸)과 **정확히 같은 수**였고,
            // 표시 패턴 문자열의 `SS` 인접쌍 수와도 1:1이었다 (fast=0 구간에선 SS도 정확히 0).
            // 즉 fast 1회 = 보간이 통째로 빠진 S-S 1회다. 그런데 스냅의 steps 클램프
            // (`max(1.0, …)`)가 콘텐츠를 반드시 한 간격 전진시키므로, 그 쌍의 콘텐츠 스텝은
            // 좁은 7ms가 **아니라 16.6ms — 정상 8.3ms의 2배 히치**다. 초당 약 10회.
            // 회귀로 잰 몫은 content σ 2.40 중 0.2~2.0ms(중앙 1.0~1.5).
            //
            // **그런데 이게 전부가 아니다:** fast≈0인 구간에서도 σ가 2.12로 측정됐다. 절반 이상은
            // 아직 설명되지 않았으므로 이 게이트를 손대는 것만으로 균일성이 해결되지 않는다.
            //
            // 판정 기준 자체가 두 오판 사이에 끼어 있다:
            //   gap(스냅)   — 스냅이 빗나가면 콘텐츠와 무관하게 좁아진다 (711b429가 고친 방향)
            //   rawGap(도착) — 배달이 뭉치면 진짜 60fps 쌍도 좁아 보인다 (지금 이 대가)
            // 그래서 A/B 다이얼로 연다. 기본 1 = 현행(rawGap).
            //   0 = 게이트 해제 (버스트 쌍도 보간 — ANE 부하 +16%, 성능 축 재개 위험)
            //   1 = 현행: 도착 간격이 좁으면 보간 0장
            //   2 = 논리곱: 도착과 스냅갭이 **둘 다** 좁을 때만 스킵 (진짜 버스트만 거른다)
            //   3 = 상한 1장: 스킵 대신 t=[0.5] 한 장만 — 16.6ms 히치를 8.3×2로 가르되
            //       ANE 과부하 방지 의도(69e7808)는 "쌍당 최대 1장"으로 보존
            // 검증지표는 fast 카운트가 아니라 **표시 패턴의 SS 수와 motion σ/stall**이다 —
            // fast는 도착 간격의 동어반복이라 게이트가 고쳐져도 그대로 남는다.
            //
            // **기본값은 고정이 아니라 엔진 여유로 정한다 (2026-08-27 실측).**
            // 같은 게이트가 엔진에 따라 정반대 역할을 한다:
            //   RIFE 4K      work 26~50ms / wait 13~22ms → ANE 포화. 게이트를 열자
            //                engFail 1~3/창 → 14~28/창, 미표시 0~5 → 16~46/창,
            //                생성은 오히려 47 → 45/s로 **감소**. 게이트가 부하 보호였다.
            //   MetalFlow 4K work  4~6ms / wait  1~2ms  → 유휴. 게이트를 열자
            //                생성 52 → 60/s, 표시 111 → 119/s, work 4 → 5ms, engFail 0.
            //                게이트가 순수 손실이었다.
            // 그래서 "켜냐 끄냐"가 아니라 **여유가 있느냐**가 옳은 질문이다.
            // **판정은 거버너 신호로 한다 — work으로 하면 안 된다 (2026-08-30 실측으로 정정).**
            //
            // 처음엔 `paceWorkP90 < 소스간격`으로 판정했는데 **틀렸다.**
            // work은 파이프라인 **지연**이지 점유율이 아니고, RIFE는 그 대부분이 ANE 대기다.
            // 위 348행 주석이 이미 그 분해를 기록해 뒀는데(대기 0.32 vs 16.28, GPU 합은 둘 다
            // ≈4.5ms) 그걸 안 보고 work을 부하로 읽었다. 결과: 여유가 있는데 없다고 판정해
            // 초당 7장(RIFE)·3장(AppleFI)의 보간을 버렸다.
            //
            // 같은 소스(4K 60fps AV1) 3엔진 A/B, 전환 구간 제외 26~34창:
            //             게이트(work 판정)        구제(강제)              engFail
            //   AppleFI   116.6/s sd 4.18  →  **119.7/s sd 0.93**  생성 56.6→59.9   0
            //   RIFE      113.5/s sd 2.15  →  **121.2/s sd 1.95**  생성 53.7→61.1   0
            //   MetalFlow 119.9/s sd 0.82 (이미 구제 모드라 변화 없음)
            // 표시율이 오르면서 **안정성도 같이 좋아졌고**(AppleFI sd 4.18→0.93),
            // work은 1ms만 올랐다(10.0→10.9, 14.7→15.9). 셋 다 ×2 상한 119.8에 도달.
            // 실제 점유는 셋 다 절반 미만이었다 — RIFE ANE 42%, MetalFlow 12~22%.
            //
            // **판정은 거버너의 t 개수 정책으로 한다.**
            //
            // 두 번 틀렸다. ① `paceWorkP90 < 소스간격` — work은 지연이지 점유가 아니다(위 참조).
            // ② `gapExpansionAllowed` — 이건 두 이유로 부적합했다:
            //    · `level == .full`만 통과하는데 거버너는 그 자리에 머물지 않는다(GOV 로그가
            //      800↔640을 계속 오간다)
            //    · 그리고 `&& srcSpreadMs < 20`이 붙어 있는데, 그건 **배달 지터** 신호다.
            //      브라우저 버스트 배달은 srcInt [7~28] = 스프레드 21ms라 상시 문턱을 넘는다.
            //      연산 여유와 아무 상관이 없다.
            //    실측 결과: 가장 가벼운 MetalFlow(work 4.0ms)에서 게이트가 닫혀
            //    생성 60.0 → 50.6, 표시 119.9 → 110.2으로 **회귀했다.**
            //
            // 옳은 기준: **구제는 정확히 1장을 만든다.** 거버너는 쌍당 몇 장이 괜찮은지를
            // tCountCap으로 이미 말한다 — .full=제한없음 / .light=2 / .heavy=1 / .bypass=0.
            // 즉 .heavy에서조차 1장은 정책상 허용이고, 막아야 할 것은 .bypass(0장)뿐이다.
            // 판정 단위(1장)와 신호 단위(장수)가 정확히 일치하는 유일한 기준이다.
            // 2026-08-27에 강제 구제가 engFail을 10배로 터뜨렸던 조건(RIFE 사다리 288↔432
            // 진동, work 26~50ms)에서는 거버너가 내려앉으므로 보호가 유지된다.
            let fastGuardMode = Knob.int("MACFG_FASTGUARD") ?? ((tCountCap ?? 1) >= 1 ? 3 : 1)
            let rawFast = (rawGap > 0 ? rawGap : gap) < displayInterval
            let snapFast = gap < displayInterval
            let contentAlreadyFast: Bool
            switch fastGuardMode {
            case 0: contentAlreadyFast = false
            case 2: contentAlreadyFast = rawFast && snapFast
            default: contentAlreadyFast = rawFast   // 1(기본), 3(아래에서 상한으로 처리)
            }
            let fastCapOne = fastGuardMode == 3 && rawFast
            if fastCapOne { diagFastCapped += 1 }
            if gap > 0 && gap < 0.25 && !(contentAlreadyFast && !fastCapOne)
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
                    // **갭 확장 자체를 끄는 노브 (MACFG_NOGAPEXP=1). 측정용.**
                    //
                    // 2026-08-31 실측: 더 많이 만드는 쪽이 **더 불균일하다.**
                    //   cover@fs/disp  인코딩 123.3 = 표시 123.3(미표시 0)  motion σ 0.30  content σ 1.80
                    //   cover/disp     인코딩 137.5 → 표시 129.2            motion σ 0.39  content σ 2.70
                    // 60fps ×2의 상한은 120인데 137.5를 만들고 있고, 그 초과분이 갭 확장에서 온다
                    // (steps=2 → count=2*2−1=3장). 138장을 143틱에 밀어넣으니 97% 포화라
                    // 8.3장이 컴포지터에서 버려지고, 남은 것도 간격이 고르지 않다.
                    // 사용자 제보("130까지 올라가면서 안 부드럽다")와 정확히 일치한다.
                    //
                    // 주의: 소스가 **진짜로** 프레임을 빠뜨렸을 때는 확장이 옳다(구멍이 생긴다).
                    // 그래서 기본값은 건드리지 않고 A/B로만 판정한다. 판정선은 표시 장수가 아니라
                    // **motion σ와 content σ**다 — 장수는 줄어도 리듬이 좋아지면 이기는 것이다.
                    if !gapExpansionAllowed || Knob.int("MACFG_NOGAPEXP") == 1 { steps = 1 }
                    // **갭 확장이 진짜인지 스냅 오판인지 가른다 (2026-08-30 계측).**
                    //
                    // 실사용 4K 60fps RIFE에서 생성이 소스(60/s)를 넘어 74~87/s까지 오르고
                    // 표시가 130/s를 찍는다(사용자 제보 "안 부드러움"). 원인은 여기다:
                    // 갭이 소스 간격의 2배로 잡히면 steps=2 → count=2*2−1=3장을 만든다.
                    // 도착 간격 히스토그램의 27ms+ 칸이 11~18%라 산술도 맞는다
                    // (59쌍 × 15% × +2장 ≈ +18장 → 60+18 = 78, 관측 범위 안).
                    //
                    // **문제는 그 갭이 진짜냐다.** 소스 프레임이 실제로 빠졌으면 3장으로 메우는 게
                    // 옳다(콘텐츠 케이던스 유지). 그냥 늦게 배달된 것을 PLL이 '빠졌다'로 스냅했다면
                    // 16.7ms 구간에 3장을 넣는 셈이라 콘텐츠 간격이 8.3 → 5.5ms로 좁아진다 —
                    // 그게 곧 불균일이고 체감 불편의 후보다.
                    //
                    // 판별: 원본 **도착** 간격이 확장을 뒷받침하는가. rawGap이 스냅된 갭의
                    // 70% 이상이면 진짜 늦게 온 것(=프레임이 실제로 없었다), 그보다 훨씬 짧으면
                    // 제때 왔는데 스냅이 밀어낸 것이다. 후자가 많으면 스냅 쪽을 고쳐야 한다.
                    if steps > 1 {
                        let supported = rawGap > 0 && rawGap >= gap * 0.7
                        if supported { diagGapExpandReal += 1 } else { diagGapExpandSnap += 1 }
                    }
                    let maxUseful = max(1, Int((interval / displayInterval).rounded()))
                    let m = min(mirrorFrameMultiplier, maxUseful)
                    let count = min(m * Int(steps) - 1, 8)
                    if count >= 1 {
                        // 균등분할. 주의: M×fps < 주사율이면 홀드가 2슬롯+ 이므로 위상 오차가
                        // 가끔 1/3슬롯 홀드로 튀는 양자화 지터(σ~3ms)는 불가피 — 표시 그리드
                        // 스냅도 개선 없음 실측 (소스 프레임이 소스 그리드에 있어 혼합 케이던스)
                        tValues = (1...count).map { Float($0) / Float(count + 1) }
                    }
                    if tValues.isEmpty { diagTMultFell += 1 }
                }
                // **정수배가 아무것도 못 내면 여기서 멈추지 않는다.**
                // maxUseful = round(interval / displayInterval)은 "디스플레이가 소스 프레임당 정수
                // 개수만 표시할 수 있다"고 가정하는데 사실이 아니다. 144Hz에서 소스 96fps면 실제로는
                // 프레임당 1.5장을 낼 수 있는데 반올림이 1이 되어 count = 1×1−1 = 0 —
                // **보간이 절벽처럼 완전히 꺼진다**(90fps 1장 → 96fps 0장).
                // 실사용 증상(제보 2026-08-07): 마우스를 움직이면 소스 창의 호버 리페인트로 측정
                // 소스율이 96fps 위로 올라가고 그 순간 보간이 죽어 **출력이 120 → 96fps로 떨어진다**
                // ("소스는 오히려 올라가는데 출력만 떨어진다"). 아래 비정수 경로는 60→144(쌍당 2.4슬롯)
                // 같은 조합을 이미 처리하므로, 정수배가 표현 못 하는 구간을 그쪽에 맡기면 절벽이 사라진다.
                if tValues.isEmpty {
                if gap / displayInterval > 1.5,
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
                    // **양보 구간 배율 (MACFG_YIELD, 기본 1.0 = 기존 동작).**
                    //
                    // 위 상수 0.4/0.6은 합쳐서 정확히 1슬롯을 원본에 양보한다. 그 1슬롯이
                    // 두 증상의 공통 원인이다:
                    //   60fps → 쌍 간격 2.4슬롯, 양보 후 1.4 → 보간 1장 → **출력이 120에서 멈춘다**
                    //           (144를 채우려면 쌍당 1.4장이 필요한데 양보 구간이 2번째를 막는다)
                    //   96fps → 쌍 간격 1.5슬롯, 양보 후 0.5 → **0장, 보간이 통째로 꺼진다**
                    // 후자가 사용자 제보의 정체다 — 마우스를 움직이면 소스 창 리페인트로 측정
                    // 소스율이 92~99fps로 뛰고(실측), 그 순간 fast 스킵이 초당 70회 발동하며
                    // 보간 생성이 55 → 23/s로 반토막 난다.
                    //
                    // 이 상수가 들어간 근거는 "쌍당 1.4장이 과생성이라 큐가 적체된다(e2e +30ms)"
                    // 였는데, 그 적체는 present한 프레임의 20~46%를 컴포지터가 버리던 상황에서
                    // 측정된 것이다. 그 원인은 따로 규명됐으므로(캡처가 WindowServer의 프레임당
                    // 예산을 잠식) 이 상수도 다시 재야 한다. 그래서 지우지 않고 다이얼로 만든다.
                    //
                    // 하한을 갭 비율로도 걸어, 갭이 좁을 때 양보가 갭 전체를 삼키지 않게 한다.
                    // 기본값 0.5 — 무인 A/B(2026-08-09, 결정적 소스, 조건당 20창)에서 모든 지표가 개선됐다:
                    //   소스100: 표시 124.1 → 133.1/s, 미표시 5.6 → 1.6%, multFell 6.6 → 1.7/s
                    //   흔들림 σ: 2.82 → 2.05ms (양보를 줄이면 출렁인다던 원 우려와 **정반대**)
                    //   e2e 58 → 55ms (과생성으로 큐가 적체된다던 원 근거도 재현되지 않음)
                    // 0.25·0.0도 재봤으나 0.5보다 낫지 않았다(표시 130.5 / 130.6).
                    let yieldScale = min(max(Knob.double("MACFG_YIELD") ?? 0.5, 0.0), 1.0)
                    let yieldA = min(displayInterval * 0.4 * yieldScale, gap * 0.25)
                    let yieldB = min(displayInterval * 0.6 * yieldScale, gap * 0.35)
                    while slotTime < snappedTs - yieldB && tValues.count < vsyncCap {
                        let t = (slotTime - prev.timestamp) / gap
                        if slotTime > prev.timestamp + yieldA && t > 0.02 {
                            tValues.append(Float(t))
                        }
                        slotTime += displayInterval
                    }
                }
                if tValues.isEmpty { diagTGridEmpty += 1 }
                }   // ← tValues.isEmpty 폴스루 블록 끝

                // **표시 상한을 넘는 생산을 막는다 — 부족분만 만든다.**
                //
                // 실측(2026-08-07, 실사용 마우스가 창 경계를 넘나들 때):
                //   정상   src 110/창, 보간  98 → 표시 89/s, 갭 15, tick 136Hz
                //   붕괴   src 175/창, 보간 211 → 표시 49/s, 갭 59, tick 103Hz
                //   회복   src 100/창, 보간  92 → 표시 94/s, 갭  3, tick 142Hz
                // 경계를 넘으면 소스 창이 호버 전이로 프레임을 쏟아내고(110 → 202), 우리가 그 위에
                // **쌍당 1장을 그대로 얹어** 초당 165장을 인코딩한다. 화면은 144장만 받는다.
                // 넘치는 만큼은 만들자마자 버려지는데 GPU는 이미 썼으므로, 틱이 굶고 표시가 반토막난다.
                //
                // 규칙: 필요한 것은 배율이 아니라 **부족분**이다.
                //   필요 = 주사율 − 소스율,  쌍당 = 필요 / 소스율
                // 소스가 이미 주사율의 절반을 넘으면 쌍당 1장 미만이 정답이라, 그 경우는
                // **몇 쌍마다 한 장**으로 낸다(크레딧). 소스가 주사율을 넘으면 0장이다 —
                // 더 만들 이유가 없다.
                //
                // 이 상한은 **세 경로 공통**으로 마지막에 건다. 경로별로 걸면 설정에 따라 다른
                // 분기를 타면서 새어나간다(거버너 t 상한에서 이미 겪은 실수).
                var quotaEmptied = false
                if !tValues.isEmpty, displayInterval > 0 {
                    let iv = sourceIntervalEMA > 0 ? sourceIntervalEMA : gap
                    let srcHz = iv > 0 ? 1.0 / iv : 0
                    // **목표를 패널 주사율로 잡을 것인가, 달성 틱으로 잡을 것인가 (MACFG_QUOTATICK=1).**
                    //
                    // 기본은 패널(1/displayInterval)이다. 그런데 창모드에서는 패널이 144.048인데
                    // 달성 틱이 ~139로 떨어진다(콜백 유실, r(미표시,gap)=+0.824). 그 상태에서
                    // 144어치를 목표로 잡으면 **달성 못 할 양을 만들어 초당 20장을 버린다** —
                    // 실측: 창모드 인코딩 124/s → 표시 104/s. GPU도 쓰고 리듬도 깨지는 이중 손해다.
                    // 사용자 제보와 일치한다("창모드 커버는 과생성이 심하다").
                    //
                    // 다만 이건 되먹임이다: 덜 만들면 틱이 회복되고, 그러면 목표가 다시 올라
                    // 진동할 수 있다. 이 저장소는 페이싱 컨트롤러로 여러 번 졌다(paceAdaptive 무효).
                    // 그래서 **기본값을 바꾸지 않고 노브로 재본다.** 판정선은 표시 장수가 아니라
                    // 미표시와 motion σ다 — 만드는 양이 줄어도 버리는 게 없어지면 이기는 것이다.
                    let panelHz = 1.0 / displayInterval
                    let refreshHz = (Knob.int("MACFG_QUOTATICK") == 1 && achievedTickHz > 30)
                        ? min(panelHz, achievedTickHz) : panelHz
                    // **배율을 명시했으면 그 배율이 목표다. 주사율을 채우는 게 아니다.**
                    //
                    // 여기는 원래 `refreshHz - srcHz`였다 — 60fps@144면 84장, 쌍당 1.4장이다.
                    // 그건 **Auto(주사율 채우기) 규칙**인데 배율을 2로 지정해도 그대로 적용됐다.
                    // 그래서 갭 확장이 한 쌍에 3장을 만들어도 쿼터가 막지 않았고, 사용자가
                    // 60fps에 ×2(=120)를 걸어놨는데 실제로는 130~140이 나왔다.
                    // 지표(content σ 1.25)로는 좋아 보였지만 그건 답이 아니다 —
                    // **요청하지 않은 20장을 GPU/ANE로 만드는 것은 낭비이고 설정 무시다.**
                    //
                    // 배율 M이면 쌍당 M−1장, 즉 초당 (M−1)×srcHz가 목표다.
                    // 갭 확장 자체는 남긴다(진짜 구멍을 메우는 건 옳다) — 다만 크레딧이
                    // 총량을 잡으므로, 한 쌍이 3장을 쓰면 이후 쌍들이 0장이 되어 평균이 맞는다.
                    let quotaHz = mirrorFrameMultiplier >= 2
                        ? Double(mirrorFrameMultiplier - 1) * srcHz
                        : max(0, refreshHz - srcHz)
                    let deficitHz = min(quotaHz, max(0, refreshHz - srcHz))
                    let perPair = srcHz > 0 ? deficitHz / srcHz : Double(tValues.count)
                    // **쿼터는 정수가 아니라 크레딧으로 준다.**
                    //
                    // 예전에는 `tValues.count > Int(perPair)`로 잘랐는데, 절단이 곧 상한이 된다:
                    // 60fps→144Hz는 perPair가 정확히 1.4인데 Int(1.4)=1이라 **쌍당 영영 1장**이고
                    // 출력이 120에서 멈춘다. 1.4를 내려면 어떤 쌍은 1장, 어떤 쌍은 2장이어야 한다.
                    // (실측 2026-08-09: mult=3, 소스 48/s에서 쌍당 2장이 나와야 하는데 생성이
                    //  51/s에 머물렀고 over가 창당 38~76회 발동 중이었다 — 이 절단이 원인이다.)
                    //
                    // 크레딧은 쌍마다 perPair씩 쌓이고 **실제로 낸 만큼만** 차감한다. 그래서
                    // perPair<1이면 몇 쌍마다 한 장이 되고(옛 pairSkipCounter 분기를 흡수),
                    // perPair>1이면 정수부와 소수부가 자연히 섞인다. 상한 8은 정지 화면 뒤
                    // 크레딧이 쌓여 재생 재개 시 몰아치는 것을 막는다.
                    if perPair < 0.05 {
                        tValues = []                       // 소스만으로 주사율을 채운다
                        quotaEmptied = true
                        diagTOverSupply += 1
                    } else {
                        // 적립은 쌍당 고정이 아니라 **경과 콘텐츠 시간 비례**(deficitHz × gap) —
                        // 정상 쌍(gap=1/srcHz)에선 정확히 perPair와 같고, 드랍 갭(steps>1)에선
                        // 잃은 슬롯만큼 더 쌓여 구멍을 메울 수 있다(쌍당 고정이면 갭 쌍이
                        // 프레임을 더 내야 하는데 크레딧이 모자라 구멍이 남는다 — 리뷰 확정).
                        overSupplyCredit = min(overSupplyCredit + deficitHz * gap, 8.0)
                        let allow = Int(overSupplyCredit)
                        if allow <= 0 {
                            tValues = []
                            quotaEmptied = true
                            diagTOverSupply += 1
                        } else if tValues.count > allow {
                            let stride = Double(tValues.count) / Double(allow)
                            tValues = (0..<allow).map { tValues[min(Int(Double($0) * stride), tValues.count - 1)] }
                            diagTOverSupply += 1
                        }
                        overSupplyCredit -= Double(tValues.count)
                    }
                }
                // 비율 관측 — 절벽(1.5 근처)에 실제로 걸리는지 보여준다
                if displayInterval > 0 {
                    let iv = sourceIntervalEMA > 0 ? sourceIntervalEMA : gap
                    let r = iv / displayInterval
                    if r > 0, r < 50 { diagTRatioMin = min(diagTRatioMin, r); diagTRatioMax = max(diagTRatioMax, r) }
                }
                // 폴백은 큰 갭 + 불운한 그리드 위상일 때만. 작은 갭(≤1.5슬롯)은 소스 두 장이
                // 이미 인접 슬롯을 채우므로 [0.5] 폴백이 잉여 프레임 → 큐 적체(e2e +40ms 실측)
                // 쿼터가 **의도적으로** 비운 것이면 폴백으로 되살리지 않는다 — 안 그러면
                // perPair<1 대역(소스가 주사율 절반 초과)에서 쿼터가 무력화돼 과생산이 돌아온다.
                if tValues.isEmpty && !quotaEmptied && gap > displayInterval * 1.5 { tValues = [0.5] }
                // 모드 3: 버스트로 판정된 쌍은 **스킵 대신 한 장만**. 16.6ms 히치를 8.3×2로
                // 가르면서도 ANE 부하 증가는 쌍당 1장(=정상 쌍과 동일)으로 묶인다.
                if fastCapOne {
                    tValues = tValues.isEmpty ? [0.5] : [tValues[tValues.count / 2]]
                }
                // 거버너 t 상한 — **세 생성 경로 공통**. 경로별로 걸면 Auto 배율(=0)처럼
                // 다른 분기를 타는 설정에서 그냥 새어나간다(실측: 캡을 첫 분기에만 걸었더니
                // 강등 후에도 t×5·t×6이 계속 나옴). 균등 간격으로 솎아 케이던스는 보존.
                // 측정용 강제 상한 — 쌍당 t 개수가 work 꼬리에 얼마나 기여하는지 가른다.
                //   defaults write com.macfg.MacFG env.MACFG_TCAP -string 1
                let effTCap: Int? = {
                    guard let k = tCapOverride else { return tCountCap }
                    return min(k, tCountCap ?? k)
                }()
                if let cap = effTCap, tValues.count > cap {
                    if cap <= 0 {
                        tValues = []
                    } else {
                        let stride = Double(tValues.count) / Double(cap)
                        tValues = (0..<cap).map { tValues[min(Int(Double($0) * stride), tValues.count - 1)] }
                    }
                }
                if tValues.isEmpty {
                    // **의미를 갈라 센다.** 이 자리는 두 가지가 섞인다: 쿼터/상한이 "이 쌍은
                    // 낼 필요 없다"고 판단해 비운 정상 동작(quotaEmptied)과, 어느 경로도
                    // 수확하지 못한 진짜 손실. 한 숫자로 합치면 9.6/s가 손실인지 정상인지
                    // 구분이 안 돼 다음 판단을 못 한다(실측 2026-08-26: 생성/수용이 1.01로
                    // 정상인데 fast가 9.6/s로 찍혀 손실처럼 보였다).
                    if quotaEmptied { diagSkipQuota += 1 } else { diagSkipNoT += 1 }
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
                diagSkipContentFast += 1   // 원본 간격이 표시 슬롯보다 좁음 (진짜 버스트)
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
        let genRef = renderGen
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
                if g > 0, g < 200 {
                    // 완료 핸들러(임의 스레드)의 RMW — stageLock으로 보호 (규약)
                    self.stageLock.lock()
                    self.engineGpuMsEMA = self.engineGpuMsEMA <= 0 ? g : self.engineGpuMsEMA * 0.9 + g * 0.1
                    self.stageLock.unlock()
                }
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
                self.stageLock.lock()
                let base = self.stgWorkEMA
                self.stgWorkEMA = base <= 0 ? workNow : base * 0.95 + workNow * 0.05
                self.stageLock.unlock()
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
                // 대기 = work - (capIngest + cb1 + cb2). GPU가 아무것도 못 하고 멈춘 시간이며,
                // RIFE는 여기가 16ms(소스 간격만큼)인데 MetalFlow는 0.3ms다 — 엔진 간 성능
                // 차이의 전부가 이 항이다(2026-08-26 확정). [STAGE]는 120프레임 평균이라
                // 창별 추이를 못 보므로 EMA로 따로 남겨 [SCHED]에서 창마다 읽는다.
                let waitNow = max(0, workNow - capIngest - self.stgLastCb1Gpu - cb2Gpu)
                self.stgWaitEMA = self.stgWaitEMA <= 0 ? waitNow : self.stgWaitEMA * 0.9 + waitNow * 0.1
                self.stgWork += workNow
                self.stgCount += 1
                let n = self.stgCount
                if n >= 120 {
                    let ci = self.stgCapIngest / Double(n), c1 = self.stgCb1Gpu / Double(n)
                    let c2 = self.stgCb2Gpu / Double(n), wk = self.stgWork / Double(n)
                    let pn = self.stgPresentCount
                    let pg = pn > 0 ? self.stgPresentGpu / Double(pn) : 0
                    self.stgCapIngest = 0; self.stgCb1Gpu = 0; self.stgCb2Gpu = 0; self.stgWork = 0; self.stgCount = 0
                    self.stgPresentGpu = 0; self.stgPresentCount = 0
                    self.stageLock.unlock()
                    // **슬롯당으로 정규화해서 함께 찍는다.**
                    // 위 값들은 분모가 서로 다르다: cb1/cb2는 stgCount(=인제스트된 **소스 프레임**당,
                    // 60fps면 초당 60), present는 stgPresentCount(=**present**당, 초당 100+).
                    // 그런데 사람은 한 줄에 나란히 있으면 더해서 읽는다 — 실제로 그렇게 읽어
                    // "우리 GPU가 슬롯 6.94ms를 100% 채운다"는 틀린 결론을 냈다(2026-08-07 정정).
                    // 진짜 예산 점유는 각 항을 **자기 발생률 × 슬롯 시간**으로 환산해야 나온다.
                    // 참고: MetalFlow에서 splitQueue는 꺼져 있어 cb1·cb2는 workQueue 직렬이지만
                    // present는 presentQueue라 **동시에** 돈다 — 합계는 상한이지 실제 직렬 시간이 아니다.
                    let slotMs = 1000.0 / max(self.mirrorRefreshRate, 60)
                    let srcHz = self.sourceIntervalEMA > 0 ? 1.0 / self.sourceIntervalEMA : 0
                    let presHz = pn > 0 && self.diagLastLogWallSpan > 0
                        ? Double(pn) / self.diagLastLogWallSpan : 0
                    let cb1PerSlot = c1 * srcHz * slotMs / 1000.0
                    let cb2PerSlot = c2 * srcHz * slotMs / 1000.0
                    let presPerSlot = pg * presHz * slotMs / 1000.0
                    let occ = cb1PerSlot + cb2PerSlot + presPerSlot
                    DiagnosticLog.shared.log(String(format:
                        "[STAGE] capIngest=%.1f cb1gpu=%.1f cb2gpu=%.1f present=%.2f(n=%d) work=%.1f (대기=%.1f) chain=%.1f(%.2f×간격) ms/frame (n=120)"
                        + " | 슬롯당(%.2fms): cb1=%.2f cb2=%.2f present=%.2f 합=%.2f(%.0f%%) src=%.0fHz pres=%.0fHz",
                        ci, c1, c2, pg, pn, wk, max(0, wk - ci - c1 - c2), c1 + c2,
                        (c1 + c2) / max(1.0, self.sourceIntervalEMA * 1000.0),
                        slotMs, cb1PerSlot, cb2PerSlot, presPerSlot, occ, occ / slotMs * 100, srcHz, presHz))
                } else {
                    self.stageLock.unlock()
                }
            }
            // GPU 실패면 보간분을 버리고 stable만 등재한다 — 실패한 커맨드버퍼의 출력 텍스처는
            // 이전 쌍 내용이거나 미초기화라서, 조용히 올리면 스테일 프레임이 표시된다(리뷰 확정).
            // stable은 cb1(별도 커맨드버퍼)이 썼으므로 유효하고, releasePrevID는 반드시 전달해
            // 풀 누수를 막는다. 지금까지 GPU 오류는 로그 한 줄 없이 시각 결함으로만 나타났다.
            if let gpuErr = cb2Buf.error {
                DiagnosticLog.shared.log("[GPUERR] cb2 실패 — 보간 폐기, 원본만 등재: \(gpuErr.localizedDescription)")
                let fallback = [TimelineEntry(timestamp: entryTs, texture: stableRef, isInterpolated: false, captureTimestamp: rawCaptureTsRef)]
                mailboxRef.postCompleted(gen: genRef, entries: fallback, released: releasePrevID,
                                         workLatencyMs: (CACurrentMediaTime() - entryTs) * 1000.0, sceneCut: false)
                return
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
            if isSceneCut { self?.pendingSceneCutReset.withLock { $0 = true } }
            mailboxRef.postCompleted(gen: genRef, entries: entries, released: releasePrevID, workLatencyMs: workLatency, sceneCut: isSceneCut)
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

        // 콘텐츠 간격/모션 레이트는 여기서 재지 않는다 — 아래 addPresentedHandler(표시 확정)로
        // 옮겼다. 이유는 shownContentTs 선언부 주석 참조. lastPresentedTimestamp는 **페이싱
        // 입력**(타임라인 프루닝·후보 필터·재표시)이라 인코딩 시점 갱신을 그대로 둔다.
        lastPresentedTimestamp = entry.timestamp
        lastPresentedTexture = entry.texture
        lastPresentedStamp = entry.stamp
        diagPresentCount += 1
        if entry.isInterpolated { diagInterpPresentCount += 1 }
        diagFrameTypes.append(entry.isInterpolated ? "I" : "S")
        if diagFrameTypes.count > 60 { diagFrameTypes.removeFirst(30) }

        let mailboxRef = mailbox
        let genRef = renderGen
        let captureTs = entry.captureTimestamp
        let isInterp = entry.isInterpolated
        let contentTs = entry.timestamp
        let inFlightRef = inFlightPresents
        inFlightRef.withLock { $0 += 1 }
        let slotSec = 1.0 / max(mirrorRefreshRate, 60)
        let targetRef = targetTimestamp
        // 링크가 같은 표시 슬롯을 두 번 준 것인지 (원인이 present가 아니라 틱 쪽인지) 가른다.
        let slotIdx = Int((targetTimestamp / slotSec).rounded())
        if slotIdx == diagLastTargetSlot { diagDupTargetSlot += 1 }
        diagLastTargetSlot = slotIdx
        // **커밋 여유 (B3).** 이 present가 목표 vsync까지 얼마나 남기고 커밋됐나.
        // 홀드 히치의 다음 질문은 "왜 마감을 놓쳤나"인데, 지금 [SCHED]에는 놓친 present의
        // **시각 정보가 없어** 답할 수가 없다(2026-09-01 상관 분석: 엔진 안에서 보면 work·wait·
        // e2e·σ 어느 것도 h>=3을 안 가른다 — AppleFI는 전 지표가 1.0배로 평평했다).
        // 이 값을 표시된 present와 버려진 present로 나눠 보면 원인이 갈린다:
        //   버려진 쪽 여유가 짧다 → 우리가 늦게 커밋한 것 (스케줄러 문제, 우리가 고칠 수 있다)
        //   둘이 같다          → 여유는 충분한데 못 나간 것 (컴포지터/드로어블 쪽)
        let leadMs = (targetTimestamp - CACurrentMediaTime()) * 1000.0
        drawable.addPresentedHandler { [weak self] d in
            inFlightRef.withLock { $0 = max(0, $0 - 1) }
            if let self {
                // 핸들러는 임의 스레드 — diagSlipHist와 같은 stageLock 규약.
                // 칸 경계는 실측 후 넓혔다 — 첫 판(<0/<2/<5/<10)은 전부 마지막 칸에 몰려
                // 포화됐다. 여유는 항상 10ms를 넘는다(지연 오프셋 3~4슬롯 ≈ 21~28ms).
                let b = leadMs < 0 ? 0 : leadMs < 10 ? 1 : leadMs < 20 ? 2 : leadMs < 30 ? 3 : 4
                self.stageLock.lock()
                if d.presentedTime > 0 { self.diagLeadShown[b] += 1 } else { self.diagLeadDrop[b] += 1 }
                self.stageLock.unlock()
            }
            if let self, d.presentedTime > 0 {
                // 실제 표시가 목표 슬롯에서 몇 칸 밀렸나. 밀림이 미표시 비율과 맞아떨어지면
                // "뒤 present에 추월당해 버려진다"가 확정된다.
                let slip = Int(((d.presentedTime - targetRef) / slotSec).rounded())
                // presentedHandler는 임의 스레드 — diagGpuLateHist와 같은 규약으로 stageLock 보호.
                // (무락이면 렌더 스레드의 [SCHED] 리셋 재할당과 겹쳐 해제된 버퍼에 쓸 수 있다.)
                self.stageLock.lock()
                self.diagSlipHist[min(max(slip, 0), 3)] += 1
                // 표시 확정 기준 콘텐츠 전진 — 핸들러는 임의 스레드에 **순서 보장이 없다**
                // (드로어블 3개 인플라이트). 뒤집혀 도착한 표본은 음수 간격을 만들므로
                // 콘텐츠·벽시계가 **둘 다 전진했을 때만** 표본으로 삼고, 상태는 max로 단조 유지한다.
                if self.shownContentTs > 0, contentTs > self.shownContentTs,
                   d.presentedTime > self.shownWallTs {
                    let dContent = contentTs - self.shownContentTs
                    let dWall = d.presentedTime - self.shownWallTs
                    let cd = dContent * 1000.0
                    if cd > 0 && cd < 100 { self.diagContentIntervals.append(cd) }
                    if dWall > 0 && dWall < 0.1 {
                        let rate = dContent / dWall
                        self.diagMotionRates.append(rate)
                        let b = rate < 0.5 ? 0 : rate < 0.8 ? 1 : rate < 1.1 ? 2 : rate < 1.4 ? 3 : 4
                        self.diagMotionHist[b] += 1
                        if self.tsDumpEnabled, self.diagTsDump.count < 8192 {
                            self.diagTsDump.append((d.presentedTime, contentTs))
                        }
                    }
                }
                if contentTs > self.shownContentTs { self.shownContentTs = contentTs }
                if d.presentedTime > self.shownWallTs { self.shownWallTs = d.presentedTime }
                self.stageLock.unlock()
            }
            mailboxRef.postPresented(gen: genRef, at: d.presentedTime, captureTs: captureTs, isInterp: isInterp)
        }
        // 이 커맨드 버퍼가 끝날 때까지 소스 텍스처를 붙잡는다 — 그 전에 풀이 재사용하면
        // 읽는 중에 덮어쓰게 된다(프레임 중첩).
        let presentTex = entry.texture
        let presentingRef = presentingTextures
        // 참조 **카운트** — 재표시 경로(forceRepresentTicks/alwaysRepresent)가 같은 텍스처를
        // 연속 present하면 CB 2개가 동시에 떠 있는데, 단순 딕셔너리면 먼저 끝난 CB가 보호를
        // 제거해 나중 CB가 읽는 중에 엔진이 그 텍스처를 덮어쓸 수 있다(리뷰 확정).
        presentingRef.withLock {
            let key = ObjectIdentifier(presentTex)
            $0[key] = (presentTex, ($0[key]?.count ?? 0) + 1)
        }
        let stageDbgRef = stageDbg
        cb.addCompletedHandler { [weak self] buf in
            presentingRef.withLock {
                let key = ObjectIdentifier(presentTex)
                if let cur = $0[key] {
                    if cur.count <= 1 { $0.removeValue(forKey: key) } else { $0[key] = (cur.tex, cur.count - 1) }
                }
            }
            guard let self else { return }
            // **미표시의 직접 원인 판별.** 드로어블은 이 콜백의 표시 슬롯에 이미 바인딩돼 있어
            // (그래서 present(atTime:)이 불법이다), GPU가 그 슬롯을 넘겨서 끝나면 표시를 놓치고
            // 다음 슬롯으로 밀려 뒤 present와 충돌한다. 여기서 "GPU가 목표 슬롯보다 몇 칸 늦게
            // 끝났나"를 세면, 그 비율이 미표시 비율과 맞는지로 원인을 가를 수 있다:
            //   맞으면  → GPU 오버런이 원인 (고칠 곳은 생산 리드타임/작업량)
            //   안 맞으면 → 제때 끝났는데도 버려진 것 (고칠 곳은 컴포지터/위상)
            let lateSlots = Int(((buf.gpuEndTime - targetRef) / slotSec).rounded(.down))
            self.stageLock.lock()
            self.diagGpuLateHist[min(max(lateSlots + 1, 0), 4)] += 1
            self.stageLock.unlock()
            guard stageDbgRef else { return }
            let g = (buf.gpuEndTime - buf.gpuStartTime) * 1000.0
            guard g > 0, g < 500 else { return }
            self.stageLock.lock()
            self.stgPresentGpu += g
            self.stgPresentCount += 1
            self.stageLock.unlock()
        }
        // 표시 시각을 링크가 준 슬롯에 못박을지 여부.
        //
        // plain present는 "GPU 완료 즉시"라 표시 시각이 우리 작업 시간을 따라 흔들린다.
        // 그러면 두 present가 같은 리프레시 구간에 떨어져 앞의 것이 버려진다(presentedTime==0).
        // 등간격으로 내보낸 실측(MACFG_PRESENTEVERY=3)에서 미표시가 정확히 0이고 glass가
        // 정확히 3슬롯이었던 것이 이 해석의 근거다. atTime은 그 등간격성을 인위적 스로틀 없이
        // 만들어낸다 — 링크가 준 슬롯은 틱마다 서로 다르므로 충돌이 원리적으로 사라진다.
        // **명시 시각 present는 이 경로에서 불법이다 (실측 2026-08-05로 확정).**
        // `cb.present(drawable, atTime: targetTimestamp)`를 쓰면 즉시
        //   -[CAMetalDrawable presentWithOptions:] → NSException → SIGABRT
        // 로 죽는다. CAMetalDisplayLink가 배달하는 드로어블은 이미 그 콜백의
        // targetPresentationTimestamp 슬롯에 바인딩돼 있어서, 표시 시각을 다시 지정하는
        // 것 자체가 허용되지 않는다. (nextDrawable로 직접 얻은 드로어블과 다르다.)
        // 그래서 "present를 vsync 격자에 못박아 위상 충돌을 없앤다"는 접근은 이 구조에서
        // 쓸 수 없다 — 위상을 고치려면 present 시각이 아니라 **무엇을 언제 만들지**를 바꿔야 한다.
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
            // **씨앗은 중앙값이 아니라 평균이다 (2026-08-27, 사용자 3회 제보 "소스가 80으로 튄다").**
            //
            // SCK 배달은 디스플레이 격자에 양자화된다. 60fps 소스를 144Hz로 받으면 도착 간격이
            // **2틱(13.9ms)과 3틱(20.8ms) 두 값으로만** 나온다 (패턴 2,2,3,2,3 → 평균 16.67ms
            // = 정확히 60fps). 이때 중앙값을 씨앗으로 쓰면 접기가 무너진다:
            //     candidate = 13.9  →  13.9/13.9 = 1.00 → 1슬롯
            //                          20.8/13.9 = 1.50 → **반올림 2슬롯**
            //                          (13.9+20.8)/(1+2) = 11.6ms = 86fps
            // 실측 라벨이 60↔80~87을 오간 것이 정확히 이 값이다. 1.5가 2로 반올림되며 추정치가
            // 반토막 난다. 평균을 씨앗으로 쓰면 두 델타 모두 1슬롯으로 접혀 16.67ms가 복원된다.
            //
            // 이게 라벨만의 문제가 아닌 이유: sourceIntervalEMA는 부족분 쿼터(deficitHz/srcHz),
            // 적응지연 baseMs, staleCutoff, maxUseful 계산에 모두 쓰인다. 83fps로 오판하면
            // "주사율에 가까우니 보간이 덜 필요하다"고 계산해 실제로 덜 만든다.
            //
            // 평균은 드랍에 약하지만(한 장 빠지면 델타 하나가 2배) 링이 60개라 1.7% 영향이고,
            // 아래 접기가 그 델타를 2슬롯으로 정규화해 스스로 교정한다. 반대로 중앙값은
            // 양자화에 **구조적으로** 틀리므로 접기로도 복구되지 않는다.
            candidate = arrivalMean
            // **슬롯 세기를 두 번 반복한다. 중앙값 접기 루프는 제거했다.**
            // 접기는 양자화된 입력에 해롭다: 13.9와 20.8이 둘 다 k=1로 접히면 median이
            // 다시 짧은 쪽(13.9)으로 돌아가 86fps 오판이 그대로 재생산된다(시뮬레이션 확인).
            // 슬롯 세기(시간폭÷슬롯수)는 같은 입력에서 정확히 16.66ms를 복원한다.
            // 첫 회는 드랍으로 부푼 평균을 정규화하고 두 번째가 수렴값이다.
            //
            // 검증(시뮬레이션, 60개 델타):
            //   60fps@144Hz 격자(2,2,3,2,3틱)  현행 72.0fps → 신규 60.0fps
            //   60fps + 드랍 10%               현행 72.0fps → 신규 59.6fps
            //   30fps@144Hz 격자(5,5,4,5,5틱)  현행 28.8fps → 신규 30.0fps
            //   24fps(정확히 6틱) / 지터 없는 60fps  둘 다 불변
            // **평균이 기본이고, 명백한 드랍만 접는다 (2026-08-31 정정).**
            //
            // 직전 판은 "정수 경계에서 애매한 비율(±0.35 밖)은 버린다"였는데, 실측 링을 찍어보니
            // 그게 **양 끝을 모두 잘라 추정을 위로 편향**시켰다:
            //   d=[18.3 9.5 26.0 9.7 17.4 20.4 11.3 17.0 19.0 18.9 18.1 19.7 8.9 27.5 16.8]
            //   평균 17.23인데 9.5·9.7·8.9(비율 0.52~0.56)와 26.0·27.5(1.51~1.60)가 전부 버려져
            //   17~20 구간만 남고 **cand 18.39**가 됐다. 짧은 델타와 그걸 상쇄하는 긴 델타는
            //   같은 지터의 양면인데 둘 다 버리니 가운데만 남는다. 자기강화이기도 하다 —
            //   후보가 오르면 짧은 쪽이 더 많이 잘린다.
            //   그 오차가 콘텐츠 시각 배치에 실려 content σ가 win 4.30 / disp 2.50으로 갈렸다
            //   (사용자 체감: "마우스 폴링레이트 낮은 걸로 화면 돌리는 느낌").
            //
            // 가드를 빼기만 하면 반대로 망가진다 — 26.0을 2슬롯으로 접어 15.21ms(65.7fps).
            // 이 분포에는 깨끗한 배수가 없다. **버릴 게 아니라 접지 않으면 되는 것이었다.**
            // 접기는 진짜 드랍에만 필요하고 드랍은 평균의 2배 근처로 나타나므로 1.7배를 문턱으로 둔다.
            //
            // 시뮬레이션 6케이스 전부 실제값 5% 이내:
            //   실측 win 링 18.40 → **17.23** (실제 16.67) · 60fps@144격자 16.66(불변)
            //   60fps+드랍 16.69(불변) · 30fps@144격자 33.32(불변) · 24fps 41.67 · 지터없는 60fps 16.67
            for _ in 0..<2 {
                var slotCount = 0.0
                var slotSpan = 0.0
                // **접기 판정의 기준은 중앙값이다. 평균이 아니다.**
                //
                // 접기는 진짜 드랍(≈2배)에만 필요한데, 기준을 평균으로 두면 **드랍이 평균을
                // 오염시켜 자기 자신을 숨긴다.** 16.7ms 14개 중 4개가 33.3ms인 경우
                // 평균이 21.4로 올라가 33.3/21.4 = 1.56이 되고, 1.7 문턱을 못 넘어 안 접힌다 →
                // 추정치가 21.44ms(46fps)로 굳는다. 중앙값은 드랍에 안 흔들려 33.3/16.7 = 1.99다.
                // 실사용에서 27.7/16.3 = 1.70이 문턱에 정확히 앉아 창마다 뒤집히던 것도
                // 같은 뿌리였다(평균이 지터에 끌려다녀 경계가 움직였다). 중앙값 기준에서는
                // 27.7/16.7 = 1.66으로 안정적으로 문턱 아래다.
                //
                // 시뮬레이션 8케이스 전부 실제값 5% 이내 (평균 기준은 드랍3장에서 21.44로 실패):
                //   실측 win 링 17.23 · 문턱경계 링 16.55 · 60fps@144격자 16.66
                //   60fps+드랍1 16.69 · **60fps+드랍3 16.68**(평균기준 21.44) · 30fps@144 33.32
                //   24fps 41.67 · 지터없는 60fps 16.67
                let foldRef = deltas.sorted()[deltas.count / 2]
                for d in deltas {
                    let k = d >= foldRef * 1.75 ? max(1.0, (d / candidate).rounded()) : 1.0
                    slotCount += k
                    slotSpan += d
                }
                if slotCount >= 3 { candidate = slotSpan / slotCount }
            }
        }
        // **추정기 입력·출력 덤프 (MACFG_PLLDUMP=1). 추측을 멈추기 위한 계측.**
        //
        // 2026-08-31: 창 캡처에서 초당 59.3장을 수용하는데(→ 평균 간격 16.86ms) 추정치는
        // 18.2ms(55fps)로 나온다. 링에는 수용 프레임의 raw 도착 시각이 들어가므로
        // **추정기가 자기 입력의 평균보다 높게 나오는 셈**인데, dist 5칸으로 델타를 복원해
        // 시뮬레이션한 두 번 모두 16.2~16.3ms가 나와 재현에 실패했다.
        // 5칸 히스토그램으로는 원본 분포를 못 되살린다 — 실제 델타를 봐야 한다.
        if Knob.int("MACFG_PLLDUMP") == 1, pllDumpCount < 40 {
            pllDumpCount += 1
            var ds: [Double] = []
            for i in 1..<snapTsRing.count { ds.append((snapTsRing[i] - snapTsRing[i-1]) * 1000) }
            let mean = ds.isEmpty ? 0 : ds.reduce(0,+) / Double(ds.count)
            DiagnosticLog.shared.log(String(format: "[PLL] n=%d 평균%.2f 중앙%.2f → cand %.2f (EMA %.2f) burst=%@ d=[%@]",
                ds.count, mean, ds.isEmpty ? 0 : ds.sorted()[ds.count/2], candidate * 1000,
                sourceIntervalEMA * 1000, shortCount >= 2 ? "Y" : "N",
                ds.map { String(format: "%.1f", $0) }.joined(separator: " ")))
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
        checkRenderThread("acquireStableTexture")
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
            stageLock.lock(); shownContentTs = 0; shownWallTs = 0; stageLock.unlock()
        }

        var busy = Set<ObjectIdentifier>()
        for entry in timeline { busy.insert(ObjectIdentifier(entry.texture)) }
        if let prev = prevStable { busy.insert(ObjectIdentifier(prev.texture)) }
        if let last = lastPresentedTexture { busy.insert(ObjectIdentifier(last)) }
        busy.formUnion(inFlightTextures.keys)
        busy.formUnion(presentingTextures.withLock { Array($0.keys) })

        return stablePool.first { !busy.contains(ObjectIdentifier($0)) }
    }

    // MARK: - Diagnostics

    nonisolated private func maybeLogDiagnostics() {
        guard diagTick % 240 == 0 else { return }  // 진단 중 2초 주기 (@120Hz)

        // 틱 레이트: 240틱의 실제 벽시계 소요 — 2.0s면 무손실 120Hz, 2.05s면 ~117Hz(틱 유실)
        let nowWall = CFAbsoluteTimeGetCurrent()
        let wallSpan = diagLastLogWall > 0 ? nowWall - diagLastLogWall : 0
        let tickHz = wallSpan > 0 ? 240.0 / wallSpan : 0
        if tickHz > 30 { achievedTickHz = achievedTickHz > 0 ? achievedTickHz * 0.7 + tickHz * 0.3 : tickHz }
        lastTickHz = tickHz   // 거버너 신호용 (다음 창에서 읽음)
        diagLastLogWallSpan = wallSpan
        diagLastLogWall = nowWall
        let tickCPUAvg = diagTickCPUSum / 240.0
        let ingAvg = diagIngestSamples > 0 ? diagIngestSum / Double(diagIngestSamples) : 0
        let tickStats = String(format: "tick=%.1fHz(패널%.3f) cpu=%.1f/%.1fms over=%d gap=%d(pre%.1f) foreign=%llu ing=%.2f/%.1fms ingOver=%d",
                               tickHz, measuredTickEMA > 0 ? 1.0 / measuredTickEMA : mirrorRefreshRate, tickCPUAvg, diagTickCPUMax, diagTickOverruns, diagTickGaps, diagGapPrevCPUMax,
                               renderDriver.foreignTickDrops, ingAvg, diagIngestMax, diagIngestOver)
        diagTickCPUSum = 0; diagTickCPUMax = 0; diagTickOverruns = 0
        diagTickGaps = 0; diagGapPrevCPUMax = 0
        // 콘텐츠 간격 + 모션 레이트 통계 (wobble 지표).
        // 표본은 presentedHandler(임의 스레드)가 쌓으므로 스냅샷/리셋을 stageLock으로 감싼다 —
        // 무락이면 여기 재할당과 핸들러 append가 겹쳐 해제된 버퍼에 쓴다(diagGpuLateHist와 같은 규약).
        stageLock.lock()
        let ci = diagContentIntervals
        let mr = diagMotionRates
        let mHist = diagMotionHist
        let tsRows = diagTsDump
        diagContentIntervals = []
        diagMotionRates = []
        diagMotionHist = [0, 0, 0, 0, 0]
        diagTsDump = []
        stageLock.unlock()
        let ciAvg = ci.isEmpty ? 0 : ci.reduce(0, +) / Double(ci.count)
        let ciVar = ci.isEmpty ? 0 : ci.map { ($0 - ciAvg) * ($0 - ciAvg) }.reduce(0, +) / Double(ci.count)
        // motion=1.00이 정속. σ가 체감 매끄러움의 1차 지표이고, stall은 정속의 60% 아래로
        // 느려진 구간 수 = 눈에 보이는 히치 빈도다. content σ는 이걸 전혀 못 잡는다(홀드 무시).
        let mrAvg = mr.isEmpty ? 0 : mr.reduce(0, +) / Double(mr.count)
        let mrVar = mr.isEmpty ? 0 : mr.map { ($0 - mrAvg) * ($0 - mrAvg) }.reduce(0, +) / Double(mr.count)
        if tsDumpEnabled, !tsRows.isEmpty {
            // wall,content (초). 파일이 없으면 헤더부터.
            let path = "/tmp/MacFG_ts.csv"
            let body = tsRows.map { String(format: "%.6f,%.6f", $0.0, $0.1) }.joined(separator: "\n") + "\n"
            if let h = FileHandle(forWritingAtPath: path) {
                h.seekToEndOfFile(); h.write(Data(body.utf8)); try? h.close()
            } else {
                try? ("wall,content\n" + body).write(toFile: path, atomically: false, encoding: .utf8)
            }
        }
        let mTot = max(1, mHist.reduce(0, +))
        let mPct = mHist.map { String(format: "%.0f", Double($0) * 100.0 / Double(mTot)) }.joined(separator: "/")
        let ciStats = String(format: "content=%.1f±%.1fms motion=%.2f±%.2f m[%@]",
                             ciAvg, sqrt(ciVar), mrAvg, sqrt(mrVar), mPct)

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

        // **홀드 패턴 — σ가 못 보는 것을 본다.**
        //
        // σ는 "간격이 평균에서 얼마나 흩어졌나"만 잰다. 그런데 규칙적으로 흩어진 것과
        // 무작위로 흩어진 것을 구분하지 못한다. 144Hz에서 네이티브 60fps는 2.4슬롯이라
        // 홀드가 2,2,3이 **규칙적으로 반복**되는데 σ는 5.49로 크게 나온다 — 그런데 사람은
        // 규칙적인 리듬을 배경으로 처리해 부자연스럽게 느끼지 않는다(사용자 지적, 타당).
        // 같은 σ라도 1,2가 무작위로 섞이면 체감이 전혀 다를 수 있다.
        // 그래서 각 표시 간격을 슬롯 수로 양자화한 **순서**를 그대로 남긴다. 숫자열을 보면
        // 반복 주기가 눈에 보이고, 아래 repeat 지표가 그것을 정량화한다.
        let slotMs = 1000.0 / max(mirrorRefreshRate, 60)
        let holds = intervals.map { max(1, min(9, Int(($0 / slotMs).rounded()))) }
        // 규칙성 지표: 주기 p(1~6)로 접었을 때 일치율이 가장 높은 p와 그 일치율.
        // 완전 규칙(2,2,3 반복)이면 p=3에서 100%, 무작위면 어느 p에서도 낮다.
        var bestPeriod = 0, bestScore = 0.0
        if holds.count >= 12 {
            for p in 1...6 where holds.count > p {
                var hit = 0
                for i in p..<holds.count where holds[i] == holds[i - p] { hit += 1 }
                let sc = Double(hit) / Double(holds.count - p)
                if sc > bestScore { bestScore = sc; bestPeriod = p }
            }
        }
        let holdStr = holds.suffix(32).map(String.init).joined()

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
        if diagFastCapped > 0 { skipParts.append("fastCap:\(diagFastCapped)") }
        if diagGapExpandReal + diagGapExpandSnap > 0 {
            skipParts.append("gapExp:\(diagGapExpandReal)r/\(diagGapExpandSnap)s")
        }
        if diagSkipQuota > 0 { skipParts.append("quota:\(diagSkipQuota)") }
        if diagSkipNoT > 0 { skipParts.append("noT:\(diagSkipNoT)") }
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

        // 임의 스레드(핸들러)가 쓰는 히스토그램은 락 안에서 스냅샷을 떠서 조립한다
        stageLock.lock()
        let slipSnapshot = diagSlipHist
        let gpuLateSnapshot = diagGpuLateHist
        stageLock.unlock()
        stageLock.lock(); let waitSnapshot = stgWaitEMA; stageLock.unlock()
        // **배치와 캡처 소스를 매 창에 찍는다 (2026-08-30).**
        // 커버/뷰어 A/B가 다섯 변수로 교락됐던 이유 중 둘이 여기에 안 찍혀 있었기 때문이다 —
        // 배치는 [WIN] 로그에만, 캡처 소스는 [SCK-DISPLAY]에만 나와서 창 단위로 못 맞췄다.
        // 매 창에 라벨이 있으면 사용자가 아무 순서로 토글해도 사후에 2×2로 가를 수 있다.
        // (측정 절차를 사람이 정확히 지키게 만들 게 아니라, 지표가 엉성한 입력을 견뎌야 한다.)
        let msg = "[SCHED] place=\(placeTagMirror) src=\(diagSourceCount)(\(String(format: "%.0f", srcFps))fps) uniqOut=\(uniquePresented) dupSkip=\(diagDupSkipCount) chg=\(diagChangeHist.map(String.init).joined(separator: "/")) srcLock=\(diagSrcLockSkip) srcRestore=\(diagSrcRestore) rawDist=\(diagRawIntHist.map(String.init).joined(separator: "/")) uiGate=\(diagUiGateSkip) tsRej=\(diagTsRejectCount) interpEnc=\(diagInterpEncodedCount) skip[\(skips)] present=\(diagPresentCount) (I=\(diagInterpPresentCount) 미표시=\(diagPresentDropped)) lat=+\(Int(extraLatencySlots)) \(tickStats) \(ciStats) cut=\(cuts) resync=\(diagResyncCount) snapMiss=\(diagSnapMissCount)(pull=\(diagSnapPullableCount) lagMax=\(String(format: "%.1f", diagSnapPullLagMax * 1000))ms) poolMiss=\(diagPoolExhaustCount)(deliv=\(String(format: "%.0f%%", (diagSourceCount + diagPoolExhaustCount) > 0 ? Double(diagSourceCount) * 100.0 / Double(diagSourceCount + diagPoolExhaustCount) : 100.0))) tl=\(timeline.count) t[multFell=\(diagTMultFell) gridEmpty=\(diagTGridEmpty) over=\(diagTOverSupply) ratio=\(String(format: "%.2f~%.2f", diagTRatioMin > 900 ? 0 : diagTRatioMin, diagTRatioMax))] every=\(presentEveryN) tCap=\(tCountCap.map(String.init) ?? "-") pace=\(String(format: "%.2f", paceScale))/\(String(format: "%.0f", paceLastShownRate)) slip=\(slipSnapshot.map(String.init).joined(separator: "/")) gpuLate=\(gpuLateSnapshot.map(String.init).joined(separator: "/")) dupSlot=\(diagDupTargetSlot) | glass(ms): avg=\(String(format: "%.2f", avgInterval)) σ=\(String(format: "%.2f", sqrt(variance))) max=\(String(format: "%.1f", maxInterval)) | srcInt=\(String(format: "%.1f", sourceIntervalEMA * 1000))ms dist=\(diagSrcIntHist.map(String.init).joined(separator: "/")) [\(String(format: "%.0f", srcIntLo))~\(String(format: "%.0f", srcIntHi))] | drain=\(String(format: "%.1f", drainAvg))/\(diagDrainDepthMax) | work=\(String(format: "%.0f", avgWork))/\(String(format: "%.0f", maxWork))ms wait=\(String(format: "%.1f", waitSnapshot))ms e2e=\(String(format: "%.0f", avgLatency))ms | lead(shown)=\(diagLeadShown.map(String.init).joined(separator: "/")) lead(drop)=\(diagLeadDrop.map(String.init).joined(separator: "/")) | hold=\(holdStr) p\(bestPeriod)=\(String(format: "%.0f%%", bestScore * 100)) | \(pattern)"
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
        diagSkipContentFast = 0; diagFastCapped = 0
        diagGapExpandReal = 0; diagGapExpandSnap = 0; diagSkipQuota = 0; diagSkipNoT = 0; diagSkipBigGap = 0; diagSkipDiscontinuity = 0
        diagSkipEngineFail = 0; diagSkipOther = 0
        diagStaleDropCount = 0; diagCapDropCount = 0; diagLeaseDropCount = 0; diagSkipBackpressure = 0; diagPresentBusy = 0; diagStaleSampleCount = 0

        diagSourceCount = 0
        diagDupSkipCount = 0
        diagChangeHist = [0, 0, 0, 0, 0]
        diagSrcLockSkip = 0
        diagSrcRestore = 0
        stageLock.lock()
        for i in diagLeadShown.indices { diagLeadShown[i] = 0; diagLeadDrop[i] = 0 }
        stageLock.unlock()
        for i in diagRawIntHist.indices { diagRawIntHist[i] = 0 }
        diagUiGateSkip = 0
        stageLock.lock(); diagGpuLateHist = [0, 0, 0, 0, 0]; stageLock.unlock()
        diagTsRejectCount = 0
        _ = wallSpan
        diagPresentCount = 0
        diagPresentDropped = 0
        stageLock.lock(); diagSlipHist = [0, 0, 0, 0]; stageLock.unlock()
        diagTMultFell = 0; diagTGridEmpty = 0; diagTRatioMin = 999.0; diagTRatioMax = 0.0; diagTOverSupply = 0
        diagDupTargetSlot = 0
        if presentEveryAlternates { presentEveryN = presentEveryN == 1 ? 2 : 1 }
        diagInterpPresentCount = 0
        diagPoolExhaustCount = 0
        diagInterpEncodedCount = 0
        diagSrcIntMin = .infinity
        diagSrcIntHist = [0, 0, 0, 0, 0]
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
        // **소스가 전체화면인 동안엔 배치를 건드리지 않는다.** 전체화면에선 커버가 합성되지
        // 않아 뷰어가 강제인데, 업스케일 파생이 cover를 넣으면 다음 추적 틱(15~30Hz)에 AUTOFS가
        // 도로 viewer로 되돌린다 — 무동작이어야 할 설정 변경에 출력 창 재생성 2회 + 검은 화면
        // 번쩍임 + 스케줄러 리셋 2회.
        //
        // 예전 가드는 `!autoFsViewer`, 즉 "AUTOFS가 전환한 적이 있나"라는 **이력**이었다.
        // 업스케일 ON으로 캡처를 시작하면 배치가 처음부터 뷰어라 AUTOFS가 전환할 게 없어
        // (3801의 `placement == .coverSource` 가드에서 조기 리턴) 플래그가 false로 남았고,
        // 전체화면인데도 가드가 열려 위 왕복이 그대로 일어났다. 27000줄 로그에 `place=viewer/disp`가
        // 한 번도 없는 것이 방증이다 — 업스케일이 계속 off라 이 조합을 밟은 적이 없었다.
        // 물어야 할 것은 "지금 전체화면인가"라는 **상태**이고, 그건 캐시 없이 바로 구할 수 있다.
        // 전체화면 이탈 경로가 upscaleMode로 배치를 다시 유도하므로 사용자 선택은 보존된다.
        guard !(overlayManager?.sourceIsFullscreen ?? false) else { return }
        let target: OverlayPlacement = placementPin ?? (upscaleMode == .off ? .coverSource : .viewerWindow)
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
        // **출력 창이 새로 만들어졌다 — 디스플레이 캡처 중이면 제외 목록을 다시 박는다.**
        // 안 하면 새 windowID가 필터에 없어 우리 출력을 되먹는다(창 안에 창 무한 중첩).
        if captureManager.isDisplayCapture, let om = overlayManager {
            Task { @MainActor in
                await captureManager.refreshDisplayExclusions(
                    excludingWindowIDs: om.ownWindowIDs, requiredWindowID: om.outputWindowID)
            }
        }
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
        guard isCapturing, !captureSwitchInFlight, captureRegion == nil,
              let om = overlayManager else { return }
        // **캡처 소스 강제 (MACFG_FORCEDISP, 측정용).**
        //
        // 2026-08-31 실측이 배치 가설을 죽였다 — 같은 창 캡처에서 커버 104.9/s vs 뷰어 99.1/s로
        // 뷰어가 오히려 나쁘고, 유일하게 멀쩡한 조합(미표시 0, 표시 122.8)은 **디스플레이 캡처**
        // 쪽이었다. 즉 축은 배치가 아니라 캡처 소스다. 그런데 disp 팔은 소스가 전체화면
        // Space에 있는 상태와 묶여 있어 두 변수가 아직 안 갈린다:
        //   ① 디스플레이 캡처가 창 캡처보다 싸다   ② 전체화면 Space라 합성이 적다
        // 창모드 소스에 디스플레이 캡처를 붙이면 ②를 고정한 채 ①만 본다.
        // (updateToDisplayCapture는 captureRect=nil로 화면 전체를 잡는다. 소스 창이
        //  3755x2130으로 화면의 98%를 채우므로 픽셀 수는 사실상 같아 처리량 비교가 성립한다.)
        // 1=항상 디스플레이 캡처, 0=항상 창 캡처, 미설정=기존 자동.
        let forceDisp = Knob.int("MACFG_FORCEDISP")
        let isFS = forceDisp.map { $0 == 1 } ?? om.sourceIsFullscreen
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
            // 배치 전환은 **실제** 전체화면일 때만 — FORCEDISP는 캡처 소스만 바꾸는 노브다.
            guard om.sourceIsFullscreen else { return }
            // **핀이 있으면 배치를 건드리지 않는다.**
            // 안 그러면 핀(cover)과 AUTOFS(viewer)가 추적 틱(15~30Hz)마다 서로를 되돌리며
            // 창을 계속 재생성한다 — 화면이 깜빡이고 [SCK-DISPLAY] 제외 갱신이 300ms에 10번 찍힌다
            // (실측 2026-08-31). 핀은 측정용이므로 그동안 AUTOFS 배치 전환은 양보한다.
            guard placementPin == nil else { return }
            guard selectedOverlayPlacement == .coverSource else { return }
            selectedOverlayPlacement = placementPin ?? .viewerWindow
            updateOverlayPlacement()
            DiagnosticLog.shared.log("[AUTOFS] 소스 전체화면 → viewer 자동 전환")
        } else {
            syncCaptureSourceForFullscreen(false)
            guard placementPin == nil else { return }
            // **조건 없이 upscaleMode로 정산한다** — 전체화면 동안 억제됐던 파생을 여기서 갚는다.
            // 실제로 바뀔 때만 재생성하므로 이탈마다 창이 다시 만들어지지는 않는다.
            let target: OverlayPlacement = (upscaleMode == .off) ? .coverSource : .viewerWindow
            guard target != selectedOverlayPlacement else { return }
            selectedOverlayPlacement = target
            updateOverlayPlacement()
            DiagnosticLog.shared.log("[AUTOFS] 소스 창 복귀 → \(target == .coverSource ? "cover" : "viewer") 원복")
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
        guard isCapturing, !isRestartingCapture, !captureSwitchInFlight, captureRegion == nil else { return }
        guard captureManager.isDisplayCapture != fullscreen else { return }   // 이미 원하는 모드
        guard let om = overlayManager else { return }
        captureSwitchInFlight = true
        Task { @MainActor in
            defer { captureSwitchInFlight = false }
            do {
                if fullscreen {
                    guard let scr = om.sourceScreen,
                          let num = scr.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber
                    else { return }
                    // 출력 창 ID를 함께 남긴다 — 위 [SCK-DISPLAY] 누락 목록과 대조해
                    // "빠진 게 하필 출력 창인가"를 즉시 판정할 수 있게.
                    DiagnosticLog.shared.log("[SCK-DISPLAY] 출력 창 id=\(om.outputWindowID) 전체=\(om.ownWindowIDs)")
                    try await captureManager.updateToDisplayCapture(
                        displayID: CGDirectDisplayID(num.uint32Value),
                        excludingWindowIDs: om.ownWindowIDs,
                        requiredWindowID: om.outputWindowID)
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
        // **최소화는 위 신호로 안 잡힌다.** 프레임이 끊기는 게 아니라 창 프레임이 독 타일로
        // 줄어들 뿐이고 SCK는 그 타일을 계속 배달하므로 `lastFrameArrivalAt`이 갱신되어
        // "1초 무프레임"이 영영 참이 안 된다 — 실측 2026-09-01: ⌘M 후에도 [OVERLAY] 0줄.
        // AX 기반이라 즉답이고, 전체화면 전환과 혼동되지 않는다(WindowTracker 참조).
        // 기하 거부(독 타일)로 동결된 경우도 포함 — AX가 최소화를 놓쳐도 숨겨야 한다.
        let sourceMinimized = (overlayManager?.sourceIsMinimized ?? false)
            || (overlayManager?.trackingFrozen ?? false)
        let shouldHide = overlayUserHidden || sourceOffScreen || sourceMinimized
            || (!sourceFront && !coverKeepVisible)
        guard shouldHide != overlayHiddenState else { return }
        overlayHiddenState = shouldHide
        overlayManager?.setOverlayHidden(shouldHide)
        if shouldHide {
            DiagnosticLog.shared.log("[OVERLAY] hidden — \(sourceMinimized ? "최소화" : sourceOffScreen ? "프레임 고갈" : overlayUserHidden ? "수동" : "소스 비최전면")")
        } else {
            // 숨김 동안 프레임을 버려 연속성이 끊김 — 리셋은 렌더 스레드가 자기 틱에서 수행
            pendingShowReset = true
            DiagnosticLog.shared.log("[OVERLAY] shown → scheduler reset (render-thread)")
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
            // ⌃⌥⌘P — 배치 핀 순환 (커버 → 뷰어 → 자동). **측정용.**
            // 2×2(소스 창모드/전체화면 × 배치 커버/뷰어)를 한 실행 안에서 채우려면 재시작 없이
            // 배치를 바꿔야 한다. 재시작 A/B는 콘텐츠 드리프트로 교란된다(MEMORY 기록).
            // 사용자 지정 바인딩보다 앞에 두지 않는다 — 아래 ⌃⌥⌘M 주석의 등록 순서 규칙과 같다.
            bindings.append(.init(id: 10, keyCode: UInt32(kVK_ANSI_P),
                                  modifiers: UInt32(controlKey | optionKey | cmdKey)) { [weak self] in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    // **2단계 토글이다 (커버 ↔ 뷰어). 3단계로 두지 말 것.**
                    // 처음엔 '자동'을 끼워 3단계로 만들었는데, 측정하는 사람이 "몇 번 눌러야 하나"를
                    // 세야 했다. A/B는 두 팔뿐이므로 한 번 누르면 반대 팔로 가는 게 맞다.
                    // 핀은 메모리에만 있어 앱을 다시 켜면 자동 파생으로 돌아간다.
                    self.placementPin = (self.placementPin == .coverSource) ? .viewerWindow : .coverSource
                    let label = self.placementPin == .coverSource ? "cover(핀)" : "viewer(핀)"
                    // 핀을 바꿨으면 즉시 반영 — AUTOFS가 잡고 있던 배치도 이 대입이 덮는다.
                    self.selectedOverlayPlacement = self.placementPin
                        ?? (self.upscaleMode == .off ? .coverSource : .viewerWindow)
                    if self.isCapturing { self.updateOverlayPlacement() }
                    DiagnosticLog.shared.log("[PLACE] 핀 → \(label)")
                }
            })
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
    /// `splitQueue`를 함께 받는다 — `splitQueueEnabled`도 렌더 스레드 전용 판독값이라
    /// 엔진 교체와 같은 직렬화 지점에서 기입해야 규약이 지켜진다.
    private func swapPairEngine(_ newEngine: (any PairInterpolationEngine)?,
                                splitQueue: Bool? = nil) {
        if renderDriver.isRunning {
            // perform은 동기 + 틱과 직렬 — 블록 실행 중 메인은 대기하므로 실제 동시 접근 없음
            nonisolated(unsafe) let engineRef = newEngine
            renderDriver.perform { [weak self] in
                self?.pairEngine = engineRef
                if let splitQueue { self?.splitQueueEnabled = splitQueue }
            }
        } else {
            pairEngine = newEngine
            if let splitQueue { splitQueueEnabled = splitQueue }
        }
    }

    /// configurePairEngine 세대 — MainActor async라 prepare await 중 재진입이 교차하면
    /// "늦게 끝난 쪽"이 무조건 이겨 UI 선택과 실제 엔진이 불일치하고 패자 엔진 shutdown이
    /// 누락된다 (리뷰 확정: RIFE 로드 수백 ms 중 빠른 모드 재변경). await마다 세대 검사,
    /// stopCapture도 +1로 진행 중 설치를 무효화.
    private var configureEpoch = 0

    private func configurePairEngine() async {
        // 큐 분리는 predict 대기가 있는 엔진(RIFE)에서만 이득 — 위 선언부 주석의 실측 근거 참조.
        // **기입은 아래 swapPairEngine(nil)과 함께 렌더 드라이버로 직렬화한다.** 이 값은 렌더
        // 스레드 전용 함수(drainAndIngest, checkRenderThread로 강제)가 읽으므로 MainActor에서
        // 그냥 대입하면 규약 위반이다. 지금은 무해하다 — cb2는 splitQ와 무관하게 항상
        // stableReadyEvent를 기다리므로(2026-07-26 색노이즈 실측 이후) 최악이 "한 프레임의 cb1이
        // 반대 큐로 감"이고 `?? workQueue` 폴백이 nil을 막는다 — 그러나 그 무해함은 cb2의 대기
        // 규칙에 딸린 것이라, 그쪽이 바뀌면 조용히 깨진다.
        let wantSplit = splitQueueOverride ?? (selectedRenderMode == .rife && isInterpolationEnabled)
        configureEpoch += 1
        let epoch = configureEpoch
        let old = pairEngine
        swapPairEngine(nil, splitQueue: wantSplit)   // 틱이 더는 old를 못 보게 먼저 떼어낸 뒤
        old?.shutdown()              // 안전하게 해체 (렌더 스레드는 이미 nil만 봄)

        guard isInterpolationEnabled else {
            interpolationEngine = "Off"
            return
        }

        // **정지-UI 파라미터를 노브로 연다 (B2, 2026-09-01).**
        // 지금까지 이 값들은 하드코딩 static var라 **런타임 A/B 자체가 불가능**했다 —
        // 「열린 이슈」 B2의 "런타임 실측 미실시 / 파라미터 튜닝 미완"이 그래서다.
        // 매 configure마다 읽어 캡처를 껐다 켜면 새 값이 먹는다(앱 재시작 불요).
        // 값 범위는 셰이더 가정에 맞춰 좁게 클램프한다: clo < chi가 아니면 smoothstep이 뒤집힌다.
        if let v = Knob.double("MACFG_UIALPHA") { UIStaticDetector.alpha = Float(min(max(v, 0.005), 0.5)) }
        if let v = Knob.double("MACFG_UISTRENGTH") { UIStaticDetector.strength = Float(min(max(v, 0), 1)) }
        if let v = Knob.double("MACFG_UICLO") { UIStaticDetector.clo = Float(max(v, 0)) }
        if let v = Knob.double("MACFG_UICHI") { UIStaticDetector.chi = Float(max(v, 0)) }
        if UIStaticDetector.chi <= UIStaticDetector.clo {
            UIStaticDetector.chi = UIStaticDetector.clo + 0.1   // 뒤집힘 방지
        }
        if let v = Knob.int("MACFG_UIMASK") { UIStaticDetector.enabled = v != 0 }
        if let v = Knob.int("MACFG_UIMASKDIV") { UIStaticDetector.maskDiv = max(1, min(v, 8)) }
        if let v = Knob.double("MACFG_UIEPS") { UIStaticDetector.noiseEps = Float(max(v, 0.0001)) }
        DiagnosticLog.shared.log(String(format: "[UISTATIC] alpha=%.3f clo=%.2f chi=%.2f strength=%.2f div=%d enabled=%@",
                                        UIStaticDetector.alpha, UIStaticDetector.clo, UIStaticDetector.chi,
                                        UIStaticDetector.strength, UIStaticDetector.maskDiv,
                                        UIStaticDetector.enabled ? "1" : "0"))

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
