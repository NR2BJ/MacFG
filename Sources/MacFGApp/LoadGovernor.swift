import Foundation
import Monitoring

/// 부하 거버너 — 성능이 예산을 못 맞출 때 화질 다이얼을 단계적으로 낮추고, 여유가 생기면 되돌린다.
///
/// 왜 필요한가: GPU 경합(WindowServer + 브라우저 4K 디코드) 상황에서 프레임당 work가 30ms →
/// 85ms로 붕괴하는 것이 실측됐다. 그때 파이프라인은 아무 대응도 하지 않아 보간이 깨진 채로
/// 계속 돌았다. 저사양(M1/8GB)은 이 상태가 상시일 수 있다. 커널 미시 최적화로는 못 메우는
/// 폭이라, "무엇을 포기할지"를 정하는 계층이 따로 필요하다.
///
/// 설계 원칙:
///  - **입력은 전부 기존 계측 재사용** — 추가 측정 비용 0.
///  - **레벨은 인코딩 파라미터만 조작** — 페이싱(latencyOffset/present 선택)·색·타임스탬프는
///    건드리지 않는다. 따라서 강등이 σ나 지연을 직접 악화시키지 않는다.
///  - **AIMD + 히스테리시스** — 강등은 빠르게(붕괴 중이므로), 복귀는 느리게(플래핑 방지).
///  - **엔진 자율 사다리와 충돌 금지** — 거버너는 *상한*만 내린다. RIFE의 자체 해상도 사다리는
///    그 상한 안에서 평소대로 동작한다. 같은 다이얼을 두 컨트롤러가 올리고 내리면 발진한다.
@MainActor
@Observable
public final class LoadGovernor {
    /// 강등 단계. 숫자가 클수록 더 많이 포기한다.
    public enum Level: Int, Comparable, Sendable {
        case full = 0        // 제한 없음
        case light = 1       // 업스케일 체인 경량화 + flow 해상도 한 단
        case heavy = 2       // flow 해상도 최저 + 보간 배율 상한 2x
        case bypass = 3      // 보간 중단(원본 패스스루) — 엔진은 살려둬 즉시 복귀 가능

        public static func < (a: Level, b: Level) -> Bool { a.rawValue < b.rawValue }
    }

    public private(set) var level: Level = .full
    /// 사용자가 수동 고정한 상한 (nil이면 자동)
    public var manualCap: Level?
    /// 마지막 전이 사유 (UI/로그용)
    public private(set) var lastReason: String = ""

    /// 강제 레벨 (A/B 측정용) — MACFG_GOV_FORCE=0..3
    private let forced: Level?
    /// 거버너 자체를 끄기 — MACFG_GOV=0
    private let enabled: Bool

    private var badWindows = 0
    private var goodWindows = 0
    private var lastChangeAt: CFAbsoluteTime = 0

    public init() {
        let env = ProcessInfo.processInfo.environment
        enabled = env["MACFG_GOV"] != "0"
        forced = env["MACFG_GOV_FORCE"].flatMap { Int($0) }.flatMap { Level(rawValue: $0) }
        if let forced {
            level = forced
            lastReason = "강제(MACFG_GOV_FORCE=\(forced.rawValue))"
        }
    }

    /// 부하 신호. 전부 기존 계측에서 가져온다 (adaptPacing이 2초마다 호출).
    public struct Signals: Sendable {
        /// 프레임당 work의 p90 (ms) — 강등 판정용. 감쇠가 느려(10%/2s) 스파이크에 견고.
        public let workP90Ms: Double
        /// 이번 창의 work 평균 (ms) — 복귀 판정용 즉응 신호. p90으로 복귀를 재면 부하가
        /// 걷힌 뒤에도 30초 넘게 갇힌다(감쇠 속도 때문, 실측).
        public let workAvgMs: Double
        /// 소스 프레임 간격 (ms) — 예산의 기준
        public let sourceIntervalMs: Double
        /// 실제 틱 레이트 (Hz) — 렌더 루프가 굶주리는지
        public let tickHz: Double
        /// 목표 주사율 (Hz)
        public let refreshHz: Double
        /// 이 창에서 놓친 프레임 수
        public let missCount: Int
        /// present 처리량 / 이론 상한 (1.0=목표 달성). **주 과부하 신호.**
        public let presentRatio: Double

        public init(workP90Ms: Double, workAvgMs: Double, sourceIntervalMs: Double, tickHz: Double, refreshHz: Double, missCount: Int, presentRatio: Double) {
            self.workP90Ms = workP90Ms
            self.workAvgMs = workAvgMs
            self.sourceIntervalMs = sourceIntervalMs
            self.tickHz = tickHz
            self.refreshHz = refreshHz
            self.missCount = missCount
            self.presentRatio = presentRatio
        }
    }

    /// 신호를 먹여 레벨을 갱신한다. 2초 창마다 1회 호출 전제.
    public func update(_ s: Signals) {
        guard enabled, forced == nil else { return }

        // 과부하 판정은 **결과**로 한다: 목표만큼 프레임을 내보내고 있는가.
        //
        // work(프레임이 파이프라인을 통과하는 지연)를 소스 간격과 비교하면 안 된다 — 파이프라인이
        // 병렬이라 work 15ms로도 120fps를 온전히 뽑는다. 실제로 그 오판 때문에 RIFE의 정상
        // 동작(work 15ms > 예산 13.4ms)을 과부하로 보고 강등 → 바이패스에서 work가 떨어지니
        // 복귀 → 다시 강등이 반복되는 발진이 관측됐다(실측 3→2→3→2).
        // 진짜 붕괴(85ms)는 present 처리량이 같이 무너지므로 이 신호로 잡힌다.
        // tickHz == 0은 "아직 측정 전"이지 굶주림이 아니다 — 이 구분을 안 하면 캡처 시작
        // 첫 창마다 강등됐다가 10초 뒤 복귀하는 헛왕복이 매번 일어난다(실측 전 런 강등=2).
        let tickStarved = s.refreshHz > 0 && s.tickHz > 1 && s.tickHz < s.refreshHz * 0.85
        let shortfall = s.presentRatio < 0.85
        let budget = min(max(s.sourceIntervalMs * 0.8, 8.0), 33.0)   // 로그 표시용

        if shortfall || tickStarved {
            badWindows += 1
            goodWindows = 0
        } else if s.presentRatio >= 0.97 && !tickStarved {
            // 복귀 판정에 miss 수를 넣지 않는다: 강등 상태(특히 바이패스)에선 프레임 공급
            // 패턴 자체가 달라 miss가 상시 0이 아니고, 그러면 영영 복귀하지 못한다(실측:
            // 부하 종료 후 work 4ms인데 L3에 30초+ 갇힘).
            //
            // 또한 강등 상태의 work는 "능력"이 아니라 "줄어든 부하"를 재는 순환 논리다.
            // 그래서 복귀는 판정이 아니라 **탐침**이다 — 한 단 올려보고, 그 부하에서 다시
            // 예산을 넘으면 위 강등 로직이 2창(≈4s) 만에 되돌린다. 복귀가 강등보다 훨씬
            // 느린(5창/8s vs 2창/3s) 이유도 이 탐침 비용을 드물게 치르기 위함이다.
            goodWindows += 1
            badWindows = 0
        } else {
            // 중간 지대 — 유지 (히스테리시스: 경계에서 진동하지 않게)
            badWindows = 0
            goodWindows = 0
        }

        if ProcessInfo.processInfo.environment["MACFG_GOV_DEBUG"] == "1" {
            DiagnosticLog.shared.log(String(format: "[GOV?] L%d ratio=%.2f work=%.1f tick=%.0f/%.0f miss=%d good=%d bad=%d",
                level.rawValue, s.presentRatio, s.workAvgMs, s.tickHz, s.refreshHz, s.missCount, goodWindows, badWindows))
        }
        let now = CFAbsoluteTimeGetCurrent()
        // 강등은 2창(≈4s) 연속이면 즉시 — 붕괴 중엔 빨리 손을 써야 한다
        if badWindows >= 2, level < .bypass, now - lastChangeAt > 3.0 {
            let next = Level(rawValue: level.rawValue + 1) ?? .bypass
            apply(next, reason: String(format: "처리량 %.0f%% (work %.0fms)%@", s.presentRatio * 100, s.workAvgMs, tickStarved ? " + 틱 굶주림" : ""), at: now)
            badWindows = 0
        }
        // 복귀는 5창(≈10s) 연속 여유일 때 한 단계씩 — 천천히 (플래핑 방지)
        else if goodWindows >= 5, level > .full, now - lastChangeAt > 8.0 {
            let next = Level(rawValue: level.rawValue - 1) ?? .full
            apply(next, reason: String(format: "처리량 회복 %.0f%%", s.presentRatio * 100), at: now)
            goodWindows = 0
        }
    }

    private func apply(_ new: Level, reason: String, at now: CFAbsoluteTime) {
        var target = new
        if let cap = manualCap, target > cap { target = cap }
        guard target != level else { return }
        let old = level
        level = target
        lastReason = reason
        lastChangeAt = now
        DiagnosticLog.shared.log("[GOV] \(old.rawValue)→\(target.rawValue) — \(reason)")
    }

    /// 캡처 시작 시 호출 — 기기 등급으로 시작 레벨을 시딩한다.
    /// 시딩은 어디까지나 *시작값*이고, 정착값은 위 실측 신호가 결정한다. 오분류의 피해는
    /// "처음 몇 초 보수적 화질"로 상한이 잡히고, 반대로 저사양에서 첫 수 초 붕괴하며
    /// 적응지연이 헛램프하던 낭비를 막는다.
    public func seedForDevice(gpuCoreCount: Int, memoryGB: Double, sourcePixels: Int) {
        guard enabled, forced == nil else { return }
        // 4K급 소스 + 낮은 등급이면 한 단 낮춰 시작
        let heavySource = sourcePixels >= 7_000_000
        let lowEnd = gpuCoreCount <= 8 || memoryGB <= 8.5
        let seeded: Level = (heavySource && lowEnd) ? .light : .full
        if seeded != level {
            level = seeded
            lastReason = "기기 시딩 (GPU \(gpuCoreCount)코어, \(String(format: "%.0f", memoryGB))GB, 소스 \(sourcePixels / 1_000_000)MP)"
            DiagnosticLog.shared.log("[GOV] 시작 레벨 \(seeded.rawValue) — \(lastReason)")
        }
        badWindows = 0; goodWindows = 0
        lastChangeAt = CFAbsoluteTimeGetCurrent()
    }

    public func reset() {
        guard forced == nil else { return }
        level = manualCap ?? .full
        badWindows = 0; goodWindows = 0
        lastChangeAt = 0
        lastReason = ""
    }

    // MARK: - 레벨 → 다이얼 (소비자들이 읽는다)

    /// MetalFlow flow 해상도 상한 (긴 변). nil이면 제한 없음.
    public var flowBaseCap: Double? {
        switch level {
        case .full:   return nil
        case .light:  return 800
        case .heavy:  return 640
        case .bypass: return 640
        }
    }

    /// 보간 배율 상한. nil이면 제한 없음.
    public var multiplierCap: Int? {
        switch level {
        case .full, .light: return nil
        case .heavy:        return 2
        case .bypass:       return 1   // 보간 없음
        }
    }

    /// MetalFX 공간 업스케일을 쓸 수 있는지 (ANE 2x는 GPU-free라 계속 유지).
    public var allowsMetalFX: Bool { level == .full }

    /// 보간 자체를 건너뛸지 (원본 패스스루)
    public var bypassInterpolation: Bool { level == .bypass }

    /// 사용자에게 보일 상태 문자열 (nil이면 표시 안 함)
    public var statusText: String? {
        switch level {
        case .full:   return nil
        case .light:  return "성능 보호 (경량)"
        case .heavy:  return "성능 보호 (강함)"
        case .bypass: return "성능 보호 — 보간 일시 중단"
        }
    }
}
