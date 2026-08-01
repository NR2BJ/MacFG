import Foundation
import Monitoring

/// MetalFlow의 flow 해상도(사다리)를 **목표 프레임을 맞출 때까지 자동으로** 조절한다.
///
/// 왜: flow 해상도는 화질(빠른 모션의 큰 변위 추적)과 GPU 비용(≈해상도²)을 동시에 좌우하는데,
/// 적정값은 기기마다 다르다 — M1은 낮게, M4는 중간, M5 Max는 높게. 고정 기본값은 한쪽을 반드시
/// 손해 본다(저사양=버벅임 / 고사양=화질 낭비). 그래서 실측으로 각 기기의 한계까지 올린다.
///
/// 설계 원칙 (이 세션에서 거버너가 두 번 실패한 경험 반영):
///  - **하강엔 근거가 필요하다.** 목표 미달이라고 무조건 내리지 않는다. 엔진 GPU 비용이 예산의
///    의미 있는 몫일 때만 내린다 — 배달 지터/present 경합이 원인이면 flow를 낮춰도 안 고쳐지고
///    화질만 잃는다.
///  - **효과 없으면 되돌린다.** 내린 뒤 실제로 목표가 개선됐는지 확인하고, 두 번 연속 무효면
///    원위치 + 일정 시간 동결. "화질을 팔았는데 산 게 없는" 상태를 스스로 빠져나온다.
///  - **올릴 땐 천장을 학습한다.** 올려보고 실패하면 그 칸을 천장으로 기억해(TCP ssthresh 식)
///    같은 자리에서 오르내리는 발진을 막는다.
///  - **변경엔 비용이 있다.** 해상도 변경 = 피라미드 텍스처 재할당 + 시간적 prior 리셋이라
///    최소 간격을 두고 히스테리시스를 건다.
@MainActor
public final class AutoFlowScaler {
    /// flow 해상도 사다리 (긴 변 px). 4K 실측(M4): 960≈6ms · 1440≈6.2ms · 1920≈8.9ms · 2160≈10.3ms.
    public static let rungs: [Double] = [480, 640, 800, 960, 1200, 1440, 1728, 2160]

    /// 현재 선택된 flow 해상도
    public private(set) var current: Double = 1440
    /// 마지막 전이 사유 (로그/UI용)
    public private(set) var lastReason: String = ""
    /// 현재 학습된 천장 (긴 변 px) — 진단 표시 및 테스트 관측용.
    public var learnedCeiling: Double { Self.rungs[ceilingIdx] }

    private let enabled: Bool
    private let debug: Bool
    /// 사용자가 --flow-base로 고정했으면 자동 조절 중지
    public var manualOverride = false

    private var idx: Int = 5                       // rungs[5] = 1440
    /// 학습된 천장. 초기값 = 1440(idx 5) — **실프레임 측정 근거**: 실제 4K 덤프 4세트 × flow 640~2160
    /// 삼중항 PSNR에서 해상도에 따른 화질 변화가 ±0.5dB 안이고 방향도 콘텐츠마다 갈렸다(중립).
    /// 반면 비용은 4K에서 960≈6ms → 1728≈8.9ms → 2160≈10.3ms로 확실히 는다. 즉 그 위로 올리는 건
    /// GPU만 쓰고 화질은 0이므로 탐침 자체를 막고, 남는 여유는 틱 안정성/발열/저사양 여유로 남긴다.
    private var ceilingIdx: Int = 5
    /// 학습 천장의 절대 상한 — 위 근거대로 그 위는 비용만 늘고 화질 이득이 0이라 탐침 자체를 막는다.
    /// 천장은 이 값까지만 **회복**할 수 있다(넘어서 오르지 않는다).
    private let maxCeilingIdx: Int = 5
    /// 천장 칸에서 연속 달성한 창 수 — 천장 회복의 근거.
    private var goodAtCeiling = 0
    private var goodWindows = 0
    private var badWindows = 0
    private var lastChangeAt: CFAbsoluteTime = 0
    private var frozenUntil: CFAbsoluteTime = 0
    /// 하강 직전의 달성도 — 하강이 효과 있었는지 다음 창에서 비교
    private var achievedBeforeDescent: Double = -1
    private var uselessDescents = 0
    private var lastAscendIdx: Int = -1
    private var lastAscendAt: CFAbsoluteTime = 0

    /// 시계 주입 — 변경 최소간격/동결 같은 시간 기반 히스테리시스를 테스트에서 결정적으로 돌리기 위함.
    /// 실사용에선 실제 시각을 쓴다.
    public var nowProvider: () -> CFAbsoluteTime = { CFAbsoluteTimeGetCurrent() }

    public init() {
        let env = ProcessInfo.processInfo.environment
        enabled = env["MACFG_AUTOFLOW"] != "0"
        debug = env["MACFG_AUTOFLOW_DEBUG"] == "1"
    }

    /// 기기 등급으로 시작점 시딩 — 수렴을 몇 창 앞당길 뿐, 정착값은 실측이 정한다.
    public func seed(gpuCoreCount: Int, sourcePixels: Int) {
        guard enabled, !manualOverride else { return }
        // 4K급 소스는 한 칸 보수적으로 (같은 코어수라도 픽셀이 4배)
        let heavy = sourcePixels >= 7_000_000
        let start: Int
        switch gpuCoreCount {
        case ..<9:   start = heavy ? 2 : 3      // M1/M2 base (8코어) → 800/960
        case 9...11: start = heavy ? 4 : 5      // M3/M4 base (10코어) → 1200/1440
        case 12...20: start = heavy ? 5 : 6     // Pro → 1440/1728
        default:     start = heavy ? 6 : 7      // Max/Ultra → 1728/2160
        }
        idx = min(max(start, 0), Self.rungs.count - 1)
        current = Self.rungs[idx]
        lastReason = "기기 시딩 (GPU \(gpuCoreCount)코어, 소스 \(sourcePixels / 1_000_000)MP)"
        DiagnosticLog.shared.log("[AUTOFLOW] 시작 \(Int(current)) — \(lastReason)")
        lastChangeAt = nowProvider()
        goodWindows = 0; badWindows = 0; uselessDescents = 0; goodAtCeiling = 0
        achievedBeforeDescent = -1
    }

    /// 두 신호를 하나의 달성도로 합친다 — 호출자가 쓰기 전에 반드시 통과시켜야 하는 관문.
    ///
    /// - tickRatio: 디스플레이에 실제로 프레임을 낸 비율 (틱Hz / 주사율). **진짜 배달 성적**.
    /// - keepRatio: 만든 보간 프레임 중 살아남은 비율 (1 - 폐기율).
    ///
    /// keepRatio를 그대로 달성도로 쓰면 안 된다. 폐기(staleDrop)는 배달 실패가 아니라
    /// **필요보다 많이 만들어 늦은 걸 버린 낭비**다. 틱이 주사율을 온전히 내는 동안에도 폐기가
    /// 7%만 넘으면 달성도가 하강 문턱(0.93) 밑으로 떨어졌고, flow를 내리면 GPU 시간이 줄어
    /// 폐기가 **조금** 개선되므로 "하강이 유효했다"고 판정돼 다시 내려간다 — 자기강화 하강이다.
    /// 실측(4K/M4): 프레임은 120으로 멀쩡한데 flow만 1200→480까지 단조 하강했다.
    /// 그래서 폐기는 정말 심각할 때(생산의 1/5 이상을 버릴 때)만 미달 신호로 인정한다.
    /// - deliveryRatio: **소스 프레임 중 실제로 파이프라인에 들어간 비율** (1 - 풀고갈 폐기율).
    ///
    /// deliveryRatio가 왜 필요한가 — 이게 없어서 제어기가 붕괴를 성공으로 읽었다.
    /// 풀이 고갈되면 소스 프레임을 **cb1 인코딩 전에 통째로 버린다.** 그래도 타임라인엔
    /// 이미 만든 프레임이 남아 있어 **틱은 주사율을 그대로 낸다.** keepRatio도 무사하다 —
    /// 그건 "우리가 만든 보간 프레임" 중 살아남은 비율이라 애초에 만들지 못한 프레임은 안 센다.
    /// 실측(2026-08-01, MetalFlow 4K): work 97~140ms · e2e 164~179ms · **소스의 40%가 파괴**되는
    /// 붕괴 구간 내내 tick=144.0Hz, achieved=1.00 → 스케일러는 "여유"로 판정해 화질을 **올리려
    /// 탐침**했다. r(poolMiss, work)=0.90이고 AppleFI도 같은 서명을 보이므로 엔진 문제가 아니라
    /// 파이프라인 병리다. 소스 손실은 낭비가 아니라 **진짜 손실**이므로 그대로 달성도에 넣는다.
    /// 다만 리사이즈 순간의 한 창짜리 튐으로 하강이 걸리지 않게 5% 문턱을 둔다.
    ///
    /// (순수 함수 — feedLoadGovernor가 렌더 스레드에서 부르므로 nonisolated)
    public nonisolated static func combinedAchieved(tickRatio: Double,
                                                    keepRatio: Double,
                                                    deliveryRatio: Double = 1.0) -> Double {
        let tick = max(0, min(1.0, tickRatio))
        let keep = max(0, min(1.0, keepRatio))
        let delivered = max(0, min(1.0, deliveryRatio))
        let base = keep < 0.80 ? min(tick, keep) : tick
        return delivered < 0.95 ? min(base, delivered) : base
    }

    /// "하강이 무효였으니 원복" 시 되돌아갈 칸. **max(ceilingIdx, idx)가 핵심**:
    /// 천장이 현재 칸보다 낮으면(Pro/Max 시딩은 천장 위에서 출발한다) min(idx+2, ceilingIdx)가
    /// 현재보다 아래를 가리켜, "원복"이라 로그하면서 실제로는 화질을 더 깎는다.
    /// 원복은 정의상 **올라가거나 제자리**여야 한다.
    public nonisolated static func restoreIndex(from idx: Int, ceilingIdx: Int) -> Int {
        min(idx + 2, max(ceilingIdx, idx))
    }

    /// 2초 창마다 호출. 반환값이 바뀌면 호출자가 엔진에 반영한다.
    /// - achievedRatio: 목표 대비 달성도 (1.0 = 목표 프레임 완전 달성)
    /// - engineMs: 보간 엔진의 GPU 실행시간 EMA
    /// - budgetMs: 쌍당 예산 (소스 간격)
    @discardableResult
    public func update(achievedRatio: Double, engineMs: Double, budgetMs: Double) -> Double {
        guard enabled, !manualOverride, budgetMs > 0 else { return current }
        let now = nowProvider()

        // 엔진이 예산에서 차지하는 비중 — 하강이 유효할 수 있는지의 근거.
        // 25% 미만이면 flow를 반으로 줄여도 전체가 거의 안 변한다 → 화질만 손해.
        let share = engineMs / budgetMs
        let computeRelevant = share >= 0.25
        let hitting = achievedRatio >= 0.97
        let missing = achievedRatio < 0.93

        if debug {
            DiagnosticLog.shared.log(String(format:
                "[AUTOFLOW?] base=%.0f achieved=%.2f engine=%.1fms/%.1fms share=%.0f%% good=%d bad=%d ceil=%.0f%@",
                current, achievedRatio, engineMs, budgetMs, share * 100, goodWindows, badWindows,
                Self.rungs[ceilingIdx], now < frozenUntil ? " [동결]" : ""))
        }

        // 직전 하강의 효과 판정 — 개선이 없으면 화질만 판 것이므로 되돌린다.
        if achievedBeforeDescent >= 0 {
            let gained = achievedRatio - achievedBeforeDescent
            achievedBeforeDescent = -1
            if gained < 0.015 {
                uselessDescents += 1
                if uselessDescents >= 2 {
                    // 두 번 내렸는데 목표가 안 올랐다 = 병목이 flow가 아니다. 원위치 + 동결.
                    // **max(ceilingIdx, idx) 필수**: 천장이 학습으로 현재 칸보다 낮아져 있으면
                    // min(idx+2, ceilingIdx)가 현재보다 **아래**를 가리켜, "원복"이라 로그하면서
                    // 실제로는 더 내려가 버린다(실측: 1200→480 단조 하강의 절반이 이 경로였다).
                    let restore = Self.restoreIndex(from: idx, ceilingIdx: ceilingIdx)
                    if restore != idx {
                        idx = restore
                        current = Self.rungs[idx]
                        lastReason = "하강 무효 → 원복 (병목이 flow 아님)"
                        DiagnosticLog.shared.log("[AUTOFLOW] ↩︎ \(Int(current)) — \(lastReason)")
                        lastChangeAt = now
                    }
                    frozenUntil = now + 30
                    uselessDescents = 0
                    badWindows = 0
                    return current
                }
            } else {
                uselessDescents = 0   // 효과 있었다 — 계속 내려도 됨
            }
        }

        if missing && computeRelevant {
            badWindows += 1; goodWindows = 0
        } else if hitting {
            goodWindows += 1; badWindows = 0
        } else {
            badWindows = 0; goodWindows = 0   // 중간지대 유지
        }

        // 천장 회복 — 천장 학습은 내려가기만 해서, 일시적 과부하 한 번이 세션 전체의 화질 상한을
        // 영구히 깎았다(실측: 4K에서 1200으로 시작해 480까지 흘러내린 뒤 복귀 불가. 상승 조건이
        // idx < ceilingIdx라 천장이 바닥이면 영영 못 오른다). 천장 칸에서 충분히 오래 안정적이면
        // 한 칸 돌려주어 다시 탐침할 기회를 준다. 상한(maxCeilingIdx)은 넘지 않는다.
        if hitting, idx >= ceilingIdx, ceilingIdx < maxCeilingIdx {
            goodAtCeiling += 1
            if goodAtCeiling >= 15 {          // ≈30s 연속 달성
                ceilingIdx += 1
                goodAtCeiling = 0
                DiagnosticLog.shared.log("[AUTOFLOW] 천장 회복 → \(Int(Self.rungs[ceilingIdx]))")
            }
        } else if !hitting {
            goodAtCeiling = 0
        }

        guard now >= frozenUntil else { return current }

        // 하강: 2창(≈4s) 연속 미달 + 컴퓨트 근거. 빠르게 (버벅임 중이므로)
        if badWindows >= 2, idx > 0, now - lastChangeAt > 3.0 {
            // 방금 올린 직후 실패면 그 칸을 천장으로 학습 (발진 방지)
            if lastAscendIdx == idx, now - lastAscendAt < 20 {
                ceilingIdx = max(0, idx - 1)
                DiagnosticLog.shared.log("[AUTOFLOW] 천장 학습 → \(Int(Self.rungs[ceilingIdx]))")
            }
            achievedBeforeDescent = achievedRatio
            idx -= 1
            current = Self.rungs[idx]
            lastReason = String(format: "목표 미달 %.0f%% (엔진 %.0f%% 점유)", achievedRatio * 100, share * 100)
            DiagnosticLog.shared.log("[AUTOFLOW] ↓ \(Int(current)) — \(lastReason)")
            lastChangeAt = now
            badWindows = 0
            return current
        }

        // 상승: 4창(≈8s) 연속 달성 + 엔진 여유. 천천히 (탐침 비용을 드물게)
        // engineMs>0 필수 — 보간이 안 도는 구간(비활성/엔진 전환/종료)에선 EMA가 0이라 share=0이
        // 되어 "여유"로 오판, 근거 없이 올라간다(실측 "↑ 960 — 엔진 0% 점유"). 데이터 없으면 유지.
        if goodWindows >= 4, idx < ceilingIdx, engineMs > 0.05, share < 0.5, now - lastChangeAt > 8.0 {
            lastAscendIdx = idx + 1
            lastAscendAt = now
            idx += 1
            current = Self.rungs[idx]
            lastReason = String(format: "여유 (엔진 %.0f%% 점유) — 화질 상향 탐침", share * 100)
            DiagnosticLog.shared.log("[AUTOFLOW] ↑ \(Int(current)) — \(lastReason)")
            lastChangeAt = now
            goodWindows = 0
        }
        return current
    }

    /// 캡처 재시작 시 — 학습을 유지하되 카운터만 리셋 (같은 기기면 정착값이 유효)
    public func softReset() {
        goodWindows = 0; badWindows = 0; uselessDescents = 0; goodAtCeiling = 0
        achievedBeforeDescent = -1
        lastChangeAt = nowProvider()
    }
}
