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

    private let enabled: Bool
    private let debug: Bool
    /// 사용자가 --flow-base로 고정했으면 자동 조절 중지
    public var manualOverride = false

    private var idx: Int = 5                       // rungs[5] = 1440
    private var ceilingIdx: Int = rungs.count - 1  // 학습된 천장
    private var goodWindows = 0
    private var badWindows = 0
    private var lastChangeAt: CFAbsoluteTime = 0
    private var frozenUntil: CFAbsoluteTime = 0
    /// 하강 직전의 달성도 — 하강이 효과 있었는지 다음 창에서 비교
    private var achievedBeforeDescent: Double = -1
    private var uselessDescents = 0
    private var lastAscendIdx: Int = -1
    private var lastAscendAt: CFAbsoluteTime = 0

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
        lastChangeAt = CFAbsoluteTimeGetCurrent()
        goodWindows = 0; badWindows = 0; uselessDescents = 0
        achievedBeforeDescent = -1
    }

    /// 2초 창마다 호출. 반환값이 바뀌면 호출자가 엔진에 반영한다.
    /// - achievedRatio: 목표 대비 달성도 (1.0 = 목표 프레임 완전 달성)
    /// - engineMs: 보간 엔진의 GPU 실행시간 EMA
    /// - budgetMs: 쌍당 예산 (소스 간격)
    @discardableResult
    public func update(achievedRatio: Double, engineMs: Double, budgetMs: Double) -> Double {
        guard enabled, !manualOverride, budgetMs > 0 else { return current }
        let now = CFAbsoluteTimeGetCurrent()

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
                    let restore = min(idx + 2, ceilingIdx)
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
        if goodWindows >= 4, idx < ceilingIdx, share < 0.5, now - lastChangeAt > 8.0 {
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
        goodWindows = 0; badWindows = 0; uselessDescents = 0
        achievedBeforeDescent = -1
        lastChangeAt = CFAbsoluteTimeGetCurrent()
    }
}
