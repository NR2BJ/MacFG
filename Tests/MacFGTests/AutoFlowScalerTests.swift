import Testing
import Foundation
@testable import Monitoring

/// AutoFlowScaler 상태기계 검증.
///
/// 왜 단위 테스트인가: 이 컨트롤러의 안전장치(특히 "내렸는데 효과 없으면 원복")는 **non-compute 병목**
/// 상황에서만 발동하는데, 그 조건을 실기기에서 합성으로 만들기가 어렵다(실측 시도: 4K + 1440p 창 3개를
/// 띄워도 achieved 0.99~1.00로 하강 자체가 안 일어남). 반면 로직 자체는 순수 함수라 신호를 주입하면
/// 결정적으로 검증된다. 시간 기반 히스테리시스는 nowProvider 주입으로 통제한다.
@MainActor
struct AutoFlowScalerTests {

    /// 테스트용 스케일러 — 가짜 시계로 최소 변경간격/동결을 통제
    private func makeScaler(startIdx: Int = 5) -> (AutoFlowScaler, () -> Void) {
        let s = AutoFlowScaler()
        var t: CFAbsoluteTime = 1_000_000
        s.nowProvider = { t }
        s.seed(gpuCoreCount: 10, sourcePixels: 8_000_000)
        return (s, { t += 10 })   // 한 호출 = 10초 경과 (상승 8s/하강 3s 게이트 통과)
    }

    /// 여유가 지속되면 화질을 올린다 (상승 경로)
    @Test func ascendsWhenHeadroomPersists() {
        let (s, tick) = makeScaler()
        let start = s.current
        for _ in 0..<6 {
            tick()
            s.update(achievedRatio: 1.0, engineMs: 5.0, budgetMs: 16.7)   // share 30% < 50% = 여유
        }
        #expect(s.current > start, "여유가 4창 이상 지속되면 한 칸 올라가야 한다")
    }

    /// **컴퓨트 근거가 있을 때만 내려간다** — 목표 미달이어도 엔진이 예산의 25% 미만이면 유지.
    /// 배달 지터/present 경합이 원인이면 flow를 낮춰도 안 고쳐지고 화질만 잃기 때문.
    @Test func doesNotDescendWithoutComputeEvidence() {
        let (s, tick) = makeScaler()
        let start = s.current
        for _ in 0..<8 {
            tick()
            s.update(achievedRatio: 0.70, engineMs: 1.0, budgetMs: 16.7)  // share 6% = 컴퓨트 무관
        }
        #expect(s.current == start, "컴퓨트 근거 없이는 목표 미달이어도 내려가면 안 된다")
    }

    /// 컴퓨트 과부하가 근거로 있으면 내려간다 (하강 경로)
    @Test func descendsWithComputeEvidence() {
        let (s, tick) = makeScaler()
        let start = s.current
        for _ in 0..<4 {
            tick()
            s.update(achievedRatio: 0.70, engineMs: 8.0, budgetMs: 16.7)  // share 48% = 컴퓨트 바운드
        }
        #expect(s.current < start, "목표 미달 + 컴퓨트 근거면 내려가야 한다")
    }

    /// **핵심 안전장치**: 내렸는데 목표가 개선되지 않으면 되돌리고 동결한다.
    /// (화질을 팔았는데 산 게 없는 상태를 스스로 빠져나오는 경로)
    @Test func restoresWhenDescentDoesNotHelp() {
        let (s, tick) = makeScaler()
        let start = s.current
        // 컴퓨트 근거는 있지만 아무리 내려도 달성도가 그대로인 상황을 계속 먹인다
        var lowest = start
        for _ in 0..<14 {
            tick()
            s.update(achievedRatio: 0.70, engineMs: 8.0, budgetMs: 16.7)
            lowest = min(lowest, s.current)
        }
        #expect(lowest < start, "먼저 내려가긴 해야 한다(전제)")
        #expect(s.current > lowest,
                "두 번 연속 무효한 하강 뒤에는 원위치해야 한다 — 개선 없는 화질 손실을 유지하면 안 된다")
    }

    /// 수동 지정(--flow-base)이면 자동 조절이 멈춘다
    @Test func manualOverrideStopsAutoAdjustment() {
        let (s, tick) = makeScaler()
        s.manualOverride = true
        let fixed = s.current
        for _ in 0..<8 {
            tick()
            s.update(achievedRatio: 0.50, engineMs: 12.0, budgetMs: 16.7)
        }
        #expect(s.current == fixed, "수동 지정 시 자동 조절이 개입하면 안 된다")
    }

    /// 사다리 밖으로 나가지 않는다 (경계 안전)
    @Test func staysWithinLadderBounds() {
        let (s, tick) = makeScaler()
        for _ in 0..<40 {
            tick()
            s.update(achievedRatio: 0.30, engineMs: 15.0, budgetMs: 16.7)   // 계속 과부하
        }
        #expect(s.current >= AutoFlowScaler.rungs.first!, "사다리 최저 아래로 내려가면 안 된다")
        #expect(s.current <= AutoFlowScaler.rungs.last!, "사다리 최고 위로 올라가면 안 된다")
    }
}
