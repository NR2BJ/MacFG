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
    /// 사다리 천장이 800(idx 2)로 내려가 가용 칸이 480/640/800 셋뿐이다. 시나리오마다 필요한
    /// 여유가 달라 시작점을 명시적으로 고른다 — 상승 시험은 아래 칸에서, 원복 시험은 맨 위 칸에서.
    /// (heavy=8MP면 한 칸 아래에서 시작한다)
    private func makeScaler(heavy: Bool = true) -> (AutoFlowScaler, () -> Void) {
        let s = AutoFlowScaler()
        var t: CFAbsoluteTime = 1_000_000
        s.nowProvider = { t }
        s.seed(gpuCoreCount: 10, sourcePixels: heavy ? 8_000_000 : 2_000_000)
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
        let (s, tick) = makeScaler(heavy: false)   // 800에서 시작 — 두 칸 하강할 여유 필요
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

    /// **실측 회귀**: 틱이 주사율을 온전히 내고 있으면 폐기가 좀 있어도 하강하지 않는다.
    /// 4K/M4에서 "프레임은 120으로 멀쩡한데 flow만 1200→480으로 흘러내린" 사건의 원인.
    /// 폐기(staleDrop)는 배달 실패가 아니라 과잉 생산의 낭비라, 화질을 팔아 고칠 문제가 아니다.
    @Test func fullTickRateWithSomeWasteDoesNotDescend() {
        let (s, tick) = makeScaler()
        let start = s.current
        for _ in 0..<20 {
            tick()
            // 틱은 120/120 = 1.0인데 생산분의 12%가 기한 초과로 폐기되는 상황
            let achieved = AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 0.88)
            s.update(achievedRatio: achieved, engineMs: 6.0, budgetMs: 16.7)  // share 36% = 근거 통과
        }
        #expect(s.current >= start, "틱이 100%면 폐기가 있어도 화질을 내리면 안 된다")
    }

    /// 합성 규칙 자체의 경계 — 심각한 폐기는 여전히 미달로 인정해야 한다(안전장치를 죽이지 않았는지).
    @Test func severeWasteStillCountsAsMissing() {
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 0.88) == 1.0,
                "가벼운 폐기는 무시")
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 0.50) == 0.50,
                "생산의 절반을 버리면 미달로 인정")
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 0.60, keepRatio: 1.0) == 0.60,
                "틱 자체가 모자라면 그건 진짜 배달 실패")
    }

    /// **소스 프레임 손실은 달성도에 반드시 반영돼야 한다.**
    /// 풀 고갈로 소스를 통째로 버려도 타임라인에 남은 프레임 덕에 틱은 주사율을 그대로 낸다.
    /// keepRatio도 무사하다(만들지 못한 프레임은 안 세므로). 그래서 이 신호가 없으면 제어기가
    /// 붕괴를 "여유"로 읽고 화질을 올리려 든다 — 실측된 실패 모드다.
    @Test func sourceFrameLossCountsAsMissing() {
        // 붕괴 실측 재현: 틱 100%, 폐기율 정상, 그런데 소스의 40%가 파괴됨
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 1.0, deliveryRatio: 0.60) == 0.60,
                "소스 40% 손실이 달성도에 안 잡히면 스케일러가 붕괴 중에 화질을 올린다")
        // 리사이즈 순간의 한 창짜리 튐으로 하강이 걸리면 안 된다
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 1.0, deliveryRatio: 0.97) == 1.0,
                "소소한 손실은 무시")
        // 기존 규칙은 그대로 — 인자를 안 주면 예전 동작
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 0.88) == 1.0)
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 1.0, keepRatio: 0.50) == 0.50)
        // 셋 중 가장 나쁜 것이 이긴다
        #expect(AutoFlowScaler.combinedAchieved(tickRatio: 0.70, keepRatio: 1.0, deliveryRatio: 0.50) == 0.50)
    }

    /// **원복은 절대 아래로 가면 안 된다.** 천장이 현재 칸보다 낮은 상태(Pro/Max 시딩은 천장
    /// 위에서 출발한다)에서 옛 식 min(idx+2, ceilingIdx)는 현재보다 **아래**를 가리켜,
    /// "원복"이라 로그하면서 실제로는 화질을 더 깎았다.
    @Test func restoreNeverMovesDownward() {
        // 천장이 현재보다 낮은 경우 — 옛 식은 5를 돌려줘 7→5로 두 칸 강등했다
        #expect(AutoFlowScaler.restoreIndex(from: 7, ceilingIdx: 5) == 7,
                "천장이 현재보다 낮으면 원복은 제자리여야 한다 (내려가면 안 됨)")
        #expect(AutoFlowScaler.restoreIndex(from: 6, ceilingIdx: 5) == 6, "한 칸 차이도 마찬가지")
        // 정상 경우 — 천장 안에서 두 칸 되돌린다
        #expect(AutoFlowScaler.restoreIndex(from: 2, ceilingIdx: 5) == 4, "여유가 있으면 두 칸 원복")
        #expect(AutoFlowScaler.restoreIndex(from: 4, ceilingIdx: 5) == 5, "천장을 넘지는 않는다")
    }

    /// **천장은 회복 가능해야 한다.** 천장 학습은 내려가기만 해서, 일시적 과부하 한 번이
    /// 세션 전체의 화질 상한을 영구히 깎았다 — 상승 조건이 `idx < ceilingIdx`라 천장이 눌리면
    /// 부하가 완전히 걷혀도 영영 못 오른다.
    @Test func ceilingRecoversAfterSustainedGoodWindows() {
        let s = AutoFlowScaler()
        var t: CFAbsoluteTime = 1_000_000
        s.nowProvider = { t }
        s.seed(gpuCoreCount: 10, sourcePixels: 8_000_000)   // idx 1 (640), 천장 2 (800)

        // 1단계 — 올린 직후 실패시켜 천장을 학습(깎이게) 한다.
        //   상승 게이트 8s / 하강 게이트 3s를 통과하도록 4초씩 진행하고,
        //   상승 후 20초 안에 실패해야 천장 학습이 걸린다.
        for _ in 0..<6 { t += 4; s.update(achievedRatio: 1.0, engineMs: 5.0, budgetMs: 16.7) }
        let ceilingBefore = s.learnedCeiling
        for _ in 0..<3 { t += 4; s.update(achievedRatio: 0.60, engineMs: 9.0, budgetMs: 16.7) }
        #expect(s.learnedCeiling < ceilingBefore, "올린 직후 실패하면 천장이 깎여야 한다 (전제)")
        let pressed = s.learnedCeiling

        // 2단계 — 부하가 완전히 걷힌 채로 오래 안정적이면 천장이 되돌아와야 한다.
        for _ in 0..<40 { t += 4; s.update(achievedRatio: 1.0, engineMs: 3.0, budgetMs: 16.7) }
        #expect(s.learnedCeiling > pressed,
                "부하가 걷혔는데도 천장이 눌린 채면 세션 내내 화질 상한이 갇힌다")
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
