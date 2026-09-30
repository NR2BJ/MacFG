@preconcurrency import Metal
import Monitoring
import os

/// LSFG(Lossless Scaling) 방식의 순수 GPU 보간 엔진 — NN 없음.
///
/// 핵심 통찰: flow 해상도 ≠ 출력 선명도. flow는 저해상도 피라미드에서 추정하고,
/// 워프는 풀해상도 원본 픽셀을 이동시키므로 출력은 원본 선명도를 유지한다.
/// (RIFE류는 flow 해상도에서 이미지를 "합성"하므로 해상도가 화질을 결정 — 그래서 무거움)
///
/// 파이프라인 (전부 Metal 컴퓨트, 단일 CB, FidelityFX Optical Flow 계열):
///  1. 루마 피라미드: base(flowBase 긴변) → 반씩 다운 ~7레벨, zero-mean 프리필터(Z=L-mean5x5)
///  2. coarse→fine 매칭: 최상위 ±3, 이하 ±1 정수 국소 탐색 (6x6 SAD, gather 9회/후보,
///     half4 벡터 누적), 양방향(A→B, B→A) + 레벨마다 3x3 flow 스무딩
///  3. finalize: 방향별 forward-backward 일관성 → confF/confB (오클루전 판정),
///     |A-B| → 정적 마스크, 루마 히스토그램(장면컷 판정, AppleFI와 동일 방식)
///  4. 풀해상도 워프: w0=A(p-t·F_ab), w1=B(p-(1-t)·F_ba) → 방향별 가중 합성
///     (한쪽만 일관한 가림/드러남 영역은 보이는 쪽 단방향 워프 — 경계 스텝/고스트 억제).
///     정적 영역=B 원본(선명도 유지), 양쪽 다 저신뢰=가까운 원본 폴백.
///
/// 임의 t 지원: flow 벡터 t배 스케일이므로 어떤 위상이든 동일 비용 (144Hz 대응 기반).
public final class MetalFlowEngine: PairInterpolationEngine {
    public let name = "Metal Flow (GPU)"

    // 장면 전환 판정 (AppleFI와 동일)
    /// 0.25: 빠른 게임(오버워치 시점 회전/이펙트)이 히스토그램을 크게 흔들어 0.5에선 초당 수 회
    /// 오검출 → 보간 폐기 → 25~50ms 구멍(실측 cut=1~9/2s). 진짜 하드컷은 교집합이 ~0이라 안전.
    private let sceneCutIntersectionThreshold = 0.25

    /// flow 밀도 (긴 변 목표).
    ///
    /// **"시작 시 1회 설정 후 읽기 전용 — 락 불필요"는 사실이 아니다** (예전 주석을 정정).
    /// 실사용에선 AutoFlowScaler와 거버너가 **캡처 중에** 이 값을 바꾼다 — MainActor 쓰기
    /// (AppState의 applyGovernorDials)와 렌더 스레드 읽기(아래 워프 인코딩)가 교차한다.
    /// Double 스칼라라 찢어진 값이 나오지는 않고 크래시 부류도 아니지만, 한 쌍을 인코딩하는
    /// 도중에 값이 바뀔 수 있다는 뜻이다. 그 전제로 읽어라.
    /// 4K 쌍당 실측 (M4): gather 최적화 전 8.3ms → 후 5.6ms (base 960). 1080p 3.7ms.
    /// 960 = 60fps 예산(16.7ms) 내 최대 밀도 — 기본값. 1280은 30fps 이하 콘텐츠용 여지.
    // 기본 1440 (구 960) — MetalFlow(고전 광학흐름)는 4K에서도 flow를 1440p로 6.2ms에 돌린다
    // (RIFE 신경 predict 288p=12ms의 절반, 예산 대폭 여유). 소스보다 크면 min(1.0, base/long)로 캡.
    //
    // **"flow 해상도가 높을수록 저더가 준다"는 반박됐다** (df7e6bf 실측): 같은 클립에서
    // 480 = 23.33dB vs 1440 = 23.20dB 로 오히려 낮은 쪽이 미세하게 좋았다. 그래서 같은 커밋이
    // 사다리 천장을 800으로 내렸다. 이 기본값 1440은 그 사실을 알고도 남겨둔 것이다 —
    // 실사용에선 AutoFlowScaler 천장(800)이 먼저 걸리고, InterpBench는 --flow-base 미지정 시
    // 헤더에 flow base를 안 찍어서 기본값을 바꾸면 과거 출력과 구분이 불가능해진다.
    /// MACFG_MFFLOWBASE로 오프라인 A/B 가능 — 다른 MF 노브는 전부 env 게이트가 있는데 이것만
    /// 빠져 있어 벤치에서 flow 해상도를 고정 비교할 수 없었다(실측 시도가 전부 같은 값으로 돌았다).
    /// 실사용에선 AutoFlowScaler/거버너가 이 값을 덮어쓴다.
    public nonisolated(unsafe) static var flowBaseLongSide: Double = {
        if let s = Knob.string("MACFG_MFFLOWBASE"), let v = Double(s), v >= 240 { return v }
        return 1440
    }()

    /// 오클루전 방향별 워프 (실험, --occ-directional): 가림/드러남 영역에서 보이는 쪽 단방향 워프.
    /// 반복 패턴 aliasing 리스크로 합성 벤치 -3dB — 실영상 육안 A/B 전 기본 off.
    public nonisolated(unsafe) static var occlusionDirectional: Bool = false

    /// 서브픽셀 정련을 적용할 최종 레벨 수 (기본 1 = 최종 레벨만). 늘리면 상위 레벨의 정수 반올림
    /// 오차를 더 이른 단계에서 줄여 하위 전파가 정확해질 수 있다(비용: 레벨당 4 gather).
    public nonisolated(unsafe) static var refineLevels: Int = {
        if let s = Knob.string("MACFG_MFREFINE"), let v = Int(s), (1...7).contains(v) { return v }
        return 1
    }()

    /// 하위(정련) 레벨 탐색 반경 (기본 1). prior가 빗나갔을 때 각 레벨이 되잡을 수 있는 폭.
    public nonisolated(unsafe) static var fineSearchRadius: Int32 = {
        if let s = Knob.string("MACFG_MFFINE"), let v = Int32(s), (1...3).contains(v) { return v }
        return 1
    }()

    /// 평활 페널티 계수 (기본 0.017). 크면 prior에서 안 움직이려 하고(안정), 작으면 잘 따라감(디테일).
    public nonisolated(unsafe) static var matchPenalty: Float = {
        if let s = Knob.string("MACFG_MFPENALTY"), let v = Float(s), v >= 0, v <= 0.2 { return v }
        return 0.017
    }()

    /// 순환 일관성 신뢰도 문턱 [px] — conf = 1 - smoothstep(lo, hi, cycleError).
    /// 이 게이트가 닫히면(conf≈0) 워프 대신 폴백으로 빠져 **flow 계산이 결과에 반영되지 않는다**.
    /// 빠른 콘텐츠에서 순환 오차가 hi를 넘으면 화면 대부분이 폴백이 되므로 실측으로 잡아야 한다.
    public nonisolated(unsafe) static var confLo: Float = {
        if let s = Knob.string("MACFG_MFCYCLO"), let v = Float(s), v >= 0 { return v }
        return 2.5
    }()
    public nonisolated(unsafe) static var confHi: Float = {
        if let s = Knob.string("MACFG_MFCYCHI"), let v = Float(s), v > 0 { return v }
        return 8.0
    }()

    /// 순환 오차 문턱의 **모션 비례분** (0 = 절대 px 문턱만). 문턱 = lo + rel·|flow|.
    /// 같은 3px 순환 오차라도 50px 모션에선 정상, 2px 모션에선 쓰레기다 — 절대 문턱 하나로는
    /// 빠른 콘텐츠(과도하게 폐기)와 느린 콘텐츠(엉터리 flow 통과)를 동시에 맞출 수 없다.
    public nonisolated(unsafe) static var confRel: Float = {
        if let s = Knob.string("MACFG_MFCYCREL"), let v = Float(s), v >= 0 { return v }
        return 0.3   // 실측 최적 (전 7세트: avg +0.141 / 빠른셋 +0.197 / 최악값 평균 +0.051)
    }()

    /// 합성 신뢰도 결합 (0 = confF만 — 기존, 1 = max(confF,confB)). 한쪽 방향만 일관해도
    /// 그 워프는 쓸 만한데 기존엔 순방향 신뢰도만 봐서 통째로 폴백(=blend)으로 버렸다.
    /// dirBlend(방향별 tBlend)와 분리한 축 — 그쪽은 반복패턴 aliasing 리스크가 있어 따로 다룬다.
    public nonisolated(unsafe) static var confMax: Float = {
        if let s = Knob.string("MACFG_MFCONFMAX"), let v = Float(s), v >= 0, v <= 1 { return v }
        return 0.5   // 실측 최적 (전 7세트 avg +0.121 / 빠른셋 +0.150 / 최악값 +0.016).
                     // 1.0(완전 max)은 이득이 더 작고 최악값을 깎는다 — 한 방향만 확신할 때
                     // 그걸 100% 신뢰하면 틀릴 때 크게 틀리기 때문. 절반 결합이 안전점.
    }()

    /// 신뢰도 곡선 감마 (1 = 선형). <1이면 중간 신뢰도에서 워프 비중↑(폴백=blend 의존↓).
    public nonisolated(unsafe) static var confGamma: Float = {
        if let s = Knob.string("MACFG_MFCONFGAMMA"), let v = Float(s), v > 0 { return v }
        return 1.0
    }()

    /// 광도 검증 문턱 (기본 0.04~0.14) — flow를 따라간 곳의 밝기 차로 그 방향을 기각하는 2차 방어선.
    /// 압축 노이즈가 큰 스트리밍에선 정상 워프까지 기각할 수 있어 실측 대상.
    public nonisolated(unsafe) static var photoLo: Float = {
        if let s = Knob.string("MACFG_MFPHOTOLO"), let v = Float(s), v >= 0 { return v }
        return 0.04
    }()
    public nonisolated(unsafe) static var photoHi: Float = {
        if let s = Knob.string("MACFG_MFPHOTOHI"), let v = Float(s), v > 0 { return v }
        return 0.14
    }()
    /// 정적 판정 문턱 — 조이면 정적 판정이 **줄어**(움직이는 걸 덜 고정), 느슨하면 늘어난다.
    ///
    /// **0.008/0.04로 되돌렸다(2026-07-25 실측).** 한때 0.004/0.02로 조였고 삼중항 PSNR은
    /// 실제로 올랐지만, 그 측정은 **텍스트 흔들림을 잴 수 없었다** — PSNR은 t=0.5에서 정답과의
    /// *공간* 정확도인데 흔들림은 *시간* 현상이다. staticDev 지표(정지 픽셀이 원본에서 벗어난
    /// 정도, InterpBench --quality-ab)를 만들어 4K 실프레임 3세트로 재보니:
    ///
    ///        축          PSNR 이득    staticDev 비용   효율
    ///        conf 계열   +0.237 dB    -0.150 dB       1.58
    ///        static 조임 +0.070 dB    -0.442 dB       0.16   ← 10배 나쁜 거래
    ///
    /// 조이면 압축 노이즈로 차이가 미세하게 뜨는 정적 UI 텍스트까지 "움직이는 픽셀"로 분류돼
    /// 워프를 먹는다. 세 세트 모두 같은 순서였고 사용자의 지각 판정과도 일치했다.
    /// (흔들림은 **주변 움직임이 정지 요소의 flow를 오염시킬 때** 생긴다 — 정지 영역이 움직임과
    ///  떨어져 있으면 그곳 flow는 0이라 워프해도 원본과 같다.)
    public nonisolated(unsafe) static var staticLo: Float = {
        if let s = Knob.string("MACFG_MFSTATLO"), let v = Float(s), v >= 0 { return v }
        return 0.008
    }()
    public nonisolated(unsafe) static var staticHi: Float = {
        if let s = Knob.string("MACFG_MFSTATHI"), let v = Float(s), v > 0 { return v }
        return 0.04
    }()

    /// 코스 레벨 탐색 반경 (기본 3). 큰 변위(빠른 시점 회전) 추적 한계를 정한다.
    /// **코스 레벨 영점 편향** (기본 penalty×2 = 0.034, `MACFG_MFZEROBIAS`, 0=끔).
    /// 코스 레벨은 이전 쌍의 flow를 시간 prior로 쓰고 평활 페널티가 prior 기준이라, 데이터가 방향을
    /// 못 정하는 영역(aperture: 가로 줄무늬 계단, 평탄한 어둠)에서는 **한 번 생긴 큰 벡터가 영구히
    /// 고착**됐다 — 빠른 팬이 끝나도 그 영역만 옛 벡터를 유지해 워프가 수백 px 밖(채팅 패널)을
    /// 샘플했다(2026-09-25 덤프 191839 유령 채팅). 후보 비용에 `zeroBias·(|x|+|y|)`(L1)를 더해
    /// 애매 영역이 0으로 돌아오게 하고, |prior|>반경이면 0 주변 창도 함께 탐색한다.
    /// 텍스처 영역의 실제 모션은 SAD 차이가 편향보다 커서 영향이 없다(H.264류 zero-MV 보너스와 동일 발상).
    public nonisolated(unsafe) static var zeroBias: Float = {
        if let s = Knob.string("MACFG_MFZEROBIAS"), let v = Float(s), v >= 0, v <= 1 { return v }
        return 0.034
    }()

    public nonisolated(unsafe) static var coarseSearchRadius: Int32 = {
        if let s = Knob.string("MACFG_MFRADIUS"), let v = Int32(s), (1...8).contains(v) { return v }
        return 3
    }()

    /// 모션 부드러움 0(예리)~1(부드러움), 0.5=현재 기본. 취향 슬라이더 — flow의 예리함↔매끄러움 축.
    /// 하단: flow raw(디테일↑, shimmer 가능). 상단: flow 박스+워프블러(에러 완만, AppleFI 느낌).
    public nonisolated(unsafe) static var motionSmoothness: Float = 0.5

    /// 경계 전환 0(crisp=저더)~1(soft=고스팅), 0.5=기본. 물체 경계(저신뢰) 폴백 크로스페이드 폭.
    /// 축이 부드러움과 독립(부드러움=flow, 이건=경계 처리)이라 별도 노브. 콘텐츠 의존:
    /// crisp=게임/스포츠(빠른 급전환 — 이중상이 튐, 저더는 순간이라 덜 보임),
    /// soft=영화/다큐(느린 팬 — 저더가 톡톡 끊겨 거슬림, 고스팅은 옅게 묻힘).
    public nonisolated(unsafe) static var boundarySoftness: Float = 0.5

    private var device: (any MTLDevice)?
    private let logger = Logger(subsystem: "com.macfg", category: "MetalFlow")

    // PSO
    private var downLumaPSO: (any MTLComputePipelineState)?
    private var downHalfPSO: (any MTLComputePipelineState)?
    private var zeroMeanPSO: (any MTLComputePipelineState)?
    private var matchPSO: (any MTLComputePipelineState)?
    private var smoothPSO: (any MTLComputePipelineState)?
    private var finalizePSO: (any MTLComputePipelineState)?
    private var warpPSO: (any MTLComputePipelineState)?

    // 피라미드 리소스 (소스 크기 + flow base 의존)
    private var srcWidth = 0
    private var srcHeight = 0
    /// 현재 피라미드를 만들 때 쓴 flowBaseLongSide — 이 값이 바뀌면 재구축(자동 스케일/거버너 캡 반영)
    private var builtFlowBase: Double = 0
    private var levels: [(w: Int, h: Int)] = []
    private var lumaA: [any MTLTexture] = []
    private var lumaB: [any MTLTexture] = []
    // zero-mean 루마 (Z = L - mean5x5) — 매칭 전용. 후보별 창 평균 재계산을 프리패스로 대체
    private var zmA: [any MTLTexture] = []
    private var zmB: [any MTLTexture] = []
    private var flowF: [any MTLTexture] = []   // A→B
    private var flowB: [any MTLTexture] = []   // B→A
    private var flowTmp: [any MTLTexture] = [] // 스무딩 핑퐁
    private var maskTex: (any MTLTexture)?     // base res: r=신뢰도, g=정적
    // 시간적 flow 전파: 이전 쌍의 코스 flow를 다음 쌍 시작점으로 (벡터 노이즈/shimmer 억제)
    private var prevCoarseF: (any MTLTexture)?
    private var prevCoarseB: (any MTLTexture)?
    private var hasTemporalPrior = false
    /// 직전 쌍의 tsB — 이번 tsA와 같으면 A쪽 피라미드를 재사용할 수 있다 (O1-2 핑퐁)
    private var lastPairTsB: CFTimeInterval = 0
    /// A쪽 재사용 허용 (MACFG_NOREUSEA=1로 끄고 A/B 비교)
    private let reuseA = Knob.string("MACFG_NOREUSEA") != "1"
    // 출력 링: 갭 채움으로 쌍당 최대 4장 → 넉넉히 12 (타임라인 cap 12와 정합)
    private var outputPool: [any MTLTexture] = []
    private var outputIndex = 0
    /// outputPool 슬롯별 현 점유 세대 (표시 직전 isFrameLive 검증용). encode/조회 모두 렌더 스레드.
    private var slotStamps: [UInt64] = []
    private var nextStamp: UInt64 = 0
    // 히스토그램은 쌍당 1개 — 별도 링
    private var statsBuffers: [any MTLBuffer] = []
    private var statsIndex = 0

    private var pairCount = 0

    public init() {}

    // MARK: - Prepare

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ n: String) async throws -> any MTLComputePipelineState {
            guard let f = library.makeFunction(name: n) else { throw InterpolationError.notPrepared }
            return try await device.makeComputePipelineState(function: f)
        }
        downLumaPSO = try await pso("mfDownLuma")
        downHalfPSO = try await pso("mfDownHalf")
        zeroMeanPSO = try await pso("mfZeroMean")
        matchPSO = try await pso("mfMatch")
        smoothPSO = try await pso("mfSmooth")
        finalizePSO = try await pso("mfFinalize")
        warpPSO = try await pso("mfWarp")
        DiagnosticLog.shared.log("[MetalFlow] prepared (pyramid flow + full-res warp) zeroBias=\(Self.zeroBias) occDir=\(Self.occlusionDirectional) smooth=\(Self.motionSmoothness)")
    }

    // MARK: - Encode

    public func encodePair(
        stableA: any MTLTexture,
        stableB: any MTLTexture,
        tsA: CFTimeInterval,
        tsB: CFTimeInterval,
        tValues: [Float],
        into commandBuffer: any MTLCommandBuffer
    ) -> PairEncodeResult? {
        guard let downLumaPSO, let downHalfPSO, let zeroMeanPSO, let matchPSO, let smoothPSO,
              let finalizePSO, let warpPSO,
              tsB > tsA, !tValues.isEmpty else { return nil }

        ensureResources(width: stableB.width, height: stableB.height)
        // 2×t 불변식: 연속 2쌍의 출력이 타임라인 지연버퍼에 동시 생존해도 링이 안 겹치게
        // 쌍당 t 수를 풀의 절반으로 제한 (풀 16이면 8 = 호출측 상한과 일치 → 거동 불변)
        // outputPool도 검사 — 극한 메모리 압박에서 대형 BGRA 풀만 전부 실패하면
        // 아래 outputPool[outputIndex]/% count가 빈 배열 크래시가 된다(리뷰 확정).
        guard !levels.isEmpty, let maskTex, !statsBuffers.isEmpty, !outputPool.isEmpty else { return nil }
        // 풀이 예산 축소로 작아진 대형 소스(5K 이상)에서는 t 수가 상한을 넘을 수 있다. 예전엔
        // 쌍을 통째로 버려(nil) 저fps 콘텐츠의 보간이 영구 0장이 됐다 (리뷰 확정). 버리는 대신
        // 균등 간격으로 솎아 남은 예산만큼은 보간한다 — 홀 대신 우아한 강등.
        var tValues = tValues
        let maxT = max(1, outputPool.count / 2)
        if tValues.count > maxT {
            let step = Double(tValues.count) / Double(maxT)
            tValues = (0..<maxT).map { tValues[min(Int(Double($0) * step), tValues.count - 1)] }
        }

        let statsBuffer = statsBuffers[statsIndex]
        statsIndex = (statsIndex + 1) % statsBuffers.count
        let L = levels.count

        // 모션 부드러움 매핑 — flow 부드러움만 조절 (고스팅 주는 폴백 폭은 슬라이더에서 분리).
        // 하단: flow raw(예리, 디테일↑). 상단: flow 박스+워프블러(에러 완만, AppleFI 느낌).
        let sm = min(max(Self.motionSmoothness, 0), 1)
        var smoothAmt: Float = min(sm * 2, 1)                       // flow 박스스무딩: 0→raw, 0.5→full, 1→full
        // **상반부(0.5~1.0)는 측정상 무효다 (2026-08-31 실측). 죽은 코드는 아니다.**
        // flowBlur는 셰이더까지 제대로 연결돼 flow 필드를 실제로 블러한다(아래 mfWarp 참조).
        // 그런데 smoothAmt가 이미 0.5에서 포화(min(sm*2,1))라 flow가 완전 평활 상태이고,
        // 거기 블러를 더해도 결과가 안 바뀐다:
        //   sm 0.0 → 0.5 : med PSNR +3.36dB (하반부는 크게 작동)
        //   sm 0.5 → 1.0 : **+0.01dB**, sharp는 소수점 3자리까지 동일 (2시퀀스 확인)
        // 매핑을 0~1로 펴면 해상도는 좋아지지만 저장된 값의 의미가 바뀌어 마이그레이션이 필요하고
        // 얻는 것이 슬라이더 눈금뿐이라 **의도적으로 두었다.** 다시 재보지 말 것.
        let flowBlur: Float = max((sm - 0.5) * 2, 0)               // 워프 flow 블러: 0(≤0.5)→1(=1.0)
        // 경계 폴백 폭 = boundarySoftness 슬라이더 (별도 축). crisp(0)=±0.03(거의 단일프레임=저더),
        // 0.5=±0.16(= v1.0.4 기본 동작), soft(1)=±0.29(넓은 블렌드=부드럽지만 고스팅). 콘텐츠 취향.
        let bs = min(max(Self.boundarySoftness, 0), 1)
        let fadeHalf: Float = 0.03 + 0.26 * bs
        let fadeLo = 0.5 - fadeHalf
        let fadeHi = 0.5 + fadeHalf

        // 0) 히스토그램 클리어
        if let fill = commandBuffer.makeBlitCommandEncoder() {
            fill.fill(buffer: statsBuffer, range: 0..<(64 * 4), value: 0)
            fill.endEncoding()
        }

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return nil }

        // O1-2: 연속 쌍이면 A쪽 피라미드는 직전 쌍의 B쪽 결과와 같다 (stableA == 직전 stableB).
        // 매 쌍 A를 재계산하던 것을 핑퐁 스왑으로 건너뛴다 — 피라미드/zero-mean 스테이지의
        // 절반 + 4K BGRA 읽기 1회분 절감. 불연속(리셋/드랍/크기변경)이면 정상 계산.
        let continuous = reuseA && lastPairTsB > 0 && abs(tsA - lastPairTsB) < 1e-6
        if continuous {
            swap(&lumaA, &lumaB)   // 직전 B피라미드가 이번 A — 내용 그대로 재사용
            swap(&zmA, &zmB)
        }

        // 1) 루마 피라미드
        enc.setComputePipelineState(downLumaPSO)
        if !continuous {
            enc.setTexture(stableA, index: 0); enc.setTexture(lumaA[0], index: 1)
            dispatch(enc, levels[0].w, levels[0].h, downLumaPSO)
        }
        enc.setTexture(stableB, index: 0); enc.setTexture(lumaB[0], index: 1)
        dispatch(enc, levels[0].w, levels[0].h, downLumaPSO)
        enc.setComputePipelineState(downHalfPSO)
        for l in 1..<L {
            if !continuous {
                enc.setTexture(lumaA[l - 1], index: 0); enc.setTexture(lumaA[l], index: 1)
                dispatch(enc, levels[l].w, levels[l].h, downHalfPSO)
            }
            enc.setTexture(lumaB[l - 1], index: 0); enc.setTexture(lumaB[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, downHalfPSO)
        }

        // 1.5) zero-mean 프리필터: Z = L - mean5x5(L). 매치가 후보(≤49)마다 25탭 창 평균을
        // 재계산하던 것을 프리패스 1회로 대체 — 매치 커널 ALU/레지스터 대폭 절감 (4K 실측 8.3→벤치 참조)
        enc.setComputePipelineState(zeroMeanPSO)
        for l in 0..<L {
            if !continuous {
                enc.setTexture(lumaA[l], index: 0); enc.setTexture(zmA[l], index: 1)
                dispatch(enc, levels[l].w, levels[l].h, zeroMeanPSO)
            }
            enc.setTexture(lumaB[l], index: 0); enc.setTexture(zmB[l], index: 1)
            dispatch(enc, levels[l].w, levels[l].h, zeroMeanPSO)
        }
        lastPairTsB = tsB

        // 2) coarse→fine 매칭 (양방향) + 스무딩.
        // 코스 레벨은 이전 쌍의 flow를 prior로 사용 (시간적 전파 — 프레임 간 벡터 일관성).
        // 주의: 시간적 prior는 스케일 1(동일 레벨)이므로 priorScale=1, 공간 prior는 2.
        for l in stride(from: L - 1, through: 0, by: -1) {
            let isCoarsest = (l == L - 1)
            let useTemporal = isCoarsest && hasTemporalPrior
            var params = MatchParams(
                // 최상위(코스) 탐색 반경 — 여기서 잡는 최대 변위가 곧 "빠른 시점 회전을 따라갈 수 있는
                // 한계"다. 코스 레벨은 가장 작아서(4K/flow1440 기준 22×12px) 반경을 넓혀도 비용이
                // 거의 안 는다(후보 수는 (2r+1)²이지만 픽셀 수가 1/4096). MACFG_MFRADIUS로 실측 스윕.
                searchRadius: isCoarsest ? Self.coarseSearchRadius : Self.fineSearchRadius,
                hasPrior: (useTemporal || !isCoarsest) ? 1 : 0,
                refine: l < Self.refineLevels ? 1 : 0,   // 서브픽셀 정련 레벨 수 (기본 1=최종만)
                priorScale: isCoarsest ? 1.0 : 2.0,
                penalty: Self.matchPenalty,
                zeroBias: isCoarsest ? Self.zeroBias : 0
            )
            // forward: A→B (zero-mean 루마로 매칭)
            enc.setComputePipelineState(matchPSO)
            enc.setTexture(zmA[l], index: 0)
            enc.setTexture(zmB[l], index: 1)
            enc.setTexture(isCoarsest ? (prevCoarseF ?? flowF[l]) : flowF[l + 1], index: 2)
            enc.setTexture(flowTmp[l], index: 3)
            enc.setBytes(&params, length: MemoryLayout<MatchParams>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, matchPSO)
            enc.setComputePipelineState(smoothPSO)
            enc.setTexture(flowTmp[l], index: 0); enc.setTexture(flowF[l], index: 1)
            enc.setBytes(&smoothAmt, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, smoothPSO)
            // backward: B→A
            enc.setComputePipelineState(matchPSO)
            enc.setTexture(zmB[l], index: 0)
            enc.setTexture(zmA[l], index: 1)
            enc.setTexture(isCoarsest ? (prevCoarseB ?? flowB[l]) : flowB[l + 1], index: 2)
            enc.setTexture(flowTmp[l], index: 3)
            enc.setBytes(&params, length: MemoryLayout<MatchParams>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, matchPSO)
            enc.setComputePipelineState(smoothPSO)
            enc.setTexture(flowTmp[l], index: 0); enc.setTexture(flowB[l], index: 1)
            enc.setBytes(&smoothAmt, length: MemoryLayout<Float>.stride, index: 0)
            dispatch(enc, levels[l].w, levels[l].h, smoothPSO)
        }
        enc.endEncoding()

        // 다음 쌍을 위한 시간적 prior 백업 (코스 레벨 flow)
        if let pf = prevCoarseF, let pb = prevCoarseB,
           let backupBlit = commandBuffer.makeBlitCommandEncoder() {
            let cl = levels[L - 1]
            backupBlit.copy(from: flowF[L - 1], sourceSlice: 0, sourceLevel: 0,
                            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                            sourceSize: MTLSize(width: cl.w, height: cl.h, depth: 1),
                            to: pf, destinationSlice: 0, destinationLevel: 0,
                            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            backupBlit.copy(from: flowB[L - 1], sourceSlice: 0, sourceLevel: 0,
                            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                            sourceSize: MTLSize(width: cl.w, height: cl.h, depth: 1),
                            to: pb, destinationSlice: 0, destinationLevel: 0,
                            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            backupBlit.endEncoding()
            hasTemporalPrior = true
        }

        guard let enc2 = commandBuffer.makeComputeCommandEncoder() else { return nil }

        // 3) finalize: 신뢰도/정적 마스크 + 히스토그램
        enc2.setComputePipelineState(finalizePSO)
        enc2.setTexture(lumaA[0], index: 0)
        enc2.setTexture(lumaB[0], index: 1)
        enc2.setTexture(flowF[0], index: 2)
        enc2.setTexture(flowB[0], index: 3)
        enc2.setTexture(maskTex, index: 4)
        enc2.setBuffer(statsBuffer, offset: 0, index: 0)
        var fp = FinalizeParams(confLo: Self.confLo, confHi: Self.confHi, confRel: Self.confRel,
                                photoLo: Self.photoLo, photoHi: Self.photoHi, statLo: Self.staticLo, statHi: Self.staticHi)
        enc2.setBytes(&fp, length: MemoryLayout<FinalizeParams>.stride, index: 1)
        dispatch(enc2, levels[0].w, levels[0].h, finalizePSO)

        // 4) 풀해상도 워프 + 합성 — flow는 한 번 계산, t별로 워프만 반복 (장당 ~1ms)
        // 이게 갭 채움(드랍된 소스 프레임 자리 메꾸기)을 싸게 만드는 핵심.
        var frames: [(t: Float, texture: any MTLTexture, stamp: UInt64)] = []
        enc2.setComputePipelineState(warpPSO)
        enc2.setTexture(stableA, index: 0)
        enc2.setTexture(stableB, index: 1)
        enc2.setTexture(flowF[0], index: 2)
        enc2.setTexture(flowB[0], index: 3)
        enc2.setTexture(maskTex, index: 4)
        let dirBlend: Float = Self.occlusionDirectional ? 1.0 : 0.0
        for t in tValues.sorted() {
            let slotIdx = outputIndex
            let output = outputPool[outputIndex]
            outputIndex = (outputIndex + 1) % outputPool.count
            nextStamp &+= 1
            let stamp = nextStamp
            if slotIdx < slotStamps.count { slotStamps[slotIdx] = stamp }   // 이 슬롯의 현 점유 세대
            var wp = WarpParams(t: t, dirBlend: dirBlend, fadeLo: fadeLo, fadeHi: fadeHi, flowBlur: flowBlur)
            enc2.setTexture(output, index: 5)
            enc2.setBytes(&wp, length: MemoryLayout<WarpParams>.stride, index: 0)
            dispatch(enc2, output.width, output.height, warpPSO)
            frames.append((t: t, texture: output, stamp: stamp))
        }
        enc2.endEncoding()

        pairCount += 1
        if pairCount <= 3 || pairCount % 600 == 0 {
            DiagnosticLog.shared.log("[MetalFlow] pair #\(pairCount) (\(stableB.width)x\(stableB.height), base=\(levels[0].w)x\(levels[0].h), levels=\(L), tCount=\(tValues.count))")
        }

        let threshold = sceneCutIntersectionThreshold
        let evaluator: @Sendable () -> Bool = {
            let ptr = statsBuffer.contents().bindMemory(to: UInt32.self, capacity: 64)
            var hA = [Double](repeating: 0, count: 32)
            var hB = [Double](repeating: 0, count: 32)
            var totalA = 0.0, totalB = 0.0, sumA = 0.0, sumB = 0.0
            for i in 0..<32 {
                hA[i] = Double(ptr[i]); hB[i] = Double(ptr[32 + i])
                totalA += hA[i]; totalB += hB[i]
                sumA += hA[i] * Double(i); sumB += hB[i] * Double(i)
            }
            guard totalA > 1000, totalB > 0 else { return false }
            // 밝기 정렬 교집합 — 평균 bin 차이만큼 B를 시프트해 비교. 플래시/이펙트(균일 밝기
            // 변화)는 정렬돼 통과하고, 진짜 컷(구조 변화)만 낮게 남는다 (게임 오검출 차단).
            let shift = Int((sumB / totalB - sumA / totalA).rounded())
            var intersect = 0.0
            for i in 0..<32 {
                let j = i + shift
                if j >= 0, j < 32 { intersect += min(hA[i], hB[j]) }
            }
            return intersect / totalA < threshold
        }
        return PairEncodeResult(frames: frames, sceneCutEvaluator: evaluator)
    }

    private struct MatchParams {
        var searchRadius: Int32
        var hasPrior: Int32
        var refine: Int32
        var priorScale: Float
        var penalty: Float
        var zeroBias: Float   // 코스 레벨 영점 편향 (0=끔)
    }

    private struct FinalizeParams { var confLo: Float; var confHi: Float; var confRel: Float; var photoLo: Float; var photoHi: Float; var statLo: Float; var statHi: Float }
    private struct WarpParams {
        var t: Float
        var dirBlend: Float
        var fadeLo: Float
        var fadeHi: Float
        var flowBlur: Float
        var confMax: Float = MetalFlowEngine.confMax
        var confGamma: Float = MetalFlowEngine.confGamma
        /// 디버그 출력 (MACFG_MFDEBUG, 0=끔) — 모드 목록은 warpDebug 주석.
        var debug: Float = MetalFlowEngine.warpDebug
    }

    /// 워프 디버그 모드 (MACFG_MFDEBUG): 1=(0,0,|f|/100) 2=(confF,confB,static) 3=(tBlend,conf,0) 5=w0 6=w1 7=nearestPix 8=interp
    public nonisolated(unsafe) static var warpDebug: Float = Float(Knob.string("MACFG_MFDEBUG") ?? "") ?? 0

    private func dispatch(_ enc: any MTLComputeCommandEncoder, _ w: Int, _ h: Int, _ pso: any MTLComputePipelineState) {
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg
        )
    }

    // MARK: - Resources

    private func ensureResources(width: Int, height: Int) {
        guard let device else { return }
        // flowBaseLongSide 변경도 재구축 트리거 — 예전엔 소스 크기만 봐서, 세션 중 flow 해상도를
        // 바꿔도(거버너 캡·자동 스케일러) 피라미드가 옛 해상도 그대로였다 = 다이얼이 통째로 무효.
        // 자동 flow 스케일이 성립하려면 이 반영이 전제.
        let baseNow = Self.flowBaseLongSide
        guard width != srcWidth || height != srcHeight || levels.isEmpty
                || abs(baseNow - builtFlowBase) > 0.5 else { return }
        srcWidth = width
        srcHeight = height
        builtFlowBase = baseNow

        // flow base: 긴 변 기준 flow 밀도 (이미지가 아니라 모션 지도의 해상도 —
        // 워프는 항상 풀해상도 원본 픽셀이므로 출력 선명도와 무관, 모션 경계 정밀도만 좌우).
        // 픽셀당 5x5 SAD 매칭이라 base²가 비용 지배: 1900 실측 77ms / 480 실측 ~3ms.
        let longSide = max(width, height)
        let s = min(1.0, baseNow / Double(longSide))   // 재구축 트리거와 동일 스냅샷 사용
        var bw = Int(Double(width) * s), bh = Int(Double(height) * s)
        bw = max(bw & ~1, 64); bh = max(bh & ~1, 64)

        levels = []
        var w = bw, h = bh
        while w >= 15 && h >= 15 && levels.count < 7 {
            levels.append((w, h))
            w /= 2; h /= 2
        }

        func tex(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat, usage: MTLTextureUsage = [.shaderRead, .shaderWrite]) -> (any MTLTexture)? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }

        lumaA = []; lumaB = []; zmA = []; zmB = []; flowF = []; flowB = []; flowTmp = []
        for (w, h) in levels {
            guard let la = tex(w, h, .r16Float), let lb = tex(w, h, .r16Float),
                  let za = tex(w, h, .r16Float), let zb = tex(w, h, .r16Float),
                  let ff = tex(w, h, .rg16Float), let fb = tex(w, h, .rg16Float),
                  let ft = tex(w, h, .rg16Float) else { levels = []; return }
            lumaA.append(la); lumaB.append(lb)
            zmA.append(za); zmB.append(zb)
            flowF.append(ff); flowB.append(fb); flowTmp.append(ft)
        }
        // r=confF(A→B 일관성), g=정적, b=confB(B→A 일관성) — 방향별 오클루전 판정용
        maskTex = tex(levels[0].w, levels[0].h, .rgba8Unorm)
        let cl = levels[levels.count - 1]
        prevCoarseF = tex(cl.w, cl.h, .rg16Float)
        prevCoarseB = tex(cl.w, cl.h, .rg16Float)
        hasTemporalPrior = false
        lastPairTsB = 0

        outputPool = []; statsBuffers = []; outputIndex = 0; statsIndex = 0; slotStamps = []
        // 출력 풀 크기: 호출측이 쌍당 최대 8 t를 보내므로(갭 채움) 연속 2쌍=16장이 타임라인
        // 지연버퍼(60~100ms) 동안 동시 생존 가능 — 12장이면 표시 전 텍스처를 후속 워프가
        // 덮어씀 (RIFE 8→16과 동일 클래스, 감사 확정). 4K 16×33MB=531MB.
        // 16MP(5120x3140)는 메모리 압박 실측(770MB)이라 예산으로 축소하되, encodePair의
        // 2×t 가드가 풀의 절반까지만 쌍을 수용해 오버런을 구조적으로 차단한다.
        let bytesPerTexture = max(width * height * 4, 1)
        let poolCount = min(16, max(6, (600 << 20) / bytesPerTexture))
        for _ in 0..<poolCount {
            if let t = tex(width, height, .bgra8Unorm) {
                outputPool.append(t)
            }
        }
        // statsBuffers도 풀과 동수 — 4개면 완료 핸들러(sceneCutEvaluator)가 N쌍 히스토그램을
        // 읽기 전에 N+4쌍의 GPU fill이 덮어쓸 수 있다 (AppleFI는 풀과 동수로 방어 — 동일화)
        for _ in 0..<poolCount {
            if let b = device.makeBuffer(length: 64 * 4, options: .storageModeShared) {
                statsBuffers.append(b)
            }
        }
        slotStamps = [UInt64](repeating: 0, count: outputPool.count)   // 슬롯 세대 리셋
        DiagnosticLog.shared.log("[MetalFlow] resources: src=\(width)x\(height) base=\(bw)x\(bh) levels=\(levels.count)")
    }

    /// 표시 직전 검증 — stamp가 아직 어느 슬롯의 현 세대면 유효(안 덮임). 렌더 스레드 전용.
    public func isFrameLive(_ stamp: UInt64) -> Bool { slotStamps.contains(stamp) }

    public func reset() {
        hasTemporalPrior = false
        lastPairTsB = 0
    }

    public func shutdown() {
        levels = []
        lumaA = []; lumaB = []; zmA = []; zmB = []; flowF = []; flowB = []; flowTmp = []
        maskTex = nil
        outputPool = []; statsBuffers = []
    }

    // MARK: - Shaders

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct MatchParams { int searchRadius; int hasPrior; int refine; float priorScale; float penalty; float zeroBias; };
    struct FinalizeParams { float confLo; float confHi; float confRel; float photoLo; float photoHi; float statLo; float statHi; };
    struct WarpParams { float t; float dirBlend; float fadeLo; float fadeHi; float flowBlur; float confMax; float confGamma; float debug; };

    constant half3 kLuma = half3(0.2126h, 0.7152h, 0.0722h);

    // BGRA 소스 → base 루마 (박스 다운샘플)
    kernel void mfDownLuma(
        texture2d<half, access::sample> src [[texture(0)]],
        texture2d<half, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float2 o = 0.25 / float2(w, h);
        half l = dot(src.sample(s, uv + float2(-o.x,-o.y)).rgb, kLuma)
               + dot(src.sample(s, uv + float2( o.x,-o.y)).rgb, kLuma)
               + dot(src.sample(s, uv + float2(-o.x, o.y)).rgb, kLuma)
               + dot(src.sample(s, uv + float2( o.x, o.y)).rgb, kLuma);
        dst.write(half4(l * 0.25h), gid);
    }

    // 루마 반 다운
    kernel void mfDownHalf(
        texture2d<half, access::sample> src [[texture(0)]],
        texture2d<half, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        dst.write(half4(src.sample(s, uv).r), gid);
    }

    // Z = L - mean5x5(L): zero-mean 매칭 신호 사전 계산.
    // 매치 커널이 후보마다 25탭 창 평균을 구하던 것을 제거 (조명 불변성은 동일하게 유지 —
    // 창 중심이 후보 위치가 아닌 각 탭 자기 위치 기준이 되지만, 고역통과 신호 매칭이라 등가 이상).
    kernel void mfZeroMean(
        texture2d<half, access::read> src [[texture(0)]],
        texture2d<half, access::write> dst [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        int W = int(w) - 1, H = int(h) - 1;
        // 중심 탭(gid)은 한 번만 읽어 창 합과 최종 감산에 재사용 (기존 26탭 → 25탭).
        // 고정 바운드 루프는 컴파일 시 완전 언롤되어 dx=dy=0 가드는 런타임 분기 없이 제거됨.
        half center = src.read(gid).r;
        half acc = center;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                if (dx == 0 && dy == 0) continue;
                uint2 q = uint2(clamp(int(gid.x) + dx, 0, W), clamp(int(gid.y) + dy, 0, H));
                acc += src.read(q).r;
            }
        dst.write(half4(center - acc / 25.0h), gid);
    }

    // 6x6 SAD (gather 9회) — 탐색/refine 공용. gid_c = 후보의 절대 좌표(gid + 오프셋).
    static inline half sad6x6(texture2d<half, access::sample> ref, sampler s,
                              thread const half4* patch, float2 gid_c, float2 size) {
        half4 acc = half4(0.0h);
        int j = 0;
        for (int dy = -2; dy <= 2; dy += 2)
            for (int dx = -2; dx <= 2; dx += 2) {
                float2 q = (gid_c + float2(dx + 1, dy + 1)) / size;
                acc += abs(ref.gather(s, q) - patch[j++]);
            }
        return acc.x + acc.y + acc.z + acc.w;
    }
    // 피라미드 매칭: prior(코스 레벨) flow를 시작점으로 ±searchRadius 정수 국소 탐색 (6x6 SAD)
    // 입력은 zero-mean 루마(Z = L - mean5x5) — 조명/페이드 불변.
    //
    // gather 최적화: 6x6 창을 2x2 쿼드 9개(gather 9회)로 읽어 후보당 샘플 명령 25→9,
    // SAD는 half4 벡터로 누적 (실측: 샘플링 바운드 커널이라 ALU 최적화만으론 불변 — gather가 관건).
    // prior는 정수 반올림 — gather는 텍셀 정렬이 필요. 서브픽셀은 최종 레벨 refine이 복원
    // (정수 탐색 + 최종 서브픽셀 = 고전 블록매칭 표준. 워프용 flow는 스무딩이 소수화).
    kernel void mfMatch(
        texture2d<half, access::sample> src [[texture(0)]],   // 기준 프레임 zero-mean 루마
        texture2d<half, access::sample> ref [[texture(1)]],   // 대상 프레임 zero-mean 루마
        texture2d<float, access::sample> prior [[texture(2)]], // 코스 flow (없으면 미사용)
        texture2d<float, access::write> flowOut [[texture(3)]],
        constant MatchParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = flowOut.get_width(), h = flowOut.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 size = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / size;

        float2 base = float2(0.0);
        if (p.hasPrior != 0) {
            // 공간 prior(코스 레벨)는 2.0, 시간 prior(동일 레벨, 이전 쌍)는 1.0
            base = round(prior.sample(s, uv).rg * p.priorScale);
        }

        // 6x6 소스 패치를 gather 9회로 캐시 (쿼드 경계 uv: 텍셀 d..d+1 사이 = gid+d+1)
        half4 patch[9];
        int k = 0;
        for (int dy = -2; dy <= 2; dy += 2)
            for (int dx = -2; dx <= 2; dx += 2) {
                float2 q = (float2(gid) + float2(dx + 1, dy + 1)) / size;
                patch[k++] = src.gather(s, q);
            }
        half bestSAD = 65504.0h;
        float2 bestOff = float2(0.0);
        // refine 레벨(radius=1) 전용: 탐색이 계산한 raw SAD 9개를 레지스터에 보관 —
        // refine이 대다수 픽셀(bestOff가 격자 내부)에서 재-gather 없이 재사용 (36 gather 절감).
        // 값은 페널티 이전 raw = 기존 refine의 raw gather와 동일 → 비트 동일 거동.
        half sad9[3][3];
        bool cache9 = (p.refine != 0) && (p.searchRadius == 1);
        for (int oy = -p.searchRadius; oy <= p.searchRadius; oy++) {
            for (int ox = -p.searchRadius; ox <= p.searchRadius; ox++) {
                float2 cand = base + float2(ox, oy);
                half rawSad = sad6x6(ref, s, patch, float2(gid) + cand, size);
                if (cache9) sad9[oy + 1][ox + 1] = rawSad;
                // 평활 페널티 — 애매(평탄) 영역에서 벡터가 prior에서 멋대로 점프하는
                // 노이즈 억제 (shimmer의 주범). 36탭 SAD 스케일 기준 (25탭 0.012 × 36/25).
                half sad = rawSad + half(p.penalty) * half(length(float2(ox, oy)));
                // 영점 편향(코스 레벨만) — 애매 영역에서 시간 prior에 고착된 벡터를 0으로 되돌린다.
                if (p.zeroBias > 0.0) sad += half(p.zeroBias) * half(fabs(cand.x) + fabs(cand.y));
                if (sad < bestSAD) { bestSAD = sad; bestOff = cand; }
            }
        }
        // 영점 창: prior가 탐색 반경 밖이면 0 주변도 탐색 — 평활 페널티는 prior 기준 그대로,
        // 영점 편향은 |cand| 기준. SAD가 평탄하면 0이 이기고(penalty·|p| < zeroBias·|p|),
        // 텍스처가 있으면 SAD 차이가 결정한다. 코스 레벨(40×22)이라 비용은 무시할 수준.
        if (p.zeroBias > 0.0 && p.hasPrior != 0 &&
            (fabs(base.x) > float(p.searchRadius) || fabs(base.y) > float(p.searchRadius))) {
            for (int oy = -p.searchRadius; oy <= p.searchRadius; oy++) {
                for (int ox = -p.searchRadius; ox <= p.searchRadius; ox++) {
                    float2 cand = float2(ox, oy);
                    half rawSad = sad6x6(ref, s, patch, float2(gid) + cand, size);
                    half sad = rawSad + half(p.penalty) * half(length(cand - base))
                             + half(p.zeroBias) * half(fabs(cand.x) + fabs(cand.y));
                    if (sad < bestSAD) { bestSAD = sad; bestOff = cand; }
                }
            }
        }

        // 서브픽셀 refine (x/y 독립 3점 포물선) — 최종 레벨만. ±1 시프트도 gather 정렬 유지.
        // 이웃 SAD는 sad9 캐시에서 조회, bestOff가 탐색 격자 경계면 그 방향만 gather 폴백.
        float2 refined = bestOff;
        if (p.refine != 0) {
            float sadC = float(bestSAD);
            float2 c = float2(gid) + bestOff;
            int bi = cache9 ? int(bestOff.x - base.x) : 2;   // ∈[-1,1], 2=캐시 미사용
            int bj = cache9 ? int(bestOff.y - base.y) : 2;
            half sL = (bi > -1 && bi < 2 && bj < 2) ? sad9[bj + 1][bi]
                                                    : sad6x6(ref, s, patch, c + float2(-1, 0), size);
            half sR = (bi < 1)                      ? sad9[bj + 1][bi + 2]
                                                    : sad6x6(ref, s, patch, c + float2( 1, 0), size);
            half sU = (bj > -1 && bj < 2 && bi < 2) ? sad9[bj][bi + 1]
                                                    : sad6x6(ref, s, patch, c + float2( 0, -1), size);
            half sD = (bj < 1)                      ? sad9[bj + 2][bi + 1]
                                                    : sad6x6(ref, s, patch, c + float2( 0, 1), size);
            float sadL = float(sL);
            float sadR = float(sR);
            float sadU = float(sU);
            float sadD = float(sD);
            float denomX = sadL - 2.0 * sadC + sadR;
            if (denomX > 1e-5) refined.x += clamp(0.5 * (sadL - sadR) / denomX, -0.5, 0.5);
            float denomY = sadU - 2.0 * sadC + sadD;
            if (denomY > 1e-5) refined.y += clamp(0.5 * (sadU - sadD) / denomY, -0.5, 0.5);
        }

        flowOut.write(float4(refined, 0, 0), gid);
    }

    // 3x3 flow 스무딩 (노이즈로 인한 정적 영역 부들거림 억제).
    // smoothAmt: 원본 대비 박스평균 혼합량 (0=원본 flow=예리, 1=완전 박스=부드러움).
    // "Motion smoothness" 슬라이더의 하단 절반(예리 방향)이 이 값을 낮춰 flow 디테일을 살린다.
    kernel void mfSmooth(
        texture2d<float, access::read> src [[texture(0)]],
        texture2d<float, access::write> dst [[texture(1)]],
        constant float& smoothAmt [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        int W = int(w) - 1, H = int(h) - 1;
        // 접근이 전부 텍셀 정렬 정수 좌표라 bilinear sample()은 낭비 — read()+수동 clamp로
        // 샘플러 유닛 우회 (mfZeroMean과 동일 패턴). 중심 탭은 1회만 읽어 박스합·mix에 재사용.
        float2 center = src.read(gid).rg;
        float2 acc = center;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                uint2 q = uint2(clamp(int(gid.x) + dx, 0, W), clamp(int(gid.y) + dy, 0, H));
                acc += src.read(q).rg;
            }
        float2 box = acc / 9.0;
        dst.write(float4(mix(center, box, smoothAmt), 0, 0), gid);
    }

    // 신뢰도(순환 일관성) + 정적(|A-B|) 마스크 + 루마 히스토그램 (장면컷)
    kernel void mfFinalize(
        texture2d<half, access::sample> lumA [[texture(0)]],
        texture2d<half, access::sample> lumB [[texture(1)]],
        texture2d<float, access::sample> flowF [[texture(2)]],
        texture2d<float, access::sample> flowB [[texture(3)]],
        texture2d<float, access::write> mask [[texture(4)]],
        device atomic_uint* hist [[buffer(0)]],
        constant FinalizeParams& fp [[buffer(1)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = mask.get_width(), h = mask.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 size = float2(w, h);
        float2 uv = (float2(gid) + 0.5) / size;

        // 방향별 순환 검사 — 오클루전 판정의 핵심:
        // A에서 가려지는(사라지는) 픽셀은 F가 무효 → cycF 큼, B에서 새로 드러나는 픽셀은 그 반대.
        // 두 방향을 따로 재면 워프가 "보이는 쪽"만 골라 쓸 수 있다 (단방향 워프).
        float2 f = flowF.sample(s, uv).rg;
        float2 uvF = (float2(gid) + 0.5 + f) / size;
        float cycF = length(f + flowB.sample(s, uvF).rg);
        float2 b = flowB.sample(s, uv).rg;
        float2 uvB = (float2(gid) + 0.5 + b) / size;
        float cycB = length(b + flowF.sample(s, uvB).rg);
        // 실영상 압축 노이즈에서 순환 오차 ~2px는 정상 — 과민하면 화면 대부분이
        // 원본 폴백(60fps 스텝)으로 빠져 '프레임레이트 낮아 보임' (실측 보고)
        // 문턱을 모션 크기에 비례해 늘림 — 큰 변위에서 순환 오차가 커지는 건 정상이다.
        float relF = fp.confRel * length(f), relB = fp.confRel * length(b);
        float confF = 1.0 - smoothstep(fp.confLo + relF, fp.confHi + relF, cycF);
        float confB = 1.0 - smoothstep(fp.confLo + relB, fp.confHi + relB, cycB);

        // 광도 검증(brightness constancy): flow를 따라간 곳의 밝기가 다르면 그 방향 기각.
        // 순환 일관성만으론 "일관되게 틀린" flow(반복 패턴 aliasing)를 못 걸러냄 —
        // 잘못된 워프가 확신을 얻는 것을 막는 2차 방어선. 문턱은 정적 마스크(0.008~0.04)보다 관대.
        float la = float(lumA.sample(s, uv).r);
        float lb = float(lumB.sample(s, uv).r);
        float errF = fabs(float(lumB.sample(s, uvF).r) - la);
        float errB = fabs(float(lumA.sample(s, uvB).r) - lb);
        confF *= 1.0 - smoothstep(fp.photoLo, fp.photoHi, errF);
        confB *= 1.0 - smoothstep(fp.photoLo, fp.photoHi, errB);
        float d = fabs(la - lb);
        float staticness = 1.0 - smoothstep(fp.statLo, fp.statHi, d);

        mask.write(float4(confF, staticness, confB, 0), gid);

        if ((gid.x & 3) == 0 && (gid.y & 3) == 0) {
            uint binA = uint(clamp(la, 0.0, 0.999) * 32.0);
            uint binB = uint(clamp(lb, 0.0, 0.999) * 32.0);
            atomic_fetch_add_explicit(&hist[binA], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&hist[32u + binB], 1u, memory_order_relaxed);
        }
    }

    // 풀해상도 양방향 워프 + 합성.
    // flow는 base-res 픽셀 단위 → 풀해상도 UV 오프셋으로 변환해 샘플.
    kernel void mfWarp(
        texture2d<half, access::sample> imgA [[texture(0)]],
        texture2d<half, access::sample> imgB [[texture(1)]],
        texture2d<float, access::sample> flowF [[texture(2)]],
        texture2d<float, access::sample> flowB [[texture(3)]],
        texture2d<float, access::sample> mask [[texture(4)]],
        texture2d<half, access::write> dst [[texture(5)]],
        constant WarpParams& p [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float t = p.t;

        float2 baseSize = float2(flowF.get_width(), flowF.get_height());
        // flow(base 픽셀 단위) → 정규화 UV 오프셋
        float2 fRaw = flowF.sample(s, uv).rg;
        float2 bRaw = flowB.sample(s, uv).rg;
        // 부드러움 상단: flow를 4-이웃 평균으로 블러 → 워프가 완만(모션 경계 에러 부드럽게).
        // 주의(의도된 근사): mask(conf/staticness)는 블러 *이전* flow 기준(mfFinalize) —
        // 둘 다 이미 박스 스무딩된 flow를 공유하고 여기의 ±1.3px 4탭은 소폭 추가 블러라
        // 경계 게이트 어긋남은 미미. 블러 flow를 finalize에 재공급하는 비용이 이득보다 큼(감사 검증).
        if (p.flowBlur > 0.001) {
            float2 e = 1.3 / baseSize;
            float2 fb = (flowF.sample(s, uv + float2(e.x,0)).rg + flowF.sample(s, uv - float2(e.x,0)).rg
                       + flowF.sample(s, uv + float2(0,e.y)).rg + flowF.sample(s, uv - float2(0,e.y)).rg) * 0.25;
            float2 bb = (flowB.sample(s, uv + float2(e.x,0)).rg + flowB.sample(s, uv - float2(e.x,0)).rg
                       + flowB.sample(s, uv + float2(0,e.y)).rg + flowB.sample(s, uv - float2(0,e.y)).rg) * 0.25;
            fRaw = mix(fRaw, fb, p.flowBlur);
            bRaw = mix(bRaw, bb, p.flowBlur);
        }
        float2 f = fRaw / baseSize;
        float2 b = bRaw / baseSize;

        // backward 매핑: 출력 p의 물체는 A에서 p - t·F_ab 에, B에서 p - (1-t)·F_ba 에 있었다
        half3 w0 = imgA.sample(s, uv - f * t).rgb;
        half3 w1 = imgB.sample(s, uv - b * (1.0 - t)).rgb;

        float4 m = mask.sample(s, uv);
        float confF = m.r;
        float staticness = m.g;
        float confB = m.b;

        // 오클루전 합성 — dirBlend(0=기존, 1=방향별)로 A/B 가능.
        // 방향별: 한쪽 flow만 일관한 가림/드러남 영역은 보이는 쪽 단방향 워프 + 게이트 상향
        //   (기존엔 이 영역이 통째로 원본 폴백 → 가림 경계 60fps 스텝/고스트).
        // 주의: 반복 패턴에선 "일관되게 틀린"(aliased) flow에 확신을 줄 리스크 —
        //   합성 줄무늬 벤치 실측 -3dB (병리적 최악 케이스). 실영상 육안 A/B 전까지 기본 0.
        // 정적 UI는 이 엔진에 들어오기 전에 지워진다(UILayer — MacFG_UILayer_Plan.md). 여기엔 UI 경로가 없다.
        half3 bOrig = imgB.sample(s, uv).rgb;   // 원본 UV — 폴백/정적 합성 공용
        half3 aOrig = imgA.sample(s, uv).rgb;
        float wa = confF * (1.0 - t);
        float wb = confB * t;
        float denom = wa + wb;
        float dirFactor = (denom > 1e-4) ? (wb / denom) : t;
        float tBlend = mix(t, dirFactor, fabs(confF - confB) * p.dirBlend);
        float cRaw = mix(confF, max(confF, confB), max(p.dirBlend, p.confMax));
        half3 interp = mix(w0, w1, half(tBlend));
        half conf = half(p.confGamma == 1.0 ? cRaw : pow(cRaw, p.confGamma));

        // 저신뢰 폴백: A/B 원본 크로스페이드. 폭(fadeLo~fadeHi)이 smoothness 슬라이더:
        // 좁으면(예리) 단일 프레임에 가까워 저더, 넓으면(부드러움) 부드러운 블렌드(약간 고스트).
        half3 nearestPix = mix(aOrig, bOrig, half(smoothstep(p.fadeLo, p.fadeHi, t)));
        half3 moving = mix(nearestPix, interp, conf);
        half3 outc = mix(moving, bOrig, half(staticness)); // 정적 → B 원본 (선명)
        if (p.debug > 0.5) {
            int dm = int(p.debug + 0.5);
            half4 dbg = half4(0.0h, 0.0h, 0.0h, 1.0h);
            if (dm == 1) dbg.rgb = half3(0.0h, 0.0h, half(min(length(fRaw) / 100.0, 1.0)));
            else if (dm == 2) dbg.rgb = half3(half(confF), half(confB), half(staticness));
            else if (dm == 3) dbg.rgb = half3(half(tBlend), conf, 0.0h);
            else if (dm == 5) dbg.rgb = w0;
            else if (dm == 6) dbg.rgb = w1;
            else if (dm == 7) dbg.rgb = nearestPix;
            else if (dm == 8) dbg.rgb = interp;
            dst.write(dbg, gid);
            return;
        }
        dst.write(half4(outc, 1.0h), gid);
    }
    """
}
