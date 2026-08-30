import AppKit
import Metal
import ScreenCaptureKit
import CoreMedia
import QuartzCore
import FramePacing
import Monitoring
import os

/// SCK 스트림 비정상 중단(대상 창 닫힘 등) 감지용 델리게이트 — 즉시 콜백.
/// SCStreamDelegate는 NSObjectProtocol이라 별도 NSObject로 분리.
private final class StreamStopObserver: NSObject, SCStreamDelegate {
    let onStop: @Sendable () -> Void
    init(onStop: @escaping @Sendable () -> Void) { self.onStop = onStop }
    func stream(_ stream: SCStream, didStopWithError error: Error) { onStop() }
}

/// ScreenCaptureKit 기반 캡처 (IOSurface 폴백)
public final class SCKCapture: FrameSource, @unchecked Sendable {
    public let method: CaptureMethod = .screenCaptureKit

    private let logger = Logger(subsystem: "com.macfg", category: "SCKCapture")
    private var device: (any MTLDevice)?
    private var stream: SCStream?
    private var outputHandler: StreamOutputHandler?
    private var stopObserver: StreamStopObserver?
    private var latestSlot: FrameSlot?
    /// drain 대기 프레임 큐 (스케줄러용). 캡처 스레드가 push, 렌더 틱이 drain.
    private var pendingSlots: [FrameSlot] = []
    private var capturing = false
    private let lock = NSLock()

    /// 영역 캡처: 소스 창 좌상단 기준 크롭 사각형(pt). nil이면 창 전체.
    private var captureRect: CGRect?

    /// SCK 샘플 배달 전용 **직렬** 큐.
    ///
    /// 이전에는 .global(qos: .userInteractive)(동시성 큐)를 넘겼는데, StreamOutputHandler의
    /// 상태(prevSamples/curSamples/cachedColorSpace 등)는 "단일 직렬 큐 배달" 전제로 무락이다.
    /// 동시성 큐에서는 앞 콜백이 지문 스캔(4K 8k점 + CVPixelBufferLock) 중 선점된 사이 SCK가
    /// 다음 샘플을 다른 워커에 디스패치할 수 있고, 그러면 withUnsafeBufferPointer 진행 중에
    /// swap/재할당이 겹쳐 해제된 버퍼를 만진다 — 이 저장소가 이미 겪은 "엉뚱한 곳 SIGTRAP"
    /// 부류의 힙 손상이다. 직렬 큐면 전제가 실제로 성립해 핸들러는 계속 무락으로 안전하다.
    private let sampleQueue = DispatchQueue(label: "com.macfg.sck.samples", qos: .userInteractive)
    private var captureScale: CGFloat = 2.0

    public var isAvailable: Bool { true }

    /// 대상 창 닫힘 등으로 스트림이 비정상 중단됐을 때 (즉시). 우리가 stopCapture하면 호출 안 됨.
    public var onStreamStopped: (@Sendable () -> Void)?

    /// 새 프레임이 큐에 들어왔을 때 (캡처 스레드에서, 락 밖). 소비자가 즉시 drain하도록.
    ///
    /// **저장은 락으로 보호한다.** 평범한 `var`로 두면 SCK 샘플 큐가 이 프로퍼티를 *읽는* 동안
    /// MainActor가 *덮어쓸* 수 있는데, 클로저는 참조 카운트되는 박스라 그 동시 접근이 곧
    /// over-release다. 해제된 박스는 힙을 깨뜨리고, 트랩은 한참 뒤 **전혀 무관한 곳**에서
    /// 난다(실측: stablePool·MetalFlow 배열·FrameSlot 세 곳에서 각각 SIGTRAP).
    /// 캡처 재시작 때마다 이 프로퍼티가 교체되므로 노출 창이 반복해서 열렸다.
    private var _onFrameAvailable: (@Sendable () -> Void)?
    public var onFrameAvailable: (@Sendable () -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return _onFrameAvailable }
        set { lock.lock(); _onFrameAvailable = newValue; lock.unlock() }
    }

    public init() {}

    public func startCapture(windowID: CGWindowID, device: any MTLDevice, captureRect: CGRect? = nil) async throws {
        self.device = device
        self.captureRect = captureRect
        self.isDisplayCapture = false   // 새 스트림은 창 캡처로 시작 (전체 재시작 경로 포함)
        self.currentDisplayID = nil

        // 캡처 가능한 창 목록에서 대상 찾기
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
            throw CaptureError.windowNotFound
        }

        let filter = SCContentFilter(desktopIndependentWindow: window)

        // 대상 창이 있는 화면의 배율을 찾아서 적용 (외부 1x ↔ MacBook 2x)
        let scaleFactor = Self.findScaleFactor(for: window.frame)
        self.captureScale = scaleFactor
        // 영역 캡처: sourceRect로 크롭 + 출력은 영역 픽셀 크기. nil이면 창 전체.
        let regionPt = captureRect ?? CGRect(origin: .zero, size: window.frame.size)
        let w = regionPt.width > 0 ? Int(regionPt.width * scaleFactor) : 1920
        let h = regionPt.height > 0 ? Int(regionPt.height * scaleFactor) : 1080
        let config = Self.makeConfig(width: w, height: h, sourceRect: captureRect)
        DiagnosticLog.shared.log("[SCK-CFG] start: window.frame=\(Int(window.frame.width))x\(Int(window.frame.height)) scale=\(scaleFactor) region=\(captureRect.map { "\(Int($0.width))x\(Int($0.height))@\(Int($0.minX)),\(Int($0.minY))" } ?? "full") → config \(w)x\(h)→실제 \(config.width)x\(config.height) fps상한=\(Knob.int("MACFG_SCKFPS") ?? 120) 캡처배율=\(Knob.double("MACFG_CAPSCALE") ?? 1.0)")

        let handler = StreamOutputHandler(device: device) { [weak self] slot in
            guard let self else { return }
            self.lock.lock()
            self.latestSlot = slot
            self.pendingSlots.append(slot)
            // drain이 멈춰도 무한 성장 방지 (오래된 것부터 폐기)
            if self.pendingSlots.count > 8 {
                self.pendingSlots.removeFirst(self.pendingSlots.count - 8)
            }
            // 콜백을 **락 안에서 지역 변수로 복사**한 뒤, 호출은 락 밖에서 한다.
            // 복사가 락 안이라 박스의 retain이 교체와 겹치지 않고, 호출이 락 밖이라
            // 재진입 안전은 그대로다(소비자가 drainFrames를 불러도 교착하지 않는다).
            let notify = self._onFrameAvailable
            self.lock.unlock()
            // 도착 즉시 알림 — 소비자가 렌더 틱을 기다리지 않고 인제스트를 시작할 수 있게.
            // 틱을 기다리면 평균 ½틱(~4.2ms)이 그냥 버려진다.
            notify?()
        }
        self.outputHandler = handler

        // 대상 창 닫힘 → SCK가 스트림을 중단 → 즉시 콜백 (폴링 대기 없이)
        let observer = StreamStopObserver { [weak self] in
            guard let self, self.capturing else { return }
            self.capturing = false
            self.onStreamStopped?()
        }
        self.stopObserver = observer

        let stream = SCStream(filter: filter, configuration: config, delegate: observer)
        try stream.addStreamOutput(handler, type: .screen, sampleHandlerQueue: sampleQueue)
        try await stream.startCapture()

        self.stream = stream
        self.capturing = true

        logger.info("SCK capture started for window \(windowID)")
    }

    /// 공용 스트림 설정 — startCapture와 updateConfiguration이 공유
    private static func makeConfig(width: Int, height: Int, sourceRect: CGRect? = nil) -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        // MACFG_CAPSCALE — 캡처 **출력 해상도** 배율 (기본 1.0 = 소스 픽셀 그대로).
        //
        // 확정된 원인 구조: present는 공짜인데(프로브 실측, WindowServer +1.3포인트)
        // 캡처는 +36포인트다. 그리고 캡처만으로도, 많이 내미는 것만으로도 프레임은 안 죽는다 —
        // 둘이 겹칠 때만 죽는다(캡처 중 64장/s present → 미표시 3.8% / 114장/s → 20%).
        // WindowServer의 프레임당 예산(144Hz면 6.94ms)을 캡처 작업이 잠식해 합성이 마감을
        // 놓치는 구조다. 그래서 줄여야 할 것은 캡처 **빈도**가 아니라 캡처 **작업량**이다
        // (빈도는 실측으로 기각됐다 — 120→90에서 오히려 나빠졌다).
        // 0.5면 SCK가 옮기는 픽셀이 1/4이 된다. 보간 입력 해상도도 같이 낮아지므로
        // 화질은 손해지만, 출력은 업스케일로 만회할 수 있다.
        let capScale = min(max(Knob.double("MACFG_CAPSCALE") ?? 1.0, 0.25), 1.0)
        config.width = max(Int(Double(width) * capScale), 2)
        config.height = max(Int(Double(height) * capScale), 2)
        // 영역 캡처: 소스 창 좌상단 기준 크롭 (pt). 지정 시 영상 영역만 잘라 캡처.
        if let sourceRect { config.sourceRect = sourceRect }
        config.captureResolution = .best
        config.pixelFormat = kCVPixelFormatType_32BGRA
        // 1/60 게이트는 콘텐츠 60fps와 위상이 어긋나면 맥놀이로 프레임을 걸러냄 (실측 57-58fps 구멍).
        // 1/120으로 열고 중복 제거는 소비자(status+fingerprint)가 담당.
        //
        // **다만 그 결정의 WindowServer 비용을 아무도 재지 않았다.** 캡처는 WindowServer가
        // 수행하는 일이고, 4K 창을 초당 120장 뜨는 것은 공짜가 아니다. 실측(2026-08-08):
        //   TestPattern(4K 60fps)만 실행           → WindowServer 41.1%
        //   + 별도 프로브가 4K를 초당 114장 present → 42.4%  (present는 사실상 공짜)
        //   MacFG 캡처 중                          → 77%   (+36포인트)
        // 그리고 받은 프레임의 42.7%는 지문이 동일해 그대로 버려진다 — 60fps 소스에서
        // 쓸모없는 4K 캡처를 초당 45장 더 시키고 있다는 뜻이다. 그 부하가 컴포지터를 포화시키고
        // (한 실행 안 r(WindowServer CPU, 미표시) = +0.858), 우리가 제때 넘긴 프레임이 버려진다.
        // 우리 GPU·CPU·보간 연산·해상도·창 설정은 모두 측정으로 배제됐다.
        let fpsCap = Knob.int("MACFG_SCKFPS").map { min(max($0, 24), 240) } ?? 120
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(fpsCap))
        config.queueDepth = 8
        config.showsCursor = false
        config.capturesAudio = false
        return config
    }

    /// 스트림을 끊지 않고 출력 크기만 변경 — 창 리사이즈/전체화면 전환 시 프레임 끊김 없이.
    /// 전체 stop→start 재시작이 유발하는 수 초 붕괴(프레임 갭 + 재워밍업)를 없앤다.
    public func updateConfiguration(width: Int, height: Int) async throws {
        guard let stream else { throw CaptureError.notCapturing }
        // 영역 캡처 중이면 sourceRect 유지 (창 리사이즈 재구성이 크롭을 날리지 않게)
        try await stream.updateConfiguration(Self.makeConfig(width: width, height: height, sourceRect: captureRect))
        logger.info("SCK config updated → \(width)x\(height)")
        DiagnosticLog.shared.log("[SCK-CFG] reconfigure → \(width)x\(height)\(captureRect != nil ? " (region)" : "")")
    }

    /// 창 캡처 ↔ 디스플레이 캡처 무중단 전환.
    ///
    /// 소스가 자체 Space 전체화면이면 창 캡처가 합성 전 창 버퍼를 주는데, 그게 화면 크기와
    /// 다를 수 있다 (실측: 디스코드 전체화면에서 창 3840x2160인데 버퍼 콘텐츠는 3764x2117,
    /// 나머지는 알파 0). 네이티브로 볼 땐 윈도우서버가 합성하며 맞춰주므로 멀쩡한데, 우리가
    /// 캡처해 다시 그리면 그 차이가 여백으로 드러난다. 원인(왜 3764인지)은 디스코드/윈도우서버
    /// 내부라 규명 못 했지만, 전체화면일 땐 소스가 화면 전체를 차지하므로 **디스플레이를 캡처하면
    /// 합성 결과 그대로** 얻는다 — 크롭 같은 추정 보정 없이 정직하게 보이는 것을 보여준다.
    ///
    /// 우리 오버레이/뷰어 창은 반드시 제외해야 한다 (안 그러면 자기 출력을 되먹는 무한 거울).
    /// - Parameter requiredWindowID: **반드시** 제외돼야 하는 창(=우리 출력 창). 0이면 검사 생략.
    ///   이게 빠지면 우리 출력을 우리가 다시 캡처해 되먹임 거울이 된다.
    /// **디스플레이 캡처 중 출력 창이 재생성됐을 때 제외 목록을 다시 박는다.**
    ///
    /// 제외 목록은 `updateToDisplayCapture` 시점에 한 번 굳는다. 그 뒤 오버레이 창이
    /// 재생성되면(배치 전환·리사이즈·화면 이동) **새 windowID가 필터에 없어 우리 출력이
    /// 캡처에 들어온다** — 잡힌 출력을 다시 그리고 그게 또 잡히는 되먹임이다.
    /// 실사용 증상(2026-08-31): 창모드에서 배치를 토글하자 창 안에 창이 무한히 겹쳤다.
    /// isDisplayCapture는 건드리지 않는다 — 모드 전환이 아니라 필터 갱신이다.
    public func refreshDisplayExclusions(excludingWindowIDs: [CGWindowID],
                                        requiredWindowID: CGWindowID = 0) async {
        guard isDisplayCapture, let stream, let displayID = currentDisplayID else { return }
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true),
              let display = content.displays.first(where: { $0.displayID == displayID }) else { return }
        let excluded = content.windows.filter { excludingWindowIDs.contains($0.windowID) }
        // 출력 창을 못 빼면 **적용하지 않는다** — 옛 필터가 남는 편이 되먹임보다 낫다.
        if requiredWindowID != 0 && !excluded.contains(where: { $0.windowID == requiredWindowID }) {
            DiagnosticLog.shared.log("[SCK-DISPLAY] ✗ 제외 갱신 실패 — 출력 창(\(requiredWindowID)) 미포함, 옛 필터 유지")
            return
        }
        try? await stream.updateContentFilter(SCContentFilter(display: display, excludingWindows: excluded))
        DiagnosticLog.shared.log("[SCK-DISPLAY] 제외 갱신 \(excluded.count)/\(excludingWindowIDs.count)개 (출력 창 \(requiredWindowID))")
    }

    public func updateToDisplayCapture(displayID: CGDirectDisplayID,
                                       excludingWindowIDs: [CGWindowID],
                                       requiredWindowID: CGWindowID = 0) async throws {
        guard let stream else { throw CaptureError.notCapturing }
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first(where: { $0.displayID == displayID }) else {
            throw CaptureError.windowNotFound
        }
        var excluded = content.windows.filter { excludingWindowIDs.contains($0.windowID) }
        // **제외가 0개면 한 번 다시 본다.**
        // 전체화면 전환은 (a) viewer 창 재생성 → (b) 디스플레이 캡처 전환 순으로 30ms 안에
        // 일어나는데, 갓 만들어진 창은 SCShareableContent 스냅샷에 아직 안 들어와 있을 수 있다.
        // 그러면 우리 출력이 캡처에 포함돼 **자기 출력을 되먹는다** — 전체화면이라 우리 창이
        // 화면을 꽉 채우므로 되먹임이 눈에 안 띄고, 대신 합성 부하만 배로 늘어 프레임이 밀린다
        // (실측 2026-08-06: 전체화면에서 미표시 46% vs 창 모드 15%).
        // **판정 기준은 "몇 개 빠졌나"가 아니라 "출력 창이 빠졌나"다.**
        // 예전 조건은 `excluded.isEmpty`였는데, 우리 창이 둘 이상이면 설정 창 하나만 잡혀도
        // 비어 있지 않아 재시도가 안 걸린다 — 실사용에서 "제외 1/2개"인 채로 진행됐고,
        // 빠진 쪽이 출력 창이면 화면이 멈춘 것처럼 보이고 노이즈가 누적된다(제보 2026-08-07).
        // 갓 만든 창은 SCShareableContent 스냅샷에 늦게 들어오므로 짧게 여러 번 다시 본다.
        func hasRequired(_ list: [SCWindow]) -> Bool {
            requiredWindowID == 0 || list.contains { $0.windowID == requiredWindowID }
        }
        var attempt = 0
        while !hasRequired(excluded) && attempt < 5 {
            attempt += 1
            try? await Task.sleep(for: .milliseconds(120))
            guard let retry = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            else { break }
            excluded = retry.windows.filter { excludingWindowIDs.contains($0.windowID) }
        }
        if attempt > 0 {
            DiagnosticLog.shared.log("[SCK-DISPLAY] 제외 재조회 \(attempt)회 → \(excluded.count)/\(excludingWindowIDs.count)개"
                + (hasRequired(excluded) ? " (출력 창 확보)" : " **출력 창 여전히 미포함**"))
        }
        // **출력 창을 못 뺐으면 여기서 실패한다 — 적용하면 안 된다.** 재시도 소진 후에도
        // 그대로 적용하면 자기 출력을 되먹는 거울이 되고(아래 주석의 실사용 증상), 한번
        // isDisplayCapture=true로 굳으면 호출측 재진입 가드("이미 원하는 모드")가 재시도를
        // 영영 막아 실패가 끈적해진다(리뷰 확정). throw하면 호출측 catch가 로그를 남기고
        // 기존 창 캡처가 유지되며, 전체화면 상태가 지속되는 한 다음 재평가가 다시 시도한다.
        if requiredWindowID != 0 && !hasRequired(excluded) {
            DiagnosticLog.shared.log("[SCK-DISPLAY] ✗ 출력 창(\(requiredWindowID)) 제외 실패 — 디스플레이 전환 중단, 창 캡처 유지")
            throw CaptureError.windowNotFound
        }
        self.captureRect = nil
        self.captureScale = Self.findScaleFactor(for: display.frame)
        let w = display.width, h = display.height          // 디스플레이는 이미 픽셀 단위
        try await stream.updateContentFilter(SCContentFilter(display: display, excludingWindows: excluded))
        try await stream.updateConfiguration(Self.makeConfig(width: w, height: h))
        isDisplayCapture = true
        currentDisplayID = displayID   // 제외 목록 갱신 때 같은 디스플레이를 다시 찾기 위해
        // **어느 창이 빠졌는지까지 남긴다.**
        // "제외 N/M개"만으로는 빠진 게 설정 창인지 **출력 뷰어**인지 알 수 없는데, 그 차이가
        // 전부다: 출력 창이 안 빠지면 우리 출력을 우리가 다시 캡처해 **되먹임 거울**이 된다.
        // 전체화면에서는 뷰어가 화면을 꽉 채우므로 되먹임이 눈에 안 띄고, 대신 진짜 콘텐츠가
        // 영영 안 잡혀 **화면이 멈춘 것처럼** 보이며 세대마다 워프 오차가 누적돼 노이즈가 쌓인다
        // (실사용 제보 2026-08-07: "화면 자체가 멈춰" + 전면 노이즈).
        let foundIDs = Set(excluded.map { $0.windowID })
        let missing = excludingWindowIDs.filter { !foundIDs.contains($0) }
        DiagnosticLog.shared.log("[SCK-DISPLAY] → display \(displayID) \(w)x\(h), 제외 창 \(excluded.count)/\(excludingWindowIDs.count)개"
            + " 요청=\(excludingWindowIDs) 적용=\(Array(foundIDs).sorted())"
            + (missing.isEmpty ? "" : " **누락=\(missing)**"))
        if requiredWindowID != 0 && !foundIDs.contains(requiredWindowID) {
            DiagnosticLog.shared.log("[SCK-DISPLAY] ⚠︎ **출력 창(\(requiredWindowID))이 캡처에서 제외되지 않았다** — "
                + "자기 출력을 되먹어 화면이 멈춘 것처럼 보이고 노이즈가 누적된다")
        } else if !missing.isEmpty {
            DiagnosticLog.shared.log("[SCK-DISPLAY] 우리 창 \(missing.count)개 미제외(출력 창은 아님) — 되먹임 위험 낮음")
        }
    }

    /// 디스플레이 캡처 중인지 — 전체화면 이탈 시 창 캡처로 되돌릴지 판단용
    /// 디스플레이 캡처 중인 화면 — 제외 목록 갱신에 필요
    private var currentDisplayID: CGDirectDisplayID?
    public private(set) var isDisplayCapture = false

    /// 캡처 대상 창을 무중단 교체 — 전체화면/PiP가 새 창을 만들 때 재타깃 (updateContentFilter).
    /// captureRect(영역 크롭)는 원 창 기준이라 재타깃 시 무효화하고 새 창 전체를 잡는다.
    public func updateTargetWindow(windowID: CGWindowID) async throws {
        guard let stream else { throw CaptureError.notCapturing }
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: { $0.windowID == windowID }) else {
            throw CaptureError.windowNotFound
        }
        let scaleFactor = Self.findScaleFactor(for: window.frame)
        self.captureScale = scaleFactor
        self.captureRect = nil   // 재타깃 = 새 창 전체 (원 창 기준 크롭 무효)
        let w = window.frame.width > 0 ? Int(window.frame.width * scaleFactor) : 1920
        let h = window.frame.height > 0 ? Int(window.frame.height * scaleFactor) : 1080
        try await stream.updateContentFilter(SCContentFilter(desktopIndependentWindow: window))
        try await stream.updateConfiguration(Self.makeConfig(width: w, height: h))
        isDisplayCapture = false
        DiagnosticLog.shared.log("[SCK-RETARGET] → window \(windowID) \(Int(window.frame.width))x\(Int(window.frame.height)) → cfg \(w)x\(h)")
    }

    public func stopCapture() async {
        capturing = false
        if let stream {
            try? await stream.stopCapture()
        }
        stream = nil
        outputHandler = nil
        stopObserver = nil
        clearSlots()
        logger.info("SCK capture stopped")
    }

    private func clearSlots() {
        lock.lock()
        latestSlot = nil
        pendingSlots = []
        lock.unlock()
    }

    public func latestFrame() -> FrameSlot? {
        lock.lock()
        defer { lock.unlock() }
        return latestSlot
    }

    public func drainFrames() -> [FrameSlot] {
        lock.lock()
        defer { lock.unlock() }
        let drained = pendingSlots
        pendingSlots = []
        return drained
    }

    /// SCShareableContent의 CG 좌표 기반 window.frame → 해당 화면의 backingScaleFactor
    private static func findScaleFactor(for cgFrame: CGRect) -> CGFloat {
        // CG → NS 좌표 변환해서 NSScreen 매칭
        // NSScreen 접근은 어떤 스레드에서든 가능 (읽기 전용)
        let screens = NSScreen.screens
        let primaryH = screens
            .first(where: { $0.frame.origin == .zero })?
            .frame.height ?? 0
        let nsMidX = cgFrame.midX
        let nsMidY = primaryH - cgFrame.midY
        let nsPoint = CGPoint(x: nsMidX, y: nsMidY)

        for screen in screens {
            if screen.frame.contains(nsPoint) {
                return screen.backingScaleFactor
            }
        }
        return 2.0
    }
}

// MARK: - Stream Output Handler

private final class StreamOutputHandler: NSObject, SCStreamOutput, @unchecked Sendable {
    private let device: any MTLDevice
    private let onFrame: (FrameSlot) -> Void
    private let logger = Logger(subsystem: "com.macfg", category: "StreamOutput")
    private var colorSpaceLogged = false
    private var statusLogged = false
    private var detailFrameCount = 0   // 캡처당 리셋(핸들러 새로 생성) — 소스 디테일 진단
    /// 첫 프레임 어태치먼트에서 추출한 캡처 색공간 (이후 프레임에 재사용)
    private var cachedColorSpace: CGColorSpace?
    /// 변화율 산출용 격자 표본 — 직전/현재를 번갈아 쓰며(swap) 매 프레임 할당을 피한다.
    /// SCK는 프레임을 단일 직렬 큐로 배달하므로 이 상태는 락 없이 안전하다.
    private var prevSamples: [UInt8] = []
    private var curSamples: [UInt8] = []
    private var prevSampleCols = 0
    private var prevSampleRows = 0

    init(device: any MTLDevice, onFrame: @escaping (FrameSlot) -> Void) {
        self.device = device
        self.onFrame = onFrame
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen else { return }

        // SCStreamFrameInfo에서 프레임 상태 확인
        // .complete = 새 콘텐츠, .idle = 이전과 동일
        let contentChanged: Bool
        if let attachmentsArray = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[String: Any]],
           let dict = attachmentsArray.first,
           let statusRaw = dict[SCStreamFrameInfo.status.rawValue] as? Int,
           let status = SCFrameStatus(rawValue: statusRaw) {
            contentChanged = (status == .complete)
            // 첫 프레임에서 상태 로깅
            if !statusLogged {
                statusLogged = true
                logger.info("SCK frame status type detected: \(statusRaw) → \(status == .complete ? "complete" : "other")")
            }
        } else {
            // 상태를 읽을 수 없으면 새 콘텐츠로 간주
            contentChanged = true
        }

        // status 기반 필터링은 하지 않는다 — 일부 영상 창에서 idle/other 오분류 이력 (worklog 2026-06-25).
        // 실제 중복 제거는 소비자가 fingerprint로 수행.
        guard let pixelBuffer = sampleBuffer.imageBuffer else { return }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // 소스 디테일 진단: 첫 5프레임의 수평 그래디언트 평균(선명↑/흐림↓). 첫 캡처 소스가
        // 실제 저품질(브라우저 저해상도 렌더)인지, MacFG 출력만 문제인지 판별용.
        if detailFrameCount < 5 {
            detailFrameCount += 1
            let d = Self.detailMetric(pixelBuffer: pixelBuffer, width: width, height: height)
            DiagnosticLog.shared.log("[DETAIL] frame#\(detailFrameCount) \(width)x\(height) grad=\(String(format: "%.2f", d))")
        }

        // 색공간: 픽셀 버퍼 어태치먼트에서 CGColorSpace 생성 (1회, 캐시)
        if cachedColorSpace == nil {
            let attachments = CVBufferCopyAttachments(pixelBuffer, .shouldPropagate)
            if let attachments,
               let cs = CVImageBufferCreateColorSpaceFromAttachments(attachments)?.takeRetainedValue() {
                cachedColorSpace = cs
            }
            if !colorSpaceLogged {
                colorSpaceLogged = true
                let name = cachedColorSpace?.name.map { String($0) } ?? "nil(untagged)"
                logger.warning("[COLOR] capture colorSpace=\(name)")
            }
        }

        // 콘텐츠 fingerprint + 직전 프레임 대비 변화율 (보간 파이프라인 내부 중복/케이던스 판정용)
        let (fingerprint, changeRatio) = fingerprintAndChange(pixelBuffer: pixelBuffer, width: width, height: height)

        // CVPixelBuffer → IOSurface 백킹 MTLTexture (제로카피)
        guard let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
            logger.debug("No IOSurface backing for pixel buffer")
            return
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: desc, iosurface: ioSurface, plane: 0) else {
            return
        }

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        // pixelBuffer를 함께 싣는다 — texture가 이 버퍼의 IOSurface를 제로카피로 가리키므로,
        // 소비(=stable blit)가 끝날 때까지 살아 있어야 풀이 같은 표면에 다음 프레임을 덮어쓰지 않는다.
        let slot = FrameSlot(texture: texture, timestamp: timestamp, width: width, height: height, contentChanged: contentChanged, contentFingerprint: fingerprint, changeRatio: changeRatio, colorSpace: cachedColorSpace, pixelBuffer: pixelBuffer)
        onFrame(slot)
    }

    /// 소스 프레임 선명도 지표 — 중앙 영역 인접 픽셀(green) 그래디언트 평균. 높을수록 디테일↑.
    /// 선명한 1080p 소스는 높고, 저해상도를 늘린 흐린 소스는 낮다.
    private static func detailMetric(pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer), width > 8, height > 8 else { return 0 }
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var sum: Double = 0, count = 0
        var y = height / 4
        while y < height * 3 / 4 {
            let row = ptr + y * bpr
            var x = width / 4
            while x < width * 3 / 4 - 1 {
                sum += Double(abs(Int(row[(x + 1) * 4 + 1]) - Int(row[x * 4 + 1])))   // BGRA green
                count += 1; x += 2
            }
            y += max(1, height / 40)
        }
        return count > 0 ? sum / Double(count) : 0
    }

    /// CVPixelBuffer에서 384개 흩뿌린(pseudo-random 고정 좌표) 샘플을 읽어 해시 생성.
    /// 격자 샘플링은 주기적 콘텐츠(체커보드 등)와 정렬되어 실제 이동을 놓치는 앨리어싱 실측
    /// (60fps 패턴이 47fps로 판정) — 산포 좌표는 어떤 이동이든 다수 샘플을 교차한다.
    /// 양자화(>>2)로 압축 노이즈 무시.
    /// 콘텐츠 지문과 **직전 프레임 대비 변화율**을 함께 낸다.
    ///
    /// 지문 하나로는 "바뀌었다/아니다"라는 이진 판정밖에 못 하는데, 그 스위치 하나가 서로 다른
    /// 두 요구를 동시에 떠맡고 있었다:
    ///
    /// 1. **민감해야 한다** — 작은 국소 변화(텍스트 선택 ~100×15px)를 놓치면 dup-skip으로
    ///    화면이 멈춘다(실측: 가로 드래그 5초 정지). 그래서 랜덤 384점을 8k 격자로 바꿨다.
    /// 2. **둔감해야 한다** — 민감하면 캐럿 깜빡임·스크롤바·브라우저 크롬 같은 UI 리페인트도
    ///    새 소스 프레임으로 세어, 60fps 영상 창인데 입력 케이던스가 111fps로 잡힌다(실측).
    ///    그러면 보간 배수가 무너지고(생성/입력 1.60→0.78) 4K에서는 쓸모없는 flow 계산까지
    ///    두 배로 든다.
    ///
    /// 변화의 **크기**를 같이 재면 둘을 분리할 수 있다 — 영상이 진행하면 표본의 상당수가 바뀌고,
    /// UI만 깜빡이면 1% 미만이 바뀐다. 표시는 지금처럼 모든 변화를 받아 멈춤을 막고(요구 1),
    /// 케이던스 추정만 큰 변화로 게이팅하면 된다(요구 2). 판정은 소비자(AppState)가 한다.
    ///
    /// 추가 비용은 격자 표본 8k×3바이트를 다음 프레임까지 보관하고 점별로 비교하는 것뿐이다.
    private func fingerprintAndChange(pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> (hash: UInt64, changeRatio: Float) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer),
              width > 0, height > 0 else { return (0, 1) }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)

        var hash: UInt64 = 0xcbf29ce484222325 // FNV-1a offset basis
        // 균등 격자 샘플 (~8k점). 전 영역을 ~10-30px 간격으로 덮어 작은 변화도 표본에 걸리게 한다.
        let target = 8000
        let aspect = Double(width) / Double(max(height, 1))
        let cols = min(max(Int((Double(target) * aspect).squareRoot()), 16), width)
        let rows = min(max(target / max(cols, 1), 16), height)

        // 해상도가 바뀌면 격자 자체가 달라져 점별 비교가 무의미하다 — 표본을 버리고 다시 시작.
        let sampleCount = rows * cols * 3
        if prevSampleCols != cols || prevSampleRows != rows {
            prevSampleCols = cols
            prevSampleRows = rows
            prevSamples = []
        }
        let hasPrev = prevSamples.count == sampleCount
        if curSamples.count != sampleCount {
            curSamples = [UInt8](repeating: 0, count: sampleCount)
        }
        var changed = 0

        curSamples.withUnsafeMutableBufferPointer { cur in
            prevSamples.withUnsafeBufferPointer { prev in
                var s = 0
                for row in 0..<rows {
                    let y = (row * height + height / 2) / rows
                    let rowBase = y * bytesPerRow
                    for col in 0..<cols {
                        let x = (col * width + width / 2) / cols
                        let offset = rowBase + x * 4
                        // B, G, R만 사용 (A는 항상 255). 양자화(>>2)로 압축 노이즈 무시.
                        for i in 0..<3 {
                            let quantized = ptr[offset + i] >> 2
                            hash ^= UInt64(quantized)
                            hash &*= 0x100000001b3 // FNV-1a prime
                            cur[s] = quantized
                            if hasPrev, prev[s] != quantized { changed += 1 }
                            s += 1
                        }
                    }
                }
            }
        }
        swap(&prevSamples, &curSamples)

        // 첫 프레임(비교 대상 없음)은 변화 100%로 본다 — 케이던스 게이트가 초기에 프레임을
        // 삼켜 파이프라인이 시작조차 못 하는 것을 막는다.
        let ratio = hasPrev ? Float(changed) / Float(sampleCount) : 1
        return (hash, ratio)
    }
}
