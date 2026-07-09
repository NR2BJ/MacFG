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
    private var captureScale: CGFloat = 2.0

    public var isAvailable: Bool { true }

    /// 대상 창 닫힘 등으로 스트림이 비정상 중단됐을 때 (즉시). 우리가 stopCapture하면 호출 안 됨.
    public var onStreamStopped: (@Sendable () -> Void)?

    public init() {}

    public func startCapture(windowID: CGWindowID, device: any MTLDevice, captureRect: CGRect? = nil) async throws {
        self.device = device
        self.captureRect = captureRect

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
        DiagnosticLog.shared.log("[SCK-CFG] start: window.frame=\(Int(window.frame.width))x\(Int(window.frame.height)) scale=\(scaleFactor) region=\(captureRect.map { "\(Int($0.width))x\(Int($0.height))@\(Int($0.minX)),\(Int($0.minY))" } ?? "full") → config \(w)x\(h)")

        let handler = StreamOutputHandler(device: device) { [weak self] slot in
            guard let self else { return }
            self.lock.lock()
            self.latestSlot = slot
            self.pendingSlots.append(slot)
            // drain이 멈춰도 무한 성장 방지 (오래된 것부터 폐기)
            if self.pendingSlots.count > 8 {
                self.pendingSlots.removeFirst(self.pendingSlots.count - 8)
            }
            self.lock.unlock()
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
        try stream.addStreamOutput(handler, type: .screen, sampleHandlerQueue: .global(qos: .userInteractive))
        try await stream.startCapture()

        self.stream = stream
        self.capturing = true

        logger.info("SCK capture started for window \(windowID)")
    }

    /// 공용 스트림 설정 — startCapture와 updateConfiguration이 공유
    private static func makeConfig(width: Int, height: Int, sourceRect: CGRect? = nil) -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.width = max(width, 2)
        config.height = max(height, 2)
        // 영역 캡처: 소스 창 좌상단 기준 크롭 (pt). 지정 시 영상 영역만 잘라 캡처.
        if let sourceRect { config.sourceRect = sourceRect }
        config.captureResolution = .best
        config.pixelFormat = kCVPixelFormatType_32BGRA
        // 1/60 게이트는 콘텐츠 60fps와 위상이 어긋나면 맥놀이로 프레임을 걸러냄 (실측 57-58fps 구멍).
        // 1/120으로 열고 중복 제거는 소비자(status+fingerprint)가 담당.
        config.minimumFrameInterval = CMTime(value: 1, timescale: 120)
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

        // 콘텐츠 fingerprint 계산 (보간 파이프라인 내부 중복 감지용)
        let fingerprint = Self.computeFingerprint(pixelBuffer: pixelBuffer, width: width, height: height)

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
        let slot = FrameSlot(texture: texture, timestamp: timestamp, width: width, height: height, contentChanged: contentChanged, contentFingerprint: fingerprint, colorSpace: cachedColorSpace)
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
    private static func computeFingerprint(pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> UInt64 {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer),
              width > 0, height > 0 else { return 0 }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)

        var hash: UInt64 = 0xcbf29ce484222325 // FNV-1a offset basis
        // 균등 격자 샘플 (~8k점). 랜덤 384점은 작은 국소 변화(가로 텍스트 선택 몇 글자,
        // ~100×15px)를 확률적으로 놓쳐 프레임을 dup-skip → 화면 정지(실측: 가로 드래그 5초 멈춤,
        // 세로는 줄 단위 큰 변화라 잡혀서 부드러움). 격자로 전 영역을 ~10-30px 간격 커버해
        // 작은 변화도 반드시 샘플에 걸리게 한다.
        let target = 8000
        let aspect = Double(width) / Double(max(height, 1))
        let cols = min(max(Int((Double(target) * aspect).squareRoot()), 16), width)
        let rows = min(max(target / max(cols, 1), 16), height)
        for row in 0..<rows {
            let y = (row * height + height / 2) / rows
            let rowBase = y * bytesPerRow
            for col in 0..<cols {
                let x = (col * width + width / 2) / cols
                let offset = rowBase + x * 4
                // B, G, R만 사용 (A는 항상 255)
                for i in 0..<3 {
                    let quantized = ptr[offset + i] >> 2
                    hash ^= UInt64(quantized)
                    hash &*= 0x100000001b3 // FNV-1a prime
                }
            }
        }
        return hash
    }
}
