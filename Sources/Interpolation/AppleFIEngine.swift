@preconcurrency import Metal
import MetalFX
import VideoToolbox
import CoreVideo
import CoreMedia
import Monitoring
import os

/// Apple VTLowLatencyFrameInterpolation 기반 보간 엔진 (macOS 26+).
///
/// 프로브 실측 (M4, macOS 26.5.1, 2026-07-02):
/// - 모델은 정확히 1280x720 420v만 지원. 그 외 해상도는 process에서 -19730.
/// - 720p 보간 1회 = ANE ~3.4ms (GPU 점유 없음). cmdBuf 삽입 경로 동작 확인.
///
/// 파이프라인 (모두 단일 command buffer에 인코딩):
///   stableA/B (BGRA, 원본 크기)
///     → bgra_to_420v 다운스케일+CSC (GPU) → 720p 420v CVPixelBuffer
///     → maskHist: |lumaA-lumaB| 모션 마스크 + 루마 히스토그램 (장면 전환 판정용)
///     → VTFrameProcessor.process(with: cb) (ANE)
///     → 420v→BGRA 720p → MetalFX spatial 업스케일 → upscaledTmp (원본 크기)
///     → compositeMasked: 정적 영역=stableB 원본 픽셀, 모션 영역=업스케일된 보간 픽셀
///
/// compositeMasked가 핵심 품질 장치 — 보간 프레임의 720p 소프트함이 정적 UI/텍스트에
/// 나타나 소스 프레임과 60Hz로 "선명도 와리가리"하던 문제를 제거한다 (움직이는 영역의
/// 소프트함은 시각적으로 마스킹됨). CSC는 BT.709 video-range 왕복 (자기역행렬).
public final class AppleFIEngine: PairInterpolationEngine {
    public let name = "Apple FI (ANE 720p)"

    public static var isSupported: Bool {
        guard #available(macOS 26.0, *) else { return false }
        return VTLowLatencyFrameInterpolationConfiguration.isSupported
    }

    // LLFI 모델 고정 해상도 (프로브로 확인)
    private let fiWidth = 1280
    private let fiHeight = 720

    // 장면 전환 판정: 루마 히스토그램 교집합이 이 값 미만이면 컷
    /// 0.25: 빠른 게임(오버워치 시점 회전/이펙트)이 히스토그램을 크게 흔들어 0.5에선 초당 수 회
    /// 오검출 → 보간 폐기 → 25~50ms 구멍(실측 cut=1~9/2s). 진짜 하드컷은 교집합이 ~0이라 안전.
    private let sceneCutIntersectionThreshold = 0.25

    private var device: (any MTLDevice)?
    private var processor: VTFrameProcessor?
    private var sessionActive = false
    /// 세션 지연 복구 스로틀 (마지막 재시도 시각)
    private var lastSessionRetry: CFTimeInterval = 0
    private var lastVTTimestamp: CMTime = .invalid

    // 720p 420v 버퍼 3개: prev / src / dst — 각각 Y/CbCr plane 텍스처 뷰
    private struct PlanarBuffer {
        let pixelBuffer: CVPixelBuffer
        let lumaTexture: any MTLTexture
        let chromaTexture: any MTLTexture
    }
    private var prevBuf: PlanarBuffer?
    private var srcBuf: PlanarBuffer?
    /// 직전 쌍의 B 추적 — A(=이전 B)가 그대로면 prevBuf 다운스케일 재계산 생략 (역할 스왑)
    private var lastEncodedTsB: CFTimeInterval = -1
    private var lastStableBID: ObjectIdentifier?
    private var dstBuf: PlanarBuffer?

    // 720p 중간들
    private var rgb720: (any MTLTexture)?      // MetalFX 입력
    private var maskTex: (any MTLTexture)?     // 모션 마스크 (r8)
    // 원본 크기 중간: 업스케일 결과 (composite 입력) — work 큐가 직렬이라 단일 재사용 안전
    private var upscaledTmp: (any MTLTexture)?
    // 출력 풀 (원본 크기 BGRA) — 링 4개
    private var outputPool: [any MTLTexture] = []
    // 히스토그램 버퍼 링 (출력 풀과 같은 인덱스): uint32 x 64 (yA 32빈 + yB 32빈)
    private var statsBuffers: [any MTLBuffer] = []
    private var outputIndex = 0
    /// outputPool 슬롯별 현 점유 세대 (표시 직전 isFrameLive 검증용). encode/조회 모두 렌더 스레드.
    private var slotStamps: [UInt64] = []
    private var nextStamp: UInt64 = 0
    /// 화면정지 UI 마스크 (UIStaticDetector) — compositeMasked가 이 영역을 소스B로 프리즈
    private var uiMaskTex: (any MTLTexture)?
    public func setUIMask(_ texture: (any MTLTexture)?) { uiMaskTex = texture }
    private var outputWidth = 0
    private var outputHeight = 0

    private var downscalePSO: (any MTLComputePipelineState)?
    private var maskHistPSO: (any MTLComputePipelineState)?
    private var csc720PSO: (any MTLComputePipelineState)?
    private var upscaleFallbackPSO: (any MTLComputePipelineState)?
    private var compositePSO: (any MTLComputePipelineState)?
    private var spatialScaler: (any MTLFXSpatialScaler)?

    private let logger = Logger(subsystem: "com.macfg", category: "AppleFI")
    private var pairCount = 0

    public init() {}

    // MARK: - Prepare

    public func prepare(device: any MTLDevice) async throws {
        self.device = device

        guard Self.isSupported else {
            throw InterpolationError.notPrepared
        }

        let library = try await device.makeLibrary(source: Self.shaderSource, options: nil)
        func pso(_ name: String) async throws -> any MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw InterpolationError.notPrepared }
            return try await device.makeComputePipelineState(function: fn)
        }
        downscalePSO = try await pso("bgraTo420v")
        maskHistPSO = try await pso("maskHist")
        csc720PSO = try await pso("p420vToBGRA")
        upscaleFallbackPSO = try await pso("p420vToBGRAScaled")
        compositePSO = try await pso("compositeMasked")

        prevBuf = try Self.makePlanarBuffer(width: fiWidth, height: fiHeight, device: device)
        srcBuf = try Self.makePlanarBuffer(width: fiWidth, height: fiHeight, device: device)
        dstBuf = try Self.makePlanarBuffer(width: fiWidth, height: fiHeight, device: device)

        let rgbDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: fiWidth, height: fiHeight, mipmapped: false)
        rgbDesc.usage = [.shaderRead, .shaderWrite]
        rgbDesc.storageMode = .private
        rgb720 = device.makeTexture(descriptor: rgbDesc)

        let maskDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: fiWidth, height: fiHeight, mipmapped: false)
        maskDesc.usage = [.shaderRead, .shaderWrite]
        maskDesc.storageMode = .private
        maskTex = device.makeTexture(descriptor: maskDesc)

        try startVTSession()

        DiagnosticLog.shared.log("[AppleFI] prepared: LLFI 1280x720 + motion-masked composite")
    }

    private func startVTSession() throws {
        guard #available(macOS 26.0, *) else { throw InterpolationError.notPrepared }
        guard let config = VTLowLatencyFrameInterpolationConfiguration(
            frameWidth: fiWidth, frameHeight: fiHeight, numberOfInterpolatedFrames: 1
        ) else {
            throw InterpolationError.notPrepared
        }
        let proc = VTFrameProcessor()
        try proc.startSession(configuration: config)
        processor = proc
        sessionActive = true
        lastVTTimestamp = .invalid
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
        // LLFI는 t=0.5 고정 (모델 제약) — 갭이 커도 중간 한 장만 생성
        guard #available(macOS 26.0, *) else { return nil }
        // 이전 세션 재시작이 실패해 죽어있으면 여기서 재기동 시도 — 없으면 일시적 VT/ANE
        // 실패 한 번에 엔진이 영구 소스-온리로 고착 (reset()은 세션을 안 살림, 감사 확정).
        // VT가 장기 불능이면 매 쌍(60/s) 세션 생성 시도 자체가 비용이라 2초 스로틀.
        if !sessionActive, prevBuf != nil {
            let now = CFAbsoluteTimeGetCurrent()
            if now - lastSessionRetry > 2.0 {
                lastSessionRetry = now
                try? startVTSession()
            }
        }
        // A(=이전 쌍의 B)가 동일 객체·동일 ts면 prevBuf에 넣을 내용이 직전 srcBuf와 비트
        // 동일(결정적 다운스케일) — 역할 스왑으로 A 다운스케일 패스(쌍당 ~0.1-0.3ms GPU) 생략.
        // 연속성 단절(reset/갭/실패) 시 lastEncodedTsB 불일치로 자동 전체 경로 폴백 (감사 확정).
        let reuseA = tsA == lastEncodedTsB && lastStableBID == ObjectIdentifier(stableA as AnyObject)
        if reuseA { swap(&prevBuf, &srcBuf) }

        guard processor != nil, sessionActive,
              let prevBuf, let srcBuf, let dstBuf,
              let downscalePSO, let maskHistPSO, let compositePSO,
              let maskTex,
              tsB > tsA else { return nil }

        ensureOutputPool(width: stableB.width, height: stableB.height)
        guard !outputPool.isEmpty, let upscaledTmp, outputPool.count == statsBuffers.count else { return nil }

        // VT 타임스탬프는 세션 내 단조 증가 필수 — 역행 시 세션 재시작
        let cmA = CMTime(seconds: tsA, preferredTimescale: 1_000_000)
        let cmMid = CMTime(seconds: (tsA + tsB) / 2, preferredTimescale: 1_000_000)
        let cmB = CMTime(seconds: tsB, preferredTimescale: 1_000_000)
        if lastVTTimestamp.isValid && cmA < lastVTTimestamp {
            logger.warning("VT timestamp regression, restarting session")
            processor?.endSession()
            sessionActive = false
            try? startVTSession()
            guard sessionActive else {
                logger.error("VT session restart failed — retrying on next pair")
                return nil   // 다음 encodePair 진입부의 지연 복구가 재시도
            }
        }
        // 재시작 분기 이후에 바인딩 — 분기 전에 바인딩하면 방금 endSession된 옛 인스턴스로
        // process()를 불러 이 쌍의 dst가 안 써지고 이전 쌍 720p가 합성됨 (감사 확정)
        guard let processor else { return nil }

        guard let fPrev = VTFrameProcessorFrame(buffer: prevBuf.pixelBuffer, presentationTimeStamp: cmA),
              let fSrc = VTFrameProcessorFrame(buffer: srcBuf.pixelBuffer, presentationTimeStamp: cmB),
              let fDst = VTFrameProcessorFrame(buffer: dstBuf.pixelBuffer, presentationTimeStamp: cmMid),
              let params = VTLowLatencyFrameInterpolationParameters(
                sourceFrame: fSrc, previousFrame: fPrev,
                interpolationPhase: [0.5], destinationFrames: [fDst]
              ) else { return nil }

        let slotIndex = outputIndex
        let output = outputPool[slotIndex]
        let statsBuffer = statsBuffers[slotIndex]
        outputIndex = (outputIndex + 1) % outputPool.count
        nextStamp &+= 1
        let stamp = nextStamp
        if slotIndex < slotStamps.count { slotStamps[slotIndex] = stamp }   // 이 슬롯의 현 점유 세대

        // 0) 히스토그램 버퍼 클리어
        if let fill = commandBuffer.makeBlitCommandEncoder() {
            fill.fill(buffer: statsBuffer, range: 0..<(64 * 4), value: 0)
            fill.endEncoding()
        }

        // 1) 다운스케일+CSC (A→prev, B→src) + 모션 마스크/히스토그램
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(downscalePSO)
        if !reuseA {
            encodeDownscale(enc, source: stableA, target: prevBuf)   // 재사용 시 생략 (위 스왑)
        }
        encodeDownscale(enc, source: stableB, target: srcBuf)
        enc.setComputePipelineState(maskHistPSO)
        enc.setTexture(prevBuf.lumaTexture, index: 0)
        enc.setTexture(srcBuf.lumaTexture, index: 1)
        enc.setTexture(maskTex, index: 2)
        enc.setBuffer(statsBuffer, offset: 0, index: 0)
        dispatch(enc, width: fiWidth, height: fiHeight, pso: maskHistPSO)
        enc.endEncoding()

        // 2) LLFI (ANE) — 같은 CB에 삽입, 선행 GPU 작업 완료 후 실행됨
        processor.process(with: commandBuffer, parameters: params)
        lastVTTimestamp = cmB

        // 3) 업스케일: dst 420v → (720p BGRA → MetalFX) 또는 bilinear → upscaledTmp
        if let scaler = spatialScaler, let rgb720,
           let enc2 = commandBuffer.makeComputeCommandEncoder(), let csc720PSO {
            enc2.setComputePipelineState(csc720PSO)
            enc2.setTexture(dstBuf.lumaTexture, index: 0)
            enc2.setTexture(dstBuf.chromaTexture, index: 1)
            enc2.setTexture(rgb720, index: 2)
            dispatch(enc2, width: fiWidth, height: fiHeight, pso: csc720PSO)
            enc2.endEncoding()

            scaler.colorTexture = rgb720
            scaler.outputTexture = upscaledTmp
            scaler.encode(commandBuffer: commandBuffer)
        } else if let upscaleFallbackPSO, let enc2 = commandBuffer.makeComputeCommandEncoder() {
            enc2.setComputePipelineState(upscaleFallbackPSO)
            enc2.setTexture(dstBuf.lumaTexture, index: 0)
            enc2.setTexture(dstBuf.chromaTexture, index: 1)
            enc2.setTexture(upscaledTmp, index: 2)
            dispatch(enc2, width: upscaledTmp.width, height: upscaledTmp.height, pso: upscaleFallbackPSO)
            enc2.endEncoding()
        } else {
            return nil
        }

        // 4) 모션 마스크 합성: 정적=원본 B, 모션=보간
        guard let enc3 = commandBuffer.makeComputeCommandEncoder() else { return nil }
        enc3.setComputePipelineState(compositePSO)
        enc3.setTexture(stableB, index: 0)
        enc3.setTexture(upscaledTmp, index: 1)
        enc3.setTexture(maskTex, index: 2)
        enc3.setTexture(output, index: 3)
        enc3.setTexture(uiMaskTex ?? stableB, index: 4)   // 정지-UI 마스크 (미사용 시 더미)
        var useUIMask: Float = uiMaskTex != nil ? 1 : 0
        enc3.setBytes(&useUIMask, length: MemoryLayout<Float>.size, index: 0)
        dispatch(enc3, width: output.width, height: output.height, pso: compositePSO)
        enc3.endEncoding()

        pairCount += 1
        if pairCount <= 3 || pairCount % 600 == 0 {
            DiagnosticLog.shared.log("[AppleFI] pair #\(pairCount) encoded (\(stableB.width)x\(stableB.height), scaler=\(spatialScaler != nil ? "MetalFX" : "bilinear"), composite=masked, reuseA=\(reuseA))")
        }

        // 장면 전환 판정: CB 완료 후 히스토그램 교집합 계산 (shared 버퍼 CPU 읽기)
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
        // 전체 인코딩 성공 시에만 갱신 — 중도 실패면 불일치로 남아 다음 쌍이 전체 경로 폴백
        lastEncodedTsB = tsB
        lastStableBID = ObjectIdentifier(stableB as AnyObject)
        return PairEncodeResult(frames: [(t: 0.5, texture: output, stamp: stamp)], sceneCutEvaluator: evaluator)
    }

    private func encodeDownscale(_ enc: any MTLComputeCommandEncoder, source: any MTLTexture, target: PlanarBuffer) {
        enc.setTexture(source, index: 0)
        enc.setTexture(target.lumaTexture, index: 1)
        enc.setTexture(target.chromaTexture, index: 2)
        // 그리드 = 크로마 크기 (스레드당 루마 2x2 + 크로마 1)
        if let downscalePSO {
            dispatch(enc, width: fiWidth / 2, height: fiHeight / 2, pso: downscalePSO)
        }
    }

    private func dispatch(_ enc: any MTLComputeCommandEncoder, width: Int, height: Int, pso: any MTLComputePipelineState) {
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(
            width: (width + tg.width - 1) / tg.width,
            height: (height + tg.height - 1) / tg.height,
            depth: 1
        )
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    // MARK: - Output pool / MetalFX

    private func ensureOutputPool(width: Int, height: Int) {
        guard let device else { return }
        guard width != outputWidth || height != outputHeight || outputPool.isEmpty else { return }

        outputWidth = width
        outputHeight = height
        outputPool = []
        statsBuffers = []
        outputIndex = 0

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
        desc.storageMode = .private
        // 링 크기 = 타임라인 엔트리 상한(12). 4로는 60fps→120Hz에서 표시 대기 중인 보간
        // 출력(지연버퍼 ~6프레임)이 한 바퀴 돌아 compositeMasked가 아직 안 나간 슬롯을
        // 덮어써 프레임 순서 교란/티어링 (RIFE outputPool 8→16과 동일 클래스). statsBuffers도
        // 같은 slotIndex를 공유하므로 함께 확장된다.
        for _ in 0..<12 {
            if let tex = device.makeTexture(descriptor: desc),
               let buf = device.makeBuffer(length: 64 * 4, options: .storageModeShared) {
                outputPool.append(tex)
                statsBuffers.append(buf)
            }
        }
        slotStamps = [UInt64](repeating: 0, count: outputPool.count)   // 슬롯 세대 리셋
        upscaledTmp = device.makeTexture(descriptor: desc)

        // MetalFX spatial scaler (출력 크기 의존) — 실패 시 bilinear 폴백
        spatialScaler = nil
        if MTLFXSpatialScalerDescriptor.supportsDevice(device) {
            let sd = MTLFXSpatialScalerDescriptor()
            sd.inputWidth = fiWidth
            sd.inputHeight = fiHeight
            sd.outputWidth = width
            sd.outputHeight = height
            sd.colorTextureFormat = .bgra8Unorm
            sd.outputTextureFormat = .bgra8Unorm
            sd.colorProcessingMode = .perceptual
            spatialScaler = sd.makeSpatialScaler(device: device)
        }
        DiagnosticLog.shared.log("[AppleFI] output pool \(width)x\(height) x\(outputPool.count), MetalFX=\(spatialScaler != nil)")
    }

    // MARK: - Planar buffer

    private static func makePlanarBuffer(width: Int, height: Int, device: any MTLDevice) throws -> PlanarBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pb: CVPixelBuffer?
        let st = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        guard st == kCVReturnSuccess, let pb,
              let surface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() else {
            throw InterpolationError.textureAllocationFailed
        }

        let lumaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        lumaDesc.usage = [.shaderRead, .shaderWrite]
        lumaDesc.storageMode = .shared

        let chromaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg8Unorm, width: width / 2, height: height / 2, mipmapped: false)
        chromaDesc.usage = [.shaderRead, .shaderWrite]
        chromaDesc.storageMode = .shared

        guard let luma = device.makeTexture(descriptor: lumaDesc, iosurface: surface, plane: 0),
              let chroma = device.makeTexture(descriptor: chromaDesc, iosurface: surface, plane: 1) else {
            throw InterpolationError.textureAllocationFailed
        }
        return PlanarBuffer(pixelBuffer: pb, lumaTexture: luma, chromaTexture: chroma)
    }

    // MARK: - Reset / Shutdown

    /// 표시 직전 검증 — stamp가 아직 어느 슬롯의 현 세대면 유효(안 덮임). 렌더 스레드 전용.
    public func isFrameLive(_ stamp: UInt64) -> Bool { slotStamps.contains(stamp) }

    public func reset() {
        // 타임스탬프 연속성만 관리 — 다음 encodePair에서 역행 감지 시 세션 재시작
        lastVTTimestamp = .invalid
        lastEncodedTsB = -1   // 연속성 단절 — 다음 쌍은 prevBuf 재사용 없이 전체 다운스케일
        lastStableBID = nil
    }

    public func shutdown() {
        if sessionActive {
            processor?.endSession()
            sessionActive = false
        }
        processor = nil
        outputPool = []
        statsBuffers = []
        prevBuf = nil; srcBuf = nil; dstBuf = nil
        rgb720 = nil
        maskTex = nil
        upscaledTmp = nil
        spatialScaler = nil
    }

    // MARK: - Shaders

    /// BT.709 video-range CSC — bgraTo420v와 p420vToBGRA*는 정확한 역행렬 쌍
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    constant float3 kYCoeff = float3(0.2126, 0.7152, 0.0722);

    inline float3 rgbToYCbCr709(float3 rgb) {
        float y = dot(rgb, kYCoeff);
        float cb = (rgb.b - y) / 1.8556;
        float cr = (rgb.r - y) / 1.5748;
        // video range 인코딩
        return float3(
            (16.0 + 219.0 * y) / 255.0,
            (128.0 + 224.0 * cb) / 255.0,
            (128.0 + 224.0 * cr) / 255.0
        );
    }

    inline float3 yCbCr709ToRGB(float y8, float cb8, float cr8) {
        float y = (y8 * 255.0 - 16.0) / 219.0;
        float cb = (cb8 * 255.0 - 128.0) / 224.0;
        float cr = (cr8 * 255.0 - 128.0) / 224.0;
        float r = y + 1.5748 * cr;
        float b = y + 1.8556 * cb;
        float g = (y - kYCoeff.r * r - kYCoeff.b * b) / kYCoeff.g;
        return saturate(float3(r, g, b));
    }

    // 다운스케일 + CSC. 그리드 = 크로마 크기(W/2 x H/2), 스레드당 루마 2x2 픽셀 처리.
    // 4탭 박스 필터 (3x 데시메이션 앨리어싱 완화)
    kernel void bgraTo420v(
        texture2d<float, access::sample> src [[texture(0)]],
        texture2d<float, access::write> lumaOut [[texture(1)]],
        texture2d<float, access::write> chromaOut [[texture(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint lw = lumaOut.get_width();
        uint lh = lumaOut.get_height();
        if (gid.x * 2 >= lw || gid.y * 2 >= lh) return;

        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 dstSize = float2(lw, lh);
        float2 boxOff = 0.25 / dstSize;

        float3 rgb[4];
        for (uint i = 0; i < 4; i++) {
            uint2 lp = uint2(gid.x * 2 + (i & 1), gid.y * 2 + (i >> 1));
            float2 uv = (float2(lp) + 0.5) / dstSize;
            float3 c = src.sample(s, uv + float2(-boxOff.x, -boxOff.y)).rgb
                     + src.sample(s, uv + float2( boxOff.x, -boxOff.y)).rgb
                     + src.sample(s, uv + float2(-boxOff.x,  boxOff.y)).rgb
                     + src.sample(s, uv + float2( boxOff.x,  boxOff.y)).rgb;
            rgb[i] = c * 0.25;
            float3 ycc = rgbToYCbCr709(rgb[i]);
            lumaOut.write(float4(ycc.x), lp);
        }
        float3 avg = (rgb[0] + rgb[1] + rgb[2] + rgb[3]) * 0.25;
        float3 yccAvg = rgbToYCbCr709(avg);
        chromaOut.write(float4(yccAvg.y, yccAvg.z, 0, 0), gid);
    }

    // 모션 마스크 + 루마 히스토그램 (720p 그리드).
    // mask: |yA-yB| 기반 0..1 (0=정적 → 합성 시 원본 사용).
    // hist: yA 32빈 + yB 32빈 (4x4 서브샘플 — 장면 전환 판정용, 원자 연산 부하 1/16)
    kernel void maskHist(
        texture2d<float, access::read> yA [[texture(0)]],
        texture2d<float, access::read> yB [[texture(1)]],
        texture2d<float, access::write> mask [[texture(2)]],
        device atomic_uint* hist [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = mask.get_width();
        uint h = mask.get_height();
        if (gid.x >= w || gid.y >= h) return;
        float a = yA.read(gid).r;
        float b = yB.read(gid).r;
        float d = fabs(a - b);
        // 압축 노이즈(~0.01)는 정적으로, 실제 모션은 빠르게 1로
        float m = smoothstep(0.013, 0.05, d);
        mask.write(float4(m), gid);

        if ((gid.x & 3) == 0 && (gid.y & 3) == 0) {
            uint binA = uint(clamp(a, 0.0, 0.999) * 32.0);
            uint binB = uint(clamp(b, 0.0, 0.999) * 32.0);
            atomic_fetch_add_explicit(&hist[binA], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&hist[32u + binB], 1u, memory_order_relaxed);
        }
    }

    // 420v → BGRA (동일 크기, MetalFX 입력용)
    kernel void p420vToBGRA(
        texture2d<float, access::read> luma [[texture(0)]],
        texture2d<float, access::sample> chroma [[texture(1)]],
        texture2d<float, access::write> dst [[texture(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width();
        uint h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float y = luma.read(gid).r;
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float2 cc = chroma.sample(s, uv).rg;
        float3 rgb = yCbCr709ToRGB(y, cc.x, cc.y);
        dst.write(float4(rgb, 1.0), gid);
    }

    // 420v → BGRA 임의 크기 (bilinear 업스케일 폴백)
    kernel void p420vToBGRAScaled(
        texture2d<float, access::sample> luma [[texture(0)]],
        texture2d<float, access::sample> chroma [[texture(1)]],
        texture2d<float, access::write> dst [[texture(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width();
        uint h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float y = luma.sample(s, uv).r;
        float2 cc = chroma.sample(s, uv).rg;
        float3 rgb = yCbCr709ToRGB(y, cc.x, cc.y);
        dst.write(float4(rgb, 1.0), gid);
    }

    // 모션 마스크 합성: out = mix(원본B, 업스케일된 보간, mask)
    // mask는 720p — 5탭 max로 살짝 딜레이션 (모션 경계 헤일로 방지)
    kernel void compositeMasked(
        texture2d<float, access::read> srcB [[texture(0)]],
        texture2d<float, access::read> interpUp [[texture(1)]],
        texture2d<float, access::sample> mask [[texture(2)]],
        texture2d<float, access::write> dst [[texture(3)]],
        texture2d<float, access::sample> uiMask [[texture(4)]],
        constant float& useUIMask [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint w = dst.get_width();
        uint h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = (float2(gid) + 0.5) / float2(w, h);
        float2 o = 1.5 / float2(mask.get_width(), mask.get_height());
        float m = mask.sample(s, uv).r;
        m = max(m, mask.sample(s, uv + float2( o.x, 0)).r);
        m = max(m, mask.sample(s, uv + float2(-o.x, 0)).r);
        m = max(m, mask.sample(s, uv + float2(0,  o.y)).r);
        m = max(m, mask.sample(s, uv + float2(0, -o.y)).r);
        // 시간축 정지-UI: 모션 마스크를 낮춰 UI를 소스B(정지)로 고정
        if (useUIMask > 0.5) m *= (1.0 - clamp(uiMask.sample(s, uv).r, 0.0, 1.0));
        float3 base = srcB.read(gid).rgb;
        float3 interp = interpUp.read(gid).rgb;
        dst.write(float4(mix(base, interp, m), 1.0), gid);
    }
    """
}
