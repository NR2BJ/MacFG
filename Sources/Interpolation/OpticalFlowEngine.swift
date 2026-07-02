@preconcurrency import Metal
import Vision
import CoreVideo
import CoreImage
import os

/// Apple Vision Optical Flow 기반 프레임 보간 엔진
public final class OpticalFlowEngine: FrameInterpolator, @unchecked Sendable {
    public let name = "Optical Flow (Vision)"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var warpRenderer: WarpRenderer?
    private var sceneChangeDetector: SceneChangeDetector?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "OpticalFlowEngine")

    // Vision 요청 핸들러 (재사용)
    private var requestHandler: VNSequenceRequestHandler?
    private var ciContext: CIContext?
    private var interpCount: Int = 0

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)
        self.requestHandler = VNSequenceRequestHandler()
        self.ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])

        let renderer = WarpRenderer(device: device)
        try await renderer.prepare()
        self.warpRenderer = renderer

        let detector = SceneChangeDetector(device: device)
        try await detector.prepare()
        self.sceneChangeDetector = detector

        logger.info("OpticalFlowEngine prepared")
    }

    /// Optical Flow 계산용 최대 해상도 (이 이하로 다운스케일)
    /// 640px → ~20-30ms (60fps 소스 간격 17ms — 대부분 hit 가능)
    /// 960px → ~60-80ms (60fps에서 hit율 17%밖에 안됨)
    private static let maxFlowDimension: Int = 640

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard let device, let warpRenderer, let texturePool, let requestHandler, let sceneChangeDetector else {
            throw InterpolationError.notPrepared
        }

        // 0. MAD readback 콜백 — 반드시 scene change 판정 전에 등록
        // ⚠️ scene change return nil 시에도 readResult 호출 보장 → lastMAD 갱신 필수
        let detector = sceneChangeDetector
        let pixelCount = frameA.width * frameA.height
        commandBuffer.addCompletedHandler { _ in
            detector.readResult(pixelCount: pixelCount)
        }

        // 1. 장면 전환 감지 (이전 계산 결과 사용, 새 계산은 인코딩)
        if sceneChangeDetector.isSceneChange(frameA: frameA, frameB: frameB, commandBuffer: commandBuffer) {
            logger.debug("Scene change detected, skipping interpolation")
            return nil
        }

        // 2. MTLTexture → CVPixelBuffer 변환 (flow용 다운스케일 포함)
        let flowScale = Self.computeFlowScale(width: frameA.width, height: frameA.height)
        let startTime = CFAbsoluteTimeGetCurrent()
        let pixelBufferA = try createPixelBuffer(from: frameA, device: device, scale: flowScale)
        let pixelBufferB = try createPixelBuffer(from: frameB, device: device, scale: flowScale)
        let cvtTime = CFAbsoluteTimeGetCurrent()

        // 3. Optical Flow 계산 (A → B) — 다운스케일된 해상도에서 수행
        let flowRequest = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: pixelBufferB)
        flowRequest.computationAccuracy = .low
        flowRequest.outputPixelFormat = kCVPixelFormatType_TwoComponent16Half

        try requestHandler.perform([flowRequest], on: pixelBufferA)
        let flowTime = CFAbsoluteTimeGetCurrent()

        guard let flowObservation = flowRequest.results?.first as? VNPixelBufferObservation else {
            throw InterpolationError.opticalFlowFailed("No flow observation result")
        }

        // 4. Flow CVPixelBuffer → MTLTexture
        // Vision flow: Y-flip 보정 필요 (CIImage bottom-left ↔ Metal top-left)
        let flowTexture = try createFlowTexture(from: flowObservation.pixelBuffer, device: device)

        // 타이밍 로그 (처음 20회 + 100회마다)
        interpCount += 1
        if interpCount <= 20 || interpCount % 100 == 0 {
            let cvtMs = (cvtTime - startTime) * 1000
            let flowMs = (flowTime - cvtTime) * 1000
            let flowW = CVPixelBufferGetWidth(flowObservation.pixelBuffer)
            let flowH = CVPixelBufferGetHeight(flowObservation.pixelBuffer)
            logger.info("[VNFlow #\(self.interpCount)] cvt=\(String(format: "%.1f", cvtMs))ms flow=\(String(format: "%.1f", flowMs))ms scale=\(flowScale) flowTex=\(flowW)x\(flowH)")
        }

        // 5. 출력 텍스처 할당 (원본 해상도)
        guard let output = texturePool.acquire(
            width: frameB.width,
            height: frameB.height,
            pixelFormat: .bgra8Unorm,
            usage: [.shaderRead, .shaderWrite]
        ) else {
            throw InterpolationError.textureAllocationFailed
        }

        // 6. 워프 셰이더 인코딩 (원본 해상도 프레임 + 스케일된 flow)
        // flowScale을 전달하여 워프 셰이더가 flow 벡터를 원본 좌표계로 보정
        try warpRenderer.encode(
            frameA: frameA,
            frameB: frameB,
            flowTexture: flowTexture,
            output: output,
            t: t,
            flowScale: flowScale,
            commandBuffer: commandBuffer
        )

        return output
    }

    /// 원본 대비 flow 계산 스케일 비율 (1.0이면 다운스케일 없음)
    private static func computeFlowScale(width: Int, height: Int) -> Float {
        let maxDim = max(width, height)
        if maxDim <= maxFlowDimension { return 1.0 }
        return Float(maxFlowDimension) / Float(maxDim)
    }

    public func shutdown() {
        warpRenderer = nil
        sceneChangeDetector = nil
        texturePool?.drain()
        texturePool = nil
        requestHandler = nil
        ciContext = nil
        logger.info("OpticalFlowEngine shut down")
    }

    // MARK: - Texture ↔ CVPixelBuffer 변환

    /// MTLTexture → CVPixelBuffer (scale < 1.0이면 다운스케일)
    private func createPixelBuffer(from texture: any MTLTexture, device: any MTLDevice, scale: Float = 1.0) throws -> CVPixelBuffer {
        let targetWidth = max(1, Int(Float(texture.width) * scale))
        let targetHeight = max(1, Int(Float(texture.height) * scale))

        // 스케일 1.0이고 IOSurface 백킹이면 제로카피
        if scale >= 0.999, let ioSurface = texture.iosurface {
            var unmanagedPB: Unmanaged<CVPixelBuffer>?
            let status = CVPixelBufferCreateWithIOSurface(
                kCFAllocatorDefault,
                ioSurface,
                nil,
                &unmanagedPB
            )
            if status == kCVReturnSuccess, let pb = unmanagedPB?.takeRetainedValue() {
                return pb
            }
        }

        // CVPixelBuffer 생성 (다운스케일 해상도)
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth, targetHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pb = pixelBuffer else {
            throw InterpolationError.opticalFlowFailed("Failed to create CVPixelBuffer")
        }

        if scale >= 0.999 {
            // 원본 해상도: 직접 복사
            CVPixelBufferLockBaseAddress(pb, [])
            defer { CVPixelBufferUnlockBaseAddress(pb, []) }
            if let baseAddress = CVPixelBufferGetBaseAddress(pb) {
                let bytesPerRow = targetWidth * 4
                let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                       size: MTLSize(width: targetWidth, height: targetHeight, depth: 1))
                texture.getBytes(baseAddress, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
            }
        } else {
            // 다운스케일: CIImage로 리사이즈
            try downscaleTextureToCVPixelBuffer(texture: texture, into: pb, targetWidth: targetWidth, targetHeight: targetHeight)
        }

        return pb
    }

    /// Metal 텍스처를 CIImage 기반으로 다운스케일하여 CVPixelBuffer에 기록
    private func downscaleTextureToCVPixelBuffer(texture: any MTLTexture, into pixelBuffer: CVPixelBuffer, targetWidth: Int, targetHeight: Int) throws {
        guard let device else { throw InterpolationError.notPrepared }

        // MTLTexture → CIImage
        let ciImage = CIImage(mtlTexture: texture, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])!
            .oriented(.downMirrored) // Metal 좌표계 → CoreImage 좌표계

        let scaleX = CGFloat(targetWidth) / ciImage.extent.width
        let scaleY = CGFloat(targetHeight) / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        guard let context = ciContext else { throw InterpolationError.notPrepared }
        context.render(scaled, to: pixelBuffer)
    }

    /// Flow CVPixelBuffer (float16 × 2ch) → MTLTexture
    /// Vision flow는 CIImage 좌표계(bottom-left)에서 계산됨.
    /// Metal은 top-left 좌표계이므로:
    /// - flow의 행 순서를 뒤집고 (Y축 미러)
    /// - flow Y 성분(dy)을 부정 (방향 반전)
    private func createFlowTexture(from pixelBuffer: CVPixelBuffer, device: any MTLDevice) throws -> any MTLTexture {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: desc) else {
            throw InterpolationError.textureAllocationFailed
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw InterpolationError.opticalFlowFailed("Failed to get flow pixel buffer base address")
        }

        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dstBytesPerRow = width * 4  // rg16Float = 2 × 2 bytes = 4 bytes per pixel

        // Y-flip + dy 부정: Vision (bottom-left) → Metal (top-left)
        // 각 행을 뒤집어서 복사하고, half float Y 성분을 XOR 0x8000 (부호 비트 반전)
        var flippedData = [UInt8](repeating: 0, count: dstBytesPerRow * height)
        for y in 0..<height {
            let srcRow = baseAddress.advanced(by: (height - 1 - y) * srcBytesPerRow)
            let dstOffset = y * dstBytesPerRow
            // 행 복사
            flippedData.withUnsafeMutableBytes { dstPtr in
                let dst = dstPtr.baseAddress!.advanced(by: dstOffset)
                memcpy(dst, srcRow, dstBytesPerRow)
            }
            // dy (각 픽셀의 두번째 half) 부호 반전
            flippedData.withUnsafeMutableBytes { dstPtr in
                let pixels = dstPtr.baseAddress!.advanced(by: dstOffset)
                    .assumingMemoryBound(to: UInt16.self)
                for x in 0..<width {
                    // rg16Float: [dx_half, dy_half] per pixel
                    // dy는 offset 1 (두 번째 UInt16)
                    pixels[x * 2 + 1] ^= 0x8000  // 부호 비트 반전
                }
            }
        }

        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))
        flippedData.withUnsafeBytes { ptr in
            texture.replace(region: region, mipmapLevel: 0, withBytes: ptr.baseAddress!, bytesPerRow: dstBytesPerRow)
        }

        return texture
    }
}
