import Metal
import MetalFX
import os

/// MetalFX Spatial 업스케일러 래퍼 — 저해상도 프레임을 출력 해상도로 선명하게.
/// 입력/출력 크기가 바뀌면 스케일러와 출력 텍스처를 재생성한다 (뷰어 리사이즈 대응).
/// 비지원 기기/실패 시 nil을 반환해 호출자가 기존 이중선형 경로로 폴백하게 한다.
final class MetalFXUpscaler {
    private let device: any MTLDevice
    private let logger = Logger(subsystem: "com.macfg", category: "MetalFXUpscaler")

    private var scaler: (any MTLFXSpatialScaler)?
    private var outputTex: (any MTLTexture)?
    private var inW = 0, inH = 0, outW = 0, outH = 0
    private var fmt: MTLPixelFormat = .bgra8Unorm
    private var unsupported = false

    init(device: any MTLDevice) { self.device = device }

    /// input을 outWidth×outHeight로 업스케일한 (재사용) 텍스처 반환. 실패 시 nil.
    /// commandBuffer에 스케일 패스를 인코딩한다 (호출자가 이후 commit).
    func upscale(_ input: any MTLTexture,
                 outWidth: Int, outHeight: Int,
                 commandBuffer: any MTLCommandBuffer) -> (any MTLTexture)? {
        guard !unsupported, outWidth > 0, outHeight > 0 else { return nil }

        if scaler == nil || inW != input.width || inH != input.height
            || outW != outWidth || outH != outHeight || fmt != input.pixelFormat {
            guard rebuild(inputW: input.width, inputH: input.height,
                          outputW: outWidth, outputH: outHeight,
                          format: input.pixelFormat) else {
                unsupported = true
                logger.error("MetalFX spatial scaler unavailable → bilinear fallback")
                return nil
            }
        }

        guard let scaler, let outputTex else { return nil }
        scaler.colorTexture = input
        scaler.outputTexture = outputTex
        scaler.encode(commandBuffer: commandBuffer)
        return outputTex
    }

    private func rebuild(inputW: Int, inputH: Int, outputW: Int, outputH: Int, format: MTLPixelFormat) -> Bool {
        let desc = MTLFXSpatialScalerDescriptor()
        desc.inputWidth = inputW
        desc.inputHeight = inputH
        desc.outputWidth = outputW
        desc.outputHeight = outputH
        desc.colorTextureFormat = format
        desc.outputTextureFormat = format
        // 표시용 8bit 컬러는 감마 공간 → perceptual (linear은 밝기 왜곡)
        desc.colorProcessingMode = .perceptual

        guard let s = desc.makeSpatialScaler(device: device) else { return false }

        let td = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: outputW, height: outputH, mipmapped: false)
        td.usage = [.shaderRead, .shaderWrite, .renderTarget]
        td.storageMode = .private
        guard let out = device.makeTexture(descriptor: td) else { return false }

        scaler = s
        outputTex = out
        inW = inputW; inH = inputH; outW = outputW; outH = outputH; fmt = format
        logger.info("MetalFX scaler \(inputW)x\(inputH) → \(outputW)x\(outputH)")
        return true
    }
}
