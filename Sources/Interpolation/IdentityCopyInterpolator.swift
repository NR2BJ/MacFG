@preconcurrency import Metal
import os

/// 정확한 복사 기반 기준 엔진.
/// Frame generation 품질을 보기 전에 source copy/render 경로가 흔들리지 않는지 확인한다.
public final class IdentityCopyInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "Identity Copy"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "IdentityCopy")

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)
        logger.info("IdentityCopyInterpolator prepared")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard let texturePool else {
            throw InterpolationError.notPrepared
        }

        guard let output = texturePool.acquire(
            width: frameB.width,
            height: frameB.height,
            pixelFormat: frameB.pixelFormat,
            usage: [.shaderRead]
        ), let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        blit.copy(
            from: frameB,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: frameB.width, height: frameB.height, depth: 1),
            to: output,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        return output
    }

    public func shutdown() {
        texturePool?.drain()
        texturePool = nil
        logger.info("IdentityCopyInterpolator shut down")
    }
}
