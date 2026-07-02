@preconcurrency import Metal
import os

/// MTLTexture 재사용 풀 — 프레임마다 텍스처 할당 비용을 피한다.
public final class TexturePool: @unchecked Sendable {
    private let device: any MTLDevice
    private let lock = OSAllocatedUnfairLock(initialState: [PoolEntry]())
    private let logger = Logger(subsystem: "com.macfg", category: "TexturePool")
    private let maxPoolSize = 8

    private struct PoolEntry: @unchecked Sendable {
        let texture: any MTLTexture
        let width: Int
        let height: Int
        let pixelFormat: MTLPixelFormat
    }

    public init(device: any MTLDevice) {
        self.device = device
    }

    /// 지정 크기/포맷의 텍스처를 풀에서 가져오거나 새로 생성.
    public func acquire(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) -> (any MTLTexture)? {
        // 풀에서 매칭되는 텍스처 검색
        let found: (any MTLTexture)? = lock.withLock { pool in
            if let idx = pool.firstIndex(where: {
                $0.width == width && $0.height == height && $0.pixelFormat == pixelFormat
            }) {
                let entry = pool.remove(at: idx)
                return entry.texture
            }
            return nil
        }
        if let tex = found { return tex }

        // 새 텍스처 생성
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = usage
        desc.storageMode = .private
        guard let texture = device.makeTexture(descriptor: desc) else {
            logger.error("Failed to create texture \(width)x\(height)")
            return nil
        }
        return texture
    }

    /// 텍스처를 풀에 반환.
    public func release(_ texture: any MTLTexture) {
        lock.withLock { pool in
            guard pool.count < maxPoolSize else { return }
            pool.append(PoolEntry(
                texture: texture,
                width: texture.width,
                height: texture.height,
                pixelFormat: texture.pixelFormat
            ))
        }
    }

    /// 풀 비우기 (메모리 해제)
    public func drain() {
        lock.withLock { pool in
            pool.removeAll()
        }
    }
}
