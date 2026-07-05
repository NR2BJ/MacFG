import Metal
import IOSurface
import IOKit
import QuartzCore
import FramePacing
import os

import CoreGraphics

// MARK: - Private API via dlsym

private typealias CGSMainConnectionIDFunc = @convention(c) () -> UInt32
private typealias CGSGetWindowSurfaceIDFunc = @convention(c) (UInt32, UInt32, UnsafeMutablePointer<UInt32>) -> CGError

nonisolated(unsafe) private let skylight: UnsafeMutableRawPointer? = dlopen("/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight", RTLD_LAZY)

private let _CGSMainConnectionID: CGSMainConnectionIDFunc? = {
    guard let skylight, let sym = dlsym(skylight, "CGSMainConnectionID") else { return nil }
    return unsafeBitCast(sym, to: CGSMainConnectionIDFunc.self)
}()

private let _CGSGetWindowSurfaceID: CGSGetWindowSurfaceIDFunc? = {
    guard let skylight, let sym = dlsym(skylight, "CGSGetWindowSurfaceID") else { return nil }
    return unsafeBitCast(sym, to: CGSGetWindowSurfaceIDFunc.self)
}()

/// IOSurface 직접 캡처 — 제로카피로 MTLTexture 획득
public final class IOSurfaceCapture: FrameSource, @unchecked Sendable {
    public let method: CaptureMethod = .ioSurface

    private let logger = Logger(subsystem: "com.macfg", category: "IOSurfaceCapture")
    private var device: (any MTLDevice)?
    private var windowID: CGWindowID = 0
    private var surface: IOSurfaceRef?
    private var currentSurfaceID: UInt32 = 0
    private var texture: (any MTLTexture)?
    private var capturing = false
    private var framesSinceValidation: Int = 0
    /// 매 N프레임마다 surface ID 재검증
    private let validationInterval: Int = 30

    public var isAvailable: Bool {
        // IOSurface 직접 캡처는 항상 시도 가능 (실패는 런타임에 판단)
        true
    }

    public init() {}

    public func startCapture(windowID: CGWindowID, device: any MTLDevice, captureRect: CGRect? = nil) async throws {
        // captureRect(영역 캡처)는 SCK 전용 — IOSurface 폴백은 전체 창만 지원(무시)
        self.device = device
        self.windowID = windowID

        // Surface ID 획득
        guard let surf = acquireSurface(windowID: windowID) else {
            logger.warning("Failed to acquire IOSurface for window \(windowID)")
            throw CaptureError.ioSurfaceUnavailable
        }

        self.surface = surf
        self.currentSurfaceID = currentSurfaceIDValue(for: windowID) ?? 0
        self.texture = makeTexture(from: surf, device: device)
        self.capturing = true
        self.framesSinceValidation = 0

        logger.info("IOSurface capture started for window \(windowID), size: \(IOSurfaceGetWidth(surf))x\(IOSurfaceGetHeight(surf))")
    }

    public func stopCapture() async {
        capturing = false
        texture = nil
        surface = nil
        logger.info("IOSurface capture stopped")
    }

    public func latestFrame() -> FrameSlot? {
        guard capturing, let device else { return nil }

        // 주기적으로 surface ID 재검증 — stale surface 방지
        framesSinceValidation += 1
        if framesSinceValidation >= validationInterval {
            framesSinceValidation = 0
            revalidateSurface()
        }

        guard let surface else { return nil }

        // IOSurface 내용은 CADisplayLink 콜백마다 자동 갱신됨
        let width = IOSurfaceGetWidth(surface)
        let height = IOSurfaceGetHeight(surface)
        let fingerprint = computeFingerprint(surface: surface, width: width, height: height)

        if let tex = texture, tex.width == width, tex.height == height {
            return FrameSlot(
                texture: tex,
                timestamp: CACurrentMediaTime(),
                width: width,
                height: height,
                contentChanged: true,
                contentFingerprint: fingerprint
            )
        }

        // 크기 변경 시 텍스처 재생성
        texture = makeTexture(from: surface, device: device)
        if let tex = texture {
            return FrameSlot(
                texture: tex,
                timestamp: CACurrentMediaTime(),
                width: tex.width,
                height: tex.height,
                contentChanged: true,
                contentFingerprint: fingerprint
            )
        }

        return nil
    }

    /// surface ID가 바뀌었는지 확인, 바뀌었으면 새로 획득
    private func revalidateSurface() {
        guard let device else { return }
        guard let mainConnID = _CGSMainConnectionID,
              let getSurfaceID = _CGSGetWindowSurfaceID else { return }

        let cid = mainConnID()
        var newSurfaceID: UInt32 = 0
        let err = getSurfaceID(cid, windowID, &newSurfaceID)
        guard err == .success, newSurfaceID != 0 else {
            // surface 접근 실패 — 창이 닫혔거나 권한 문제
            logger.warning("Surface revalidation failed for window \(self.windowID)")
            self.surface = nil
            self.texture = nil
            return
        }

        if newSurfaceID != currentSurfaceID {
            logger.info("IOSurface ID changed: \(self.currentSurfaceID) → \(newSurfaceID)")
            currentSurfaceID = newSurfaceID
            guard let newSurface = IOSurfaceLookup(newSurfaceID) else {
                self.surface = nil
                self.texture = nil
                return
            }
            self.surface = newSurface
            self.texture = makeTexture(from: newSurface, device: device)
        }
    }

    // MARK: - Private

    private func currentSurfaceIDValue(for windowID: CGWindowID) -> UInt32? {
        guard let mainConnID = _CGSMainConnectionID,
              let getSurfaceID = _CGSGetWindowSurfaceID else { return nil }
        let cid = mainConnID()
        var surfaceID: UInt32 = 0
        let err = getSurfaceID(cid, windowID, &surfaceID)
        guard err == .success, surfaceID != 0 else { return nil }
        return surfaceID
    }

    private func acquireSurface(windowID: CGWindowID) -> IOSurfaceRef? {
        guard let mainConnID = _CGSMainConnectionID,
              let getSurfaceID = _CGSGetWindowSurfaceID else {
            logger.warning("Private CGS APIs not available")
            return nil
        }
        let cid = mainConnID()
        var surfaceID: UInt32 = 0
        let err = getSurfaceID(cid, windowID, &surfaceID)
        guard err == .success, surfaceID != 0 else {
            return nil
        }
        return IOSurfaceLookup(surfaceID)
    }

    private func makeTexture(from surface: IOSurfaceRef, device: any MTLDevice) -> (any MTLTexture)? {
        let width = IOSurfaceGetWidth(surface)
        let height = IOSurfaceGetHeight(surface)
        guard width > 0, height > 0 else { return nil }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared

        return device.makeTexture(descriptor: desc, iosurface: surface, plane: 0)
    }

    private func computeFingerprint(surface: IOSurfaceRef, width: Int, height: Int) -> UInt64 {
        guard width > 0, height > 0 else { return 0 }
        let lockResult = IOSurfaceLock(surface, .readOnly, nil)
        guard lockResult == kIOReturnSuccess else { return 0 }
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let baseAddress = IOSurfaceGetBaseAddress(surface)
        let bytesPerRow = IOSurfaceGetBytesPerRow(surface)
        let ptr = baseAddress.assumingMemoryBound(to: UInt8.self)

        var hash: UInt64 = 0xcbf29ce484222325
        let cols = 16
        let rows = 9
        for row in 0..<rows {
            let y = (height * (row * 2 + 1)) / (rows * 2)
            for col in 0..<cols {
                let x = (width * (col * 2 + 1)) / (cols * 2)
                let offset = y * bytesPerRow + x * 4
                for i in 0..<3 {
                    hash ^= UInt64(ptr[offset + i] >> 2)
                    hash &*= 0x100000001b3
                }
            }
        }
        return hash
    }
}

public enum CaptureError: Error, Sendable {
    case ioSurfaceUnavailable
    case screenCaptureKitError(String)
    case noPermission
    case windowNotFound
    case notCapturing
}
