@preconcurrency import Metal
import FramePacing
import Monitoring
import os

/// 동기식 보간 파이프라인 — DisplayLink 콜백에서 직접 호출.
///
/// Task.detached/Lock 없음. MainActor에서만 접근.
/// commandBuffer 인코딩만 수행 (~0.1ms CPU), commit은 호출자가 담당.
@MainActor
public final class InterpolationPipeline {

    private let engine: any FrameInterpolator
    private let device: any MTLDevice
    private let texturePool: TexturePool
    private let logger = Logger(subsystem: "com.macfg", category: "InterpolationPipeline")

    // 프레임 쌍 추적 (MainActor, lock 불필요)
    // previousSourceCopy는 SCK/IOSurface 재사용에 영향받지 않는 private texture여야 한다.
    private var previousSourceCopy: (any MTLTexture)?
    private var lastDistinctFingerprint: UInt64 = 0
    private var lastDistinctTimestamp: CFTimeInterval = 0
    private var duplicateSkipCount: Int = 0

    public var isEnabled: Bool = true
    public var inputFPS: Double = 30.0
    public var displayRefreshRate: Double = 120.0

    private var interpAttemptCount: Int = 0

    // MARK: - Init

    public init(engine: any FrameInterpolator, device: any MTLDevice) {
        self.engine = engine
        self.device = device
        self.texturePool = TexturePool(device: device)
    }

    public func prepare() async throws {
        try await engine.prepare(device: device)
    }

    public var engineName: String { engine.name }

    // MARK: - 동기식 보간 인코딩

    /// 보간 결과: sourceCopy는 소스 프레임의 blit 복사본 (소스 렌더에도 사용 → 텍스처 경로 통일)
    public struct InterpolationResult {
        public let sourceCopy: any MTLTexture
        public let interpTextures: [any MTLTexture]
    }

    /// 새 소스 프레임으로 프레임 쌍 갱신 + 보간 인코딩.
    /// commandBuffer에 blit + compute 커맨드를 인코딩. commit은 호출자가 담당.
    /// sourceCopy: 소스의 blit 복사본 (소스 렌더에 사용 → IOSurface 직접 렌더 대신 동일 경로)
    /// interpTextures: 보간 텍스처 배열 (GPU 미완료 상태)
    public func encodeInterpolation(
        sourceSlot: FrameSlot,
        commandBuffer: any MTLCommandBuffer,
        tValues: [Float]
    ) -> InterpolationResult? {
        guard isEnabled else { return nil }

        // 중복 프레임 감지
        if sourceSlot.contentFingerprint != 0
            && sourceSlot.contentFingerprint == lastDistinctFingerprint
            && sourceSlot.timestamp <= lastDistinctTimestamp + 0.0005 {
            duplicateSkipCount += 1
            return nil
        }
        lastDistinctFingerprint = sourceSlot.contentFingerprint
        lastDistinctTimestamp = max(lastDistinctTimestamp, sourceSlot.timestamp)

        guard let texB = sourceSlot.texture,
              !tValues.isEmpty else {
            return nil
        }

        interpAttemptCount += 1
        let attemptNum = interpAttemptCount
        let verbose = attemptNum <= 20 || attemptNum % 100 == 0

        // IOSurface blit: SCK가 재사용하기 전에 private 텍스처로 복사
        guard let copyB = encodeBlit(texB, commandBuffer: commandBuffer) else {
            return nil
        }

        // previousSourceCopy가 없으면 소스 복사본만 반환 (보간 불가)
        guard let copyA = previousSourceCopy else {
            previousSourceCopy = copyB
            return InterpolationResult(sourceCopy: copyB, interpTextures: [])
        }

        if verbose {
            DiagnosticLog.shared.log("[INTERP #\(attemptNum)] SYNC encode tValues=\(tValues) tex=\(copyA.width)x\(copyA.height) fmtA=\(copyA.pixelFormat.rawValue) fmtB=\(texB.pixelFormat.rawValue) copyFmt=\(copyB.pixelFormat.rawValue)")
        }

        // 보간 인코딩 (동기, ~0.1ms CPU)
        do {
            let results = try engine.batchInterpolate(
                frameA: copyA,
                frameB: copyB,
                tValues: tValues,
                commandBuffer: commandBuffer
            )

            if verbose {
                DiagnosticLog.shared.log("[INTERP #\(attemptNum)] SYNC encoded \(results.count) frames")
            }

            previousSourceCopy = copyB
            return InterpolationResult(sourceCopy: copyB, interpTextures: results)
        } catch {
            DiagnosticLog.shared.log("[INTERP #\(attemptNum)] ERROR: \(error)")
            previousSourceCopy = copyB
            return InterpolationResult(sourceCopy: copyB, interpTextures: [])
        }
    }

    /// 진단 정보
    public func resetAndGetDuplicateSkipCount() -> Int {
        let c = duplicateSkipCount
        duplicateSkipCount = 0
        return c
    }

    public var debugBufferState: String {
        "engine=\(engine.name) attempts=\(interpAttemptCount)"
    }

    // MARK: - Blit

    private func encodeBlit(_ source: any MTLTexture, commandBuffer: any MTLCommandBuffer) -> (any MTLTexture)? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: source.pixelFormat,
            width: source.width,
            height: source.height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .private

        guard let copy = device.makeTexture(descriptor: desc),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            return nil
        }

        blitEncoder.copy(
            from: source, sourceSlice: 0, sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
            to: copy, destinationSlice: 0, destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blitEncoder.endEncoding()

        return copy
    }

    // MARK: - Shutdown

    public func shutdown() {
        engine.shutdown()
        previousSourceCopy = nil
        lastDistinctFingerprint = 0
        lastDistinctTimestamp = 0
        texturePool.drain()
    }
}
