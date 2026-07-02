@preconcurrency import Metal
import VideoToolbox
import CoreVideo
import CoreMedia
import Monitoring
import os

/// Apple VTFrameRateConversion 기반 프레임 보간 엔진
/// macOS 26.4+ / GPU 가속 ML 모델 / 고품질 보간
///
/// 고해상도 입력(>1080p)은 자동으로 절반 해상도로 다운스케일 후 VT 처리.
/// VT 출력은 원래 해상도로 업스케일하여 반환.
/// 벤치마크: 1080p=4ms, 1440p=15ms, 4K=125ms → 다운스케일 필수
public final class VTInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "VideoToolbox (Apple FRC)"
    public var isAvailable: Bool {
        !VTFrameRateConversionConfiguration.supportedRevisions.isEmpty
    }

    private var device: (any MTLDevice)?
    private var commandQueue: (any MTLCommandQueue)?
    private var texturePool: TexturePool?
    private var processor: VTFrameProcessor?

    // VT 세션 해상도 (다운스케일된 크기)
    private var vtWidth: Int = 0
    private var vtHeight: Int = 0
    // 원본 해상도 (입력 크기)
    private var srcWidth: Int = 0
    private var srcHeight: Int = 0

    private var sessionStarted = false
    private let logger = Logger(subsystem: "com.macfg", category: "VTInterpolator")
    private var interpCount: Int = 0

    /// 다운스케일 최대 해상도 (긴 변 기준)
    private let maxProcessDimension = 1920

    // IOSurface-backed CVPixelBuffer 풀 (VT 해상도)
    private var pixelBufferPool: CVPixelBufferPool?

    // IOSurface-backed (VT 해상도) — 다운스케일된 입력
    private var ioTexA: (any MTLTexture)?
    private var ioTexB: (any MTLTexture)?
    private var ioPbA: CVPixelBuffer?
    private var ioPbB: CVPixelBuffer?

    // Metal MPSImageBilinearScale 대용 — compute shader로 bilinear scale
    private var downscalePSO: (any MTLComputePipelineState)?
    private var upscalePSO: (any MTLComputePipelineState)?

    // 다운스케일 중간 텍스처 (IOSurface-backed .shared)
    private var downA: (any MTLTexture)?
    private var downB: (any MTLTexture)?

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        self.texturePool = TexturePool(device: device)

        // Bilinear scale compute shader
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct ScaleParams {
            uint2 srcSize;
            uint2 dstSize;
        };

        kernel void bilinearScale(
            texture2d<float, access::sample> src [[texture(0)]],
            texture2d<float, access::write> dst [[texture(1)]],
            constant ScaleParams& params [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            if (gid.x >= params.dstSize.x || gid.y >= params.dstSize.y) return;

            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 uv = (float2(gid) + 0.5) / float2(params.dstSize);
            float4 color = src.sample(s, uv);
            color.a = 1.0;
            dst.write(color, gid);
        }
        """;

        let library = try await device.makeLibrary(source: shaderSource, options: nil)
        guard let fn = library.makeFunction(name: "bilinearScale") else {
            throw InterpolationError.notPrepared
        }
        let pso = try await device.makeComputePipelineState(function: fn)
        self.downscalePSO = pso
        self.upscalePSO = pso  // 같은 셰이더로 다운/업 모두 처리

        let revisions = VTFrameRateConversionConfiguration.supportedRevisions
        DiagnosticLog.shared.log("[VT-FRC] prepare OK, revisions=\(revisions), maxDim=\(maxProcessDimension)")
    }

    /// 다운스케일 해상도 계산 (긴 변 기준, 짝수 보장)
    private func computeVTSize(srcW: Int, srcH: Int) -> (Int, Int) {
        let maxDim = max(srcW, srcH)
        if maxDim <= maxProcessDimension {
            return (srcW, srcH)  // 스케일링 불필요
        }
        let scale = Float(maxProcessDimension) / Float(maxDim)
        var w = Int(Float(srcW) * scale)
        var h = Int(Float(srcH) * scale)
        // 짝수로 정렬 (VT 요구사항)
        w = w & ~1
        h = h & ~1
        return (w, h)
    }

    /// VT 세션 초기화 (해상도 변경 시)
    private func ensureSession(width: Int, height: Int) throws {
        let (vw, vh) = computeVTSize(srcW: width, srcH: height)
        guard vw != vtWidth || vh != vtHeight || !sessionStarted else { return }

        if sessionStarted {
            processor?.endSession()
            sessionStarted = false
        }

        srcWidth = width
        srcHeight = height
        vtWidth = vw
        vtHeight = vh

        DiagnosticLog.shared.log("[VT-FRC] Creating config: src=\(width)x\(height) → vt=\(vw)x\(vh)")
        guard let config = VTFrameRateConversionConfiguration(
            frameWidth: vw,
            frameHeight: vh,
            usePrecomputedFlow: false,
            qualityPrioritization: .quality,
            revision: .revision1
        ) else {
            DiagnosticLog.shared.log("[VT-FRC] Config creation FAILED")
            throw InterpolationError.notPrepared
        }
        DiagnosticLog.shared.log("[VT-FRC] Config OK")

        let proc = VTFrameProcessor()
        do {
            try proc.startSession(configuration: config)
            DiagnosticLog.shared.log("[VT-FRC] startSession OK")
        } catch {
            DiagnosticLog.shared.log("[VT-FRC] startSession FAILED: \(error)")
            throw error
        }
        self.processor = proc
        self.sessionStarted = true

        // VT 해상도 Pixel buffer pool
        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 4]
        let pbAttrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: vw,
            kCVPixelBufferHeightKey as String: vh,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, pbAttrs as CFDictionary, &pool)
        guard status == kCVReturnSuccess, let pool else { throw InterpolationError.textureAllocationFailed }
        self.pixelBufferPool = pool

        guard let device else { throw InterpolationError.notPrepared }

        // VT 입력용 IOSurface textures (VT 해상도)
        (ioPbA, ioTexA) = try createIOSurfaceTexture(width: vw, height: vh, device: device)
        (ioPbB, ioTexB) = try createIOSurfaceTexture(width: vw, height: vh, device: device)

        // 다운스케일 중간 텍스처 (.shared — GPU 쓰기 + IOSurface 읽기 호환)
        if vw != width || vh != height {
            downA = makeSharedTexture(width: vw, height: vh, device: device)
            downB = makeSharedTexture(width: vw, height: vh, device: device)
        } else {
            downA = nil
            downB = nil
        }

        logger.info("VT-FRC session: src=\(width)x\(height) → vt=\(vw)x\(vh)")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        // VTInterpolator는 VTFrameRateConversion 콜백 + await가 필요하므로
        // 동기식 파이프라인에서는 사용할 수 없음. (실측 43-125ms로 실용 불가.)
        throw InterpolationError.opticalFlowFailed("VTInterpolator requires async and is not supported in synchronous pipeline")
    }

    // MARK: - Helpers

    private func acquireBuffer() throws -> CVPixelBuffer {
        guard let pixelBufferPool else { throw InterpolationError.notPrepared }
        var pb: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pixelBufferPool, &pb)
        guard status == kCVReturnSuccess, let pb else { throw InterpolationError.textureAllocationFailed }
        return pb
    }

    private func createIOSurfaceTexture(width: Int, height: Int, device: any MTLDevice) throws -> (CVPixelBuffer, any MTLTexture) {
        let attrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let pb else { throw InterpolationError.textureAllocationFailed }

        guard let ioSurface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() else {
            throw InterpolationError.textureAllocationFailed
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc, iosurface: ioSurface, plane: 0) else {
            throw InterpolationError.textureAllocationFailed
        }
        return (pb, texture)
    }

    private func createTextureFromPixelBuffer(_ pixelBuffer: CVPixelBuffer, device: any MTLDevice) throws -> any MTLTexture {
        guard let ioSurface = CVPixelBufferGetIOSurface(pixelBuffer)?.takeUnretainedValue() else {
            throw InterpolationError.textureAllocationFailed
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: desc, iosurface: ioSurface, plane: 0) else {
            throw InterpolationError.textureAllocationFailed
        }
        return texture
    }

    /// .shared 스토리지 텍스처 생성 (compute write + blit source 호환)
    private func makeSharedTexture(width: Int, height: Int, device: any MTLDevice) -> (any MTLTexture)? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .shared
        return device.makeTexture(descriptor: desc)
    }

    public func shutdown() {
        if sessionStarted {
            processor?.endSession()
            sessionStarted = false
        }
        processor = nil
        pixelBufferPool = nil
        ioTexA = nil; ioTexB = nil
        ioPbA = nil; ioPbB = nil
        downA = nil; downB = nil
        texturePool?.drain()
        texturePool = nil
        logger.info("VTInterpolator(FRC) shut down")
    }
}
