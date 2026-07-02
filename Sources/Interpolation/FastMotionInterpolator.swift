@preconcurrency import Metal
import Monitoring
import os

/// 초경량 모션 보상 프레임 보간 엔진 (타깃: <8ms @4K)
///
/// 전략: 1/16 해상도 블록 매칭 → 양방향 워프 + 블렌드
/// - Phase 1: 다운스케일 (bilinear, 1/16) → ~0.2ms
/// - Phase 2: 블록 매칭 (8x8 블록, ±16px 탐색) → ~1ms
/// - Phase 3: 양방향 모션 워프 + 블렌드 (원본 해상도) → ~3ms
/// - 총 예상: ~5ms @4K (3840x2160)
///
/// 지터 방지 핵심: 모든 단계가 단일 commandBuffer에 인코딩 → GPU 일괄 실행
public final class FastMotionInterpolator: FrameInterpolator, @unchecked Sendable {
    public let name = "FastMotion (<8ms)"
    public var isAvailable: Bool { true }

    private var device: (any MTLDevice)?
    private var texturePool: TexturePool?
    private let logger = Logger(subsystem: "com.macfg", category: "FastMotion")

    // Compute pipelines
    private var downsamplePSO: (any MTLComputePipelineState)?
    private var blockMatchPSO: (any MTLComputePipelineState)?
    private var motionWarpPSO: (any MTLComputePipelineState)?

    // 재사용 텍스처 (해상도별 lazy init)
    private var grayA: (any MTLTexture)?      // 1/16 grayscale A
    private var grayB: (any MTLTexture)?      // 1/16 grayscale B
    private var motionTex: (any MTLTexture)?  // 1/16 motion vectors (rg16Float)
    private var cachedWidth: Int = 0
    private var cachedHeight: Int = 0

    private var interpCount: Int = 0
    private var mvDiagReadback: (any MTLTexture)?  // CPU-readable MV copy for diagnostics
    private var mvDiagGridW: Int = 0
    private var mvDiagGridH: Int = 0

    // 설정 — 4K@8ms 버짓 최적화
    private let downscaleFactor: Int = 8      // 1/8 해상도 (16은 60fps에서 모션 감지 불가)
    private let blockSize: Int = 16           // 블록 크기 (↑ = 그리드 축소 = 빠름)
    private let searchRadius: Int = 8         // 탐색 범위 (↓ = 빠름, 스크린캡처는 느린 모션)

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 1: 다운스케일 + 그레이스케일 (1/16)
    // ★ 4×4 bilinear 샘플 그리드 — 16회 sample()로 64픽셀 커버
    //   (1회 sample = 에일리어싱, 256회 read = 너무 느림)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    struct DownsampleParams {
        uint2 dstSize;
        float2 srcSize;
    };

    kernel void downsampleGray(
        texture2d<float, access::sample> src [[texture(0)]],
        texture2d<float, access::write>  dst [[texture(1)]],
        constant DownsampleParams& p        [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.dstSize.x || gid.y >= p.dstSize.y) return;

        constexpr sampler bilinear(filter::linear, address::clamp_to_edge);

        // 출력 픽셀이 커버하는 입력 영역의 중심과 범위 계산
        float2 center = (float2(gid) + 0.5) / float2(p.dstSize);
        float2 pixelSpan = 1.0 / float2(p.dstSize);  // 출력 1픽셀이 차지하는 UV 범위

        // 4×4 그리드로 샘플 — 각 bilinear 샘플이 2×2 입력 픽셀 평균
        // → 총 4×4×2×2 = 64 입력 픽셀 커버 (256 중 25%)
        float sum = 0.0;
        for (int sy = 0; sy < 4; sy++) {
            for (int sx = 0; sx < 4; sx++) {
                float2 offset = (float2(sx, sy) - 1.5) * (pixelSpan / 4.0);
                float4 c = src.sample(bilinear, center + offset);
                sum += dot(c.rgb, float3(0.299, 0.587, 0.114));
            }
        }

        float gray = sum / 16.0;
        dst.write(float4(gray, 0, 0, 1), gid);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 2: 블록 매칭 (다운스케일 해상도에서)
    // 각 스레드 = 1 블록 → 최적 변위 벡터 출력
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    struct BlockMatchParams {
        uint2 gridSize;   // 블록 그리드 크기 (dstW/blockSize, dstH/blockSize)
        uint  blockSize;
        int   searchRadius;
        uint2 imgSize;    // 다운스케일 이미지 크기
    };

    kernel void blockMatch(
        texture2d<float, access::read>  grayA   [[texture(0)]],
        texture2d<float, access::read>  grayB   [[texture(1)]],
        texture2d<float, access::write> motion  [[texture(2)]],
        constant BlockMatchParams& p            [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.gridSize.x || gid.y >= p.gridSize.y) return;

        uint bx = gid.x * p.blockSize;
        uint by = gid.y * p.blockSize;

        float bestSAD = 1e10;
        int2  bestOff = int2(0, 0);
        float sadAtZero = 1e10;  // (0,0) 오프셋에서의 SAD — 비교 기준

        // 3-Step Search: 거친→중간→정밀 (O(25*3) vs 전수탐색 O(17*17=289))
        // step: 4 → 2 → 1
        int2 center = int2(0, 0);

        for (int step = 4; step >= 1; step /= 2) {
            int2 localBest = center;
            float localBestSAD = 1e10;

            for (int oy = -step; oy <= step; oy += step) {
                for (int ox = -step; ox <= step; ox += step) {
                    int2 off = center + int2(ox, oy);

                    // 탐색 범위 클램프
                    if (abs(off.x) > p.searchRadius || abs(off.y) > p.searchRadius) continue;

                    float sad = 0.0;
                    // 블록 내 4x4 서브샘플 (16x16 블록에서 매 4번째 픽셀)
                    uint ssStep = max(1u, p.blockSize / 4);
                    for (uint dy = 0; dy < p.blockSize; dy += ssStep) {
                        for (uint dx = 0; dx < p.blockSize; dx += ssStep) {
                            int ax = int(bx + dx);
                            int ay = int(by + dy);
                            int bxx = ax + off.x;
                            int byy = ay + off.y;

                            if (bxx >= 0 && bxx < int(p.imgSize.x) &&
                                byy >= 0 && byy < int(p.imgSize.y))
                            {
                                float ga = grayA.read(uint2(ax, ay)).r;
                                float gb = grayB.read(uint2(bxx, byy)).r;
                                sad += abs(ga - gb);
                            } else {
                                sad += 0.5;
                            }
                        }
                    }

                    // (0,0) 오프셋의 SAD 기록 (첫 이터레이션에서만)
                    if (off.x == 0 && off.y == 0) {
                        sadAtZero = sad;
                    }

                    if (sad < localBestSAD) {
                        localBestSAD = sad;
                        localBest = off;
                    }
                }
            }

            center = localBest;
            if (localBestSAD < bestSAD) {
                bestSAD = localBestSAD;
                bestOff = localBest;
            }
        }

        // 모션 판정: sadAtZero 대비 bestSAD의 개선량으로 결정
        // 정적 화면에서는 (0,0)이 최선이거나, 노이즈로 미세하게 나은 오프셋을 찾지만
        // 개선량이 매우 작음. 실제 모션이 있으면 bestOff에서 SAD가 확연히 낮아짐.
        float improvement = sadAtZero - bestSAD;
        float minImprovement = 0.15;  // SAD 개선이 이만큼은 되어야 실제 모션으로 인정
        float2 mv;
        if (bestOff.x == 0 && bestOff.y == 0) {
            mv = float2(0.0);  // 제자리가 최선 = 정적
        } else if (improvement < minImprovement) {
            mv = float2(0.0);  // 개선량 미미 = 노이즈, 정적 처리
        } else {
            mv = float2(bestOff) * float(DOWNSCALE_FACTOR);  // 유의미한 개선 = 실제 모션
        }
        motion.write(float4(mv.x, mv.y, 0, 1), gid);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Phase 3: 양방향 모션 워프 + 블렌드 (원본 해상도)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    struct WarpParams {
        float t;
        uint2 outSize;
        uint2 mvGridSize;  // 모션 벡터 텍스처 크기
        float mvScaleX;    // outSize.x / mvGridSize.x
        float mvScaleY;    // outSize.y / mvGridSize.y
    };

    kernel void motionWarpBlend(
        texture2d<float, access::sample> frameA  [[texture(0)]],
        texture2d<float, access::sample> frameB  [[texture(1)]],
        texture2d<float, access::read>   motion  [[texture(2)]],
        texture2d<float, access::write>  output  [[texture(3)]],
        constant WarpParams& p                   [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= p.outSize.x || gid.y >= p.outSize.y) return;

        constexpr sampler bilinear(filter::linear, address::clamp_to_edge);

        // 모션 벡터 bilinear 보간 (블록 그리드 → 픽셀)
        float2 mvUV = (float2(gid) + 0.5) / float2(p.outSize) * float2(p.mvGridSize);
        // 가장 가까운 4개 블록에서 bilinear 보간
        int2 mvBase = int2(floor(mvUV - 0.5));
        float2 f = fract(mvUV - 0.5);

        float2 mv00 = float2(0), mv10 = float2(0), mv01 = float2(0), mv11 = float2(0);
        if (mvBase.x >= 0 && mvBase.x < int(p.mvGridSize.x) &&
            mvBase.y >= 0 && mvBase.y < int(p.mvGridSize.y))
            mv00 = motion.read(uint2(mvBase)).rg;
        if (mvBase.x + 1 >= 0 && mvBase.x + 1 < int(p.mvGridSize.x) &&
            mvBase.y >= 0 && mvBase.y < int(p.mvGridSize.y))
            mv10 = motion.read(uint2(mvBase.x + 1, mvBase.y)).rg;
        if (mvBase.x >= 0 && mvBase.x < int(p.mvGridSize.x) &&
            mvBase.y + 1 >= 0 && mvBase.y + 1 < int(p.mvGridSize.y))
            mv01 = motion.read(uint2(mvBase.x, mvBase.y + 1)).rg;
        if (mvBase.x + 1 >= 0 && mvBase.x + 1 < int(p.mvGridSize.x) &&
            mvBase.y + 1 >= 0 && mvBase.y + 1 < int(p.mvGridSize.y))
            mv11 = motion.read(uint2(mvBase.x + 1, mvBase.y + 1)).rg;

        float2 mv = mix(mix(mv00, mv10, f.x), mix(mv01, mv11, f.x), f.y);

        float2 baseUV = (float2(gid) + 0.5) / float2(p.outSize);

        // ── 안전 blend: 모션 크기에 따라 warp ↔ blend 혼합 ──
        // 모션 벡터가 클수록 잘못된 매칭 가능성 높음
        // mvLen이 작으면 confidence 높음 → warp 사용
        // mvLen이 크면 confidence 낮음 → 단순 blend fallback
        float mvLen = length(mv);
        // ── 정적 영역: 현재 프레임(B)을 정확한 픽셀로 출력 ──
        // sample(bilinear)은 float UV 정밀도 오차로 인접 픽셀 보간 → 부들거림
        // read(gid)는 정수 좌표로 정확한 텍셀 값 반환
        if (mvLen < 1.0) {
            output.write(frameB.read(gid), gid);
            return;
        }

        // ── 이동 영역: 순수 양방향 워프 (블렌드 폴백 없음) ──
        // mv는 A→B 전방 모션. 중간 프레임에서는 역추적.
        float2 pos = float2(gid) + 0.5;
        float2 uvA = (pos - p.t * mv) / float2(p.outSize);
        float2 uvB = (pos + (1.0 - p.t) * mv) / float2(p.outSize);

        float4 warpA = frameA.sample(bilinear, uvA);
        float4 warpB = frameB.sample(bilinear, uvB);
        float4 result = mix(warpA, warpB, p.t);
        result.a = 1.0;
        output.write(result, gid);
    }
    """

    public init() {}

    public func prepare(device: any MTLDevice) async throws {
        self.device = device
        self.texturePool = TexturePool(device: device)

        // DOWNSCALE_FACTOR를 #define으로 주입
        let opts = MTLCompileOptions()
        opts.preprocessorMacros = ["DOWNSCALE_FACTOR": NSNumber(value: downscaleFactor)]

        let library = try await device.makeLibrary(source: Self.shaderSource, options: opts)

        guard let fnDown = library.makeFunction(name: "downsampleGray"),
              let fnMatch = library.makeFunction(name: "blockMatch"),
              let fnWarp = library.makeFunction(name: "motionWarpBlend") else {
            throw InterpolationError.shaderCompilationFailed
        }

        downsamplePSO = try await device.makeComputePipelineState(function: fnDown)
        blockMatchPSO = try await device.makeComputePipelineState(function: fnMatch)
        motionWarpPSO = try await device.makeComputePipelineState(function: fnWarp)

        DiagnosticLog.shared.log("[FastMotion] prepare OK, downscale=1/\(downscaleFactor), block=\(blockSize), search=±\(searchRadius)")
    }

    private func ensureTextures(width: Int, height: Int) {
        guard width != cachedWidth || height != cachedHeight, let device else { return }
        cachedWidth = width
        cachedHeight = height

        let dw = width / downscaleFactor
        let dh = height / downscaleFactor

        grayA = makeTexture(device: device, w: dw, h: dh, format: .r16Float)
        grayB = makeTexture(device: device, w: dw, h: dh, format: .r16Float)

        let gridW = dw / blockSize
        let gridH = dh / blockSize
        motionTex = makeTexture(device: device, w: gridW, h: gridH, format: .rg16Float)

        DiagnosticLog.shared.log("[FastMotion] textures: src=\(width)x\(height) down=\(dw)x\(dh) grid=\(gridW)x\(gridH)")
    }

    public func interpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        t: Float,
        commandBuffer: any MTLCommandBuffer
    ) throws -> (any MTLTexture)? {
        guard device != nil, let texturePool,
              let downsamplePSO, let blockMatchPSO, let motionWarpPSO else {
            throw InterpolationError.notPrepared
        }

        interpCount += 1

        let w = frameB.width
        let h = frameB.height
        ensureTextures(width: w, height: h)

        guard let grayA, let grayB, let motionTex else {
            throw InterpolationError.notPrepared
        }

        // 출력 텍스처 — 소스와 동일한 pixelFormat (sRGB 불일치 방지)
        guard let output = texturePool.acquire(
            width: w, height: h,
            pixelFormat: frameB.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        ) else {
            throw InterpolationError.textureAllocationFailed
        }

        // ═══ 단일 encoder로 모든 단계 인코딩 (GPU 파이프라인 최적화) ═══
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        let dw = w / downscaleFactor
        let dh = h / downscaleFactor
        let tg16 = MTLSize(width: 16, height: 16, depth: 1)

        // ── Phase 1: 다운스케일 + 그레이스케일 ──
        struct DownsampleParams {
            var dstSize: SIMD2<UInt32>
            var srcSize: SIMD2<Float32>
        }
        var dsParams = DownsampleParams(
            dstSize: SIMD2(UInt32(dw), UInt32(dh)),
            srcSize: SIMD2(Float32(w), Float32(h))
        )

        enc.setComputePipelineState(downsamplePSO)

        // Downsample A
        enc.setTexture(frameA, index: 0)
        enc.setTexture(grayA, index: 1)
        enc.setBytes(&dsParams, length: MemoryLayout<DownsampleParams>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (dw + 15) / 16, height: (dh + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg16
        )

        // Downsample B
        enc.setTexture(frameB, index: 0)
        enc.setTexture(grayB, index: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: (dw + 15) / 16, height: (dh + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg16
        )

        enc.memoryBarrier(scope: .textures)

        // ── Phase 2: 블록 매칭 ──
        let gridW = dw / blockSize
        let gridH = dh / blockSize

        struct BlockMatchParams {
            var gridSize: SIMD2<UInt32>
            var blockSize: UInt32
            var searchRadius: Int32
            var imgSize: SIMD2<UInt32>
        }
        var bmParams = BlockMatchParams(
            gridSize: SIMD2(UInt32(gridW), UInt32(gridH)),
            blockSize: UInt32(blockSize),
            searchRadius: Int32(searchRadius),
            imgSize: SIMD2(UInt32(dw), UInt32(dh))
        )

        enc.setComputePipelineState(blockMatchPSO)
        enc.setTexture(grayA, index: 0)
        enc.setTexture(grayB, index: 1)
        enc.setTexture(motionTex, index: 2)
        enc.setBytes(&bmParams, length: MemoryLayout<BlockMatchParams>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (gridW + 15) / 16, height: (gridH + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg16
        )

        enc.memoryBarrier(scope: .textures)

        // ── Phase 3: 모션 워프 + 블렌드 ──
        struct WarpParams {
            var t: Float
            var outSize: SIMD2<UInt32>
            var mvGridSize: SIMD2<UInt32>
            var mvScaleX: Float
            var mvScaleY: Float
        }
        var warpParams = WarpParams(
            t: t,
            outSize: SIMD2(UInt32(w), UInt32(h)),
            mvGridSize: SIMD2(UInt32(gridW), UInt32(gridH)),
            mvScaleX: Float(w) / Float(gridW),
            mvScaleY: Float(h) / Float(gridH)
        )

        enc.setComputePipelineState(motionWarpPSO)
        enc.setTexture(frameA, index: 0)
        enc.setTexture(frameB, index: 1)
        enc.setTexture(motionTex, index: 2)
        enc.setTexture(output, index: 3)
        enc.setBytes(&warpParams, length: MemoryLayout<WarpParams>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg16
        )

        enc.endEncoding()

        return output
    }

    /// 배치 보간: downsample+blockmatch 1회 → warp N회 (단일 encoder)
    /// 개별 interpolate()를 N번 호출하면 downsample+blockmatch가 N번 반복되지만,
    /// 이 메서드는 1번만 수행 → N개 프레임을 ~6ms에 완성 (vs N×5.6ms)
    public func batchInterpolate(
        frameA: any MTLTexture,
        frameB: any MTLTexture,
        tValues: [Float],
        commandBuffer: any MTLCommandBuffer
    ) throws -> [any MTLTexture] {
        guard let device = self.device, let texturePool,
              let downsamplePSO, let blockMatchPSO, let motionWarpPSO else {
            throw InterpolationError.notPrepared
        }

        let w = frameB.width
        let h = frameB.height
        ensureTextures(width: w, height: h)

        guard let grayA, let grayB, let motionTex else {
            throw InterpolationError.notPrepared
        }

        // 출력 텍스처 N개 할당 — 소스와 동일한 pixelFormat
        var outputs: [any MTLTexture] = []
        for _ in tValues {
            guard let output = texturePool.acquire(
                width: w, height: h,
                pixelFormat: frameB.pixelFormat,
                usage: [.shaderRead, .shaderWrite]
            ) else {
                throw InterpolationError.textureAllocationFailed
            }
            outputs.append(output)
        }

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw InterpolationError.encoderCreationFailed
        }

        let dw = w / downscaleFactor
        let dh = h / downscaleFactor
        let tg16 = MTLSize(width: 16, height: 16, depth: 1)

        // ── Phase 1: 다운스케일 (1회) ──
        struct DownsampleParams {
            var dstSize: SIMD2<UInt32>
            var srcSize: SIMD2<Float32>
        }
        var dsParams = DownsampleParams(
            dstSize: SIMD2(UInt32(dw), UInt32(dh)),
            srcSize: SIMD2(Float32(w), Float32(h))
        )
        let dsGroups = MTLSize(width: (dw + 15) / 16, height: (dh + 15) / 16, depth: 1)

        enc.setComputePipelineState(downsamplePSO)
        enc.setTexture(frameA, index: 0)
        enc.setTexture(grayA, index: 1)
        enc.setBytes(&dsParams, length: MemoryLayout<DownsampleParams>.size, index: 0)
        enc.dispatchThreadgroups(dsGroups, threadsPerThreadgroup: tg16)

        enc.setTexture(frameB, index: 0)
        enc.setTexture(grayB, index: 1)
        enc.dispatchThreadgroups(dsGroups, threadsPerThreadgroup: tg16)

        enc.memoryBarrier(scope: .textures)

        // ── Phase 2: 블록 매칭 (1회) ──
        let gridW = dw / blockSize
        let gridH = dh / blockSize

        struct BlockMatchParams {
            var gridSize: SIMD2<UInt32>
            var blockSize: UInt32
            var searchRadius: Int32
            var imgSize: SIMD2<UInt32>
        }
        var bmParams = BlockMatchParams(
            gridSize: SIMD2(UInt32(gridW), UInt32(gridH)),
            blockSize: UInt32(blockSize),
            searchRadius: Int32(searchRadius),
            imgSize: SIMD2(UInt32(dw), UInt32(dh))
        )

        enc.setComputePipelineState(blockMatchPSO)
        enc.setTexture(grayA, index: 0)
        enc.setTexture(grayB, index: 1)
        enc.setTexture(motionTex, index: 2)
        enc.setBytes(&bmParams, length: MemoryLayout<BlockMatchParams>.size, index: 0)
        enc.dispatchThreadgroups(
            MTLSize(width: (gridW + 15) / 16, height: (gridH + 15) / 16, depth: 1),
            threadsPerThreadgroup: tg16
        )

        enc.memoryBarrier(scope: .textures)

        // ── Phase 3: 워프 N회 (t값별) ──
        struct WarpParams {
            var t: Float
            var outSize: SIMD2<UInt32>
            var mvGridSize: SIMD2<UInt32>
            var mvScaleX: Float
            var mvScaleY: Float
        }
        let warpGroups = MTLSize(width: (w + 15) / 16, height: (h + 15) / 16, depth: 1)

        for (i, t) in tValues.enumerated() {
            var warpParams = WarpParams(
                t: t,
                outSize: SIMD2(UInt32(w), UInt32(h)),
                mvGridSize: SIMD2(UInt32(gridW), UInt32(gridH)),
                mvScaleX: Float(w) / Float(gridW),
                mvScaleY: Float(h) / Float(gridH)
            )

            enc.setComputePipelineState(motionWarpPSO)
            enc.setTexture(frameA, index: 0)
            enc.setTexture(frameB, index: 1)
            enc.setTexture(motionTex, index: 2)
            enc.setTexture(outputs[i], index: 3)
            enc.setBytes(&warpParams, length: MemoryLayout<WarpParams>.size, index: 0)
            enc.dispatchThreadgroups(warpGroups, threadsPerThreadgroup: tg16)

            // 워프 간 barrier 불필요 — 각 워프는 독립된 출력 텍스처에 쓰고 같은 입력을 읽음
        }

        enc.endEncoding()

        // 매 60프레임마다 MV readback 스케줄
        interpCount += tValues.count
        if interpCount % 60 < tValues.count {
            // blit motion texture to CPU-readable copy
            if mvDiagReadback == nil || mvDiagGridW != gridW || mvDiagGridH != gridH {
                let desc = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rg16Float, width: gridW, height: gridH, mipmapped: false
                )
                desc.storageMode = .managed
                desc.usage = [.shaderRead]
                mvDiagReadback = device.makeTexture(descriptor: desc)
                mvDiagGridW = gridW
                mvDiagGridH = gridH
            }
            if let readback = mvDiagReadback, let blitEnc = commandBuffer.makeBlitCommandEncoder() {
                blitEnc.copy(from: motionTex, to: readback)
                blitEnc.synchronize(resource: readback)
                blitEnc.endEncoding()

                commandBuffer.addCompletedHandler { [weak self] _ in
                    self?.logMotionVectorStats()
                }
            }
        }

        return outputs
    }

    /// GPU 완료 후 MV 텍스처 통계 로깅
    private func logMotionVectorStats() {
        guard let tex = mvDiagReadback else { return }
        let w = tex.width
        let h = tex.height
        let bytesPerRow = w * 4  // rg16Float = 4 bytes per pixel
        var data = [UInt8](repeating: 0, count: bytesPerRow * h)
        tex.getBytes(&data, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)

        var nonZeroCount = 0
        var maxMVLen: Float = 0
        var sumMVLen: Float = 0
        data.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: Float16.self, capacity: w * h * 2) { p in
                for y in 0..<h {
                    for x in 0..<w {
                        let idx = y * w * 2 + x * 2
                        let mvx = Float(p[idx])
                        let mvy = Float(p[idx + 1])
                        let len = sqrt(mvx * mvx + mvy * mvy)
                        if len > 0.01 {
                            nonZeroCount += 1
                            sumMVLen += len
                            if len > maxMVLen { maxMVLen = len }
                        }
                    }
                }
            }
        }

        let totalBlocks = w * h
        let avgMV = nonZeroCount > 0 ? sumMVLen / Float(nonZeroCount) : 0
        DiagnosticLog.shared.log("[MV DIAG] grid=\(w)x\(h) nonZero=\(nonZeroCount)/\(totalBlocks) (\(String(format: "%.1f", Float(nonZeroCount) / Float(totalBlocks) * 100))%) avgMV=\(String(format: "%.1f", avgMV))px maxMV=\(String(format: "%.1f", maxMVLen))px")
    }

    public func shutdown() {
        grayA = nil; grayB = nil; motionTex = nil
        texturePool?.drain()
        texturePool = nil
        logger.info("FastMotionInterpolator shut down")
    }

    private func makeTexture(device: any MTLDevice, w: Int, h: Int, format: MTLPixelFormat) -> (any MTLTexture)? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: w, height: h, mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }
}
