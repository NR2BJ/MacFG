import AppKit
import Metal
import QuartzCore
import Monitoring

/// 스레드 무관 렌더 표면 — OverlayWindow(@MainActor)에서 분리한 출력 인코딩 경로.
///
/// A2(전용 렌더 스레드 + CAMetalDisplayLink)의 1단계: 렌더 틱이 NSView/NSWindow를 만지지
/// 않도록 필요한 값(레터박스 영역, 배율, 스타일, 샤픈/업스케일 설정)을 락 보호 캐시로 받고,
/// CAMetalLayer 조작(frame/drawableSize/nextDrawable)은 스레드 안전 경로만 사용한다.
/// 메인 스레드는 창 이벤트(리사이즈/화면 이동/설정 변경) 때 update()로 캐시를 갱신한다.
public final class RenderSurface: @unchecked Sendable {
    /// 렌더에 필요한 전부 — 메인이 쓰고 렌더가 읽는다 (락 보호 값 복사)
    public struct Params: Sendable {
        public var isViewer: Bool
        public var sharpness: Float
        public var upscaleMode: UpscaleMode
        /// 뷰어: 레터박스 배치 기준 영역(pt, contentView 좌표) / 커버: 창 크기(pt, origin 0)
        public var contentBounds: CGRect
        public var contentsScale: CGFloat
        /// 커버 전용 모서리 반경(pt) — 뷰어는 0
        public var cornerRadiusPt: CGFloat

        public init(isViewer: Bool, sharpness: Float, upscaleMode: UpscaleMode,
                    contentBounds: CGRect, contentsScale: CGFloat, cornerRadiusPt: CGFloat) {
            self.isViewer = isViewer
            self.sharpness = sharpness
            self.upscaleMode = upscaleMode
            self.contentBounds = contentBounds
            self.contentsScale = contentsScale
            self.cornerRadiusPt = cornerRadiusPt
        }
    }

    public let metalLayer: CAMetalLayer
    private let pipelineState: any MTLRenderPipelineState
    private let sampler: any MTLSamplerState
    private let upscaler: MetalFXUpscaler?
    private let neuralUpscaler: NeuralUpscaler?

    private let lock = NSLock()
    private var params: Params
    private var _scaleStatus: String?
    private var dbgFrames = 0   // TEMP 진단

    /// 업스케일/샤픈 실동작 상태 (UI 표시용)
    public var scaleStatus: String? {
        lock.lock(); defer { lock.unlock() }
        return _scaleStatus
    }

    public init(device: any MTLDevice, metalLayer: CAMetalLayer,
                pipelineState: any MTLRenderPipelineState, sampler: any MTLSamplerState,
                params: Params) {
        self.metalLayer = metalLayer
        self.pipelineState = pipelineState
        self.sampler = sampler
        self.params = params
        self.upscaler = MetalFXUpscaler(device: device)
        self.neuralUpscaler = NeuralUpscaler(device: device)
    }

    /// 메인 스레드: 창/설정 변경 반영
    public func update(_ transform: (inout Params) -> Void) {
        lock.lock(); defer { lock.unlock() }
        transform(&params)
    }

    private func currentParams() -> Params {
        lock.lock(); defer { lock.unlock() }
        return params
    }

    /// 텍스처를 외부 commandBuffer에 렌더 인코딩. drawable 반환 — 호출자가 present+commit.
    /// 어떤 스레드에서든 호출 가능 (CAMetalLayer drawable 경로는 스레드 안전,
    /// layer.frame 변경은 명시적 CATransaction).
    public func encode(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer) -> (any CAMetalDrawable)? {
        let p = currentParams()

        if p.isViewer {
            layoutViewerLayer(textureWidth: texture.width, textureHeight: texture.height, bounds: p.contentBounds, scale: p.contentsScale)
        }

        // 업스케일: 뷰어에서 표시 목표(레터박스 영역 × 배율)가 소스보다 크면 ANE/MetalFX로
        // 선명하게 올린다 (off이거나 실패하면 이중선형 스케일로 폴백 = 아래 그대로).
        var source = texture
        if p.upscaleMode != .off || p.sharpness > 0.01 {
            var chain: [String] = []
            if p.isViewer, p.upscaleMode != .off {
                let targetW = Int((metalLayer.frame.width * p.contentsScale).rounded())
                let targetH = Int((metalLayer.frame.height * p.contentsScale).rounded())
                if targetW > texture.width || targetH > texture.height {
                    var cur = source
                    // 1) ANE 신경망 2x (모드가 ane/aneMetalfx이고 소스 ≤960)
                    if p.upscaleMode == .ane || p.upscaleMode == .aneMetalfx,
                       texture.width <= NeuralUpscaler.maxInput, texture.height <= NeuralUpscaler.maxInput,
                       let n = neuralUpscaler?.upscale(texture, into: commandBuffer) {
                        cur = n
                        chain.append("ANE→\(n.width)×\(n.height)")
                    }
                    // 2) MetalFX로 목표까지 (모드가 metalfx/aneMetalfx이고 아직 작으면)
                    if p.upscaleMode == .metalfx || p.upscaleMode == .aneMetalfx,
                       targetW > cur.width || targetH > cur.height,
                       let m = upscaler?.upscale(cur, outWidth: targetW, outHeight: targetH, commandBuffer: commandBuffer) {
                        cur = m
                        chain.append("MetalFX→\(targetW)×\(targetH)")
                    }
                    source = cur
                }
            }
            if chain.isEmpty { chain.append(p.isViewer ? "1:1" : "1:1 cover") }
            // CAS는 스케일과 무관하게 최종 블릿에서 적용 (Cover 포함) — 늘어난 저해상도 영상 복원
            chain.append(p.sharpness > 0.01 ? String(format: "sharpen %.1f", p.sharpness) : "sharpen off")
            let status = chain.joined(separator: " · ")
            lock.lock(); _scaleStatus = status; lock.unlock()
        } else {
            lock.lock(); _scaleStatus = nil; lock.unlock()
        }

        let texW = CGFloat(source.width)
        let texH = CGFloat(source.height)
        if metalLayer.drawableSize.width != texW || metalLayer.drawableSize.height != texH {
            metalLayer.drawableSize = CGSize(width: texW, height: texH)
        }

        guard let drawable = metalLayer.nextDrawable() else { return nil }
        encodeBlit(source: source, into: commandBuffer, target: drawable.texture, params: p)
        return drawable
    }

    /// 제공된 드로어블에 직접 인코딩 (CAMetalDisplayLink 경로 — nextDrawable 없음).
    /// 반환: 인코딩 성공 여부.
    @discardableResult
    public func encode(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer,
                       drawable: any CAMetalDrawable) -> Bool {
        let p = currentParams()
        if p.isViewer {
            layoutViewerLayer(textureWidth: texture.width, textureHeight: texture.height, bounds: p.contentBounds, scale: p.contentsScale)
        }
        var source = texture
        if p.upscaleMode != .off || p.sharpness > 0.01 {
            var chain: [String] = []
            if p.isViewer, p.upscaleMode != .off {
                let targetW = Int((metalLayer.frame.width * p.contentsScale).rounded())
                let targetH = Int((metalLayer.frame.height * p.contentsScale).rounded())
                if dbgFrames < 8 {
                    dbgFrames += 1
                    DiagnosticLog.shared.log("[UPSCALE-DBG] f\(dbgFrames) tex=\(texture.width)x\(texture.height) bounds=\(Int(p.contentBounds.width))x\(Int(p.contentBounds.height)) layerFrame=\(Int(metalLayer.frame.width))x\(Int(metalLayer.frame.height)) scale=\(p.contentsScale) target=\(targetW)x\(targetH) engage=\(targetW > texture.width || targetH > texture.height) mode=\(p.upscaleMode)")
                }
                if targetW > texture.width || targetH > texture.height {
                    var cur = source
                    if p.upscaleMode == .ane || p.upscaleMode == .aneMetalfx,
                       texture.width <= NeuralUpscaler.maxInput, texture.height <= NeuralUpscaler.maxInput,
                       let n = neuralUpscaler?.upscale(texture, into: commandBuffer) {
                        cur = n
                        chain.append("ANE→\(n.width)×\(n.height)")
                    }
                    if p.upscaleMode == .metalfx || p.upscaleMode == .aneMetalfx,
                       targetW > cur.width || targetH > cur.height,
                       let m = upscaler?.upscale(cur, outWidth: targetW, outHeight: targetH, commandBuffer: commandBuffer) {
                        cur = m
                        chain.append("MetalFX→\(targetW)×\(targetH)")
                    }
                    source = cur
                }
            }
            if chain.isEmpty { chain.append(p.isViewer ? "1:1" : "1:1 cover") }
            if dbgFrames <= 8 { DiagnosticLog.shared.log("[UPSCALE-DBG]   → chain=\(chain.joined(separator: " · "))") }
            chain.append(p.sharpness > 0.01 ? String(format: "sharpen %.1f", p.sharpness) : "sharpen off")
            let status = chain.joined(separator: " · ")
            lock.lock(); _scaleStatus = status; lock.unlock()
        } else {
            lock.lock(); _scaleStatus = nil; lock.unlock()
        }
        encodeBlit(source: source, into: commandBuffer, target: drawable.texture, params: p)
        return true
    }

    private func encodeBlit(source: any MTLTexture, into commandBuffer: any MTLCommandBuffer,
                            target: any MTLTexture, params p: Params) {
        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = target
        renderPassDesc.colorAttachments[0].loadAction = .dontCare
        renderPassDesc.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return }
        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentSamplerState(sampler, index: 0)
        // 모서리 마스킹: cover만 (viewer는 레터박스 배경이 검정이라 불필요)
        var bp = BlitParams(
            size: SIMD2<Float>(Float(source.width), Float(source.height)),
            radiusPx: p.isViewer ? 0 : Float(p.cornerRadiusPt * p.contentsScale),
            sharpness: p.sharpness
        )
        encoder.setFragmentBytes(&bp, length: MemoryLayout<BlitParams>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
    }

    private struct BlitParams {
        var size: SIMD2<Float>
        var radiusPx: Float
        var sharpness: Float
    }

    /// 뷰어: 캐시된 bounds 안에서 텍스처 종횡비 유지 레터박스 배치 (NSView 접근 없음)
    private func layoutViewerLayer(textureWidth: Int, textureHeight: Int, bounds: CGRect, scale: CGFloat) {
        guard textureWidth > 0, textureHeight > 0, bounds.width > 0, bounds.height > 0 else { return }
        let aspect = CGFloat(textureWidth) / CGFloat(textureHeight)
        var fitW = bounds.width
        var fitH = fitW / aspect
        if fitH > bounds.height {
            fitH = bounds.height
            fitW = fitH * aspect
        }
        let fit = CGRect(
            x: bounds.minX + (bounds.width - fitW) / 2,
            y: bounds.minY + (bounds.height - fitH) / 2,
            width: fitW,
            height: fitH
        )
        if metalLayer.frame != fit || metalLayer.contentsScale != scale {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            metalLayer.frame = fit
            if metalLayer.contentsScale != scale { metalLayer.contentsScale = scale }
            CATransaction.commit()
        }
    }
}
