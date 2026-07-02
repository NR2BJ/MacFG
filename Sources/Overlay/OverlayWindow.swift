import AppKit
import Metal
import MetalKit
import QuartzCore
import os

/// 셰이더 캐시 — 앱 라이프타임 동안 1회 컴파일 후 재사용
private final class ShaderCache: @unchecked Sendable {
    static let shared = ShaderCache()

    private var pipelineState: (any MTLRenderPipelineState)?
    private var sampler: (any MTLSamplerState)?
    private let lock = NSLock()

    func getOrCreate(device: any MTLDevice) throws -> (any MTLRenderPipelineState, any MTLSamplerState) {
        lock.lock()
        defer { lock.unlock() }

        if let ps = pipelineState, let s = sampler {
            return (ps, s)
        }

        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };

        struct BlitParams {
            float2 size;      // drawable 픽셀 크기
            float radiusPx;   // 모서리 반경 (px, 0=마스킹 없음)
        };

        vertex VertexOut blitVertex(uint vid [[vertex_id]]) {
            float2 positions[] = {
                float2(-1, -1), float2(1, -1),
                float2(-1,  1), float2(1,  1)
            };
            float2 texCoords[] = {
                float2(0, 1), float2(1, 1),
                float2(0, 0), float2(1, 0)
            };
            VertexOut out;
            out.position = float4(positions[vid], 0, 1);
            out.texCoord = texCoords[vid];
            return out;
        }

        // macOS 창의 둥근 모서리 재현 — CAMetalLayer cornerRadius는 직접 스캔아웃
        // 경로에서 무시되는 것을 실측(모든 반경에서 픽셀 동일) → 셰이더 SDF 마스킹.
        // 모서리 바깥은 alpha 0 (premultiplied) → 창 투명 영역으로 원본 모서리가 비침.
        fragment float4 blitFragment(VertexOut in [[stage_in]],
                                      texture2d<float> tex [[texture(0)]],
                                      sampler samp [[sampler(0)]],
                                      constant BlitParams& p [[buffer(0)]]) {
            float4 color = tex.sample(samp, in.texCoord);
            color.a = 1.0;
            if (p.radiusPx > 0.5) {
                float2 pos = in.texCoord * p.size;
                float2 half_ = p.size * 0.5;
                float2 q = fabs(pos - half_) - (half_ - p.radiusPx);
                float d = length(max(q, 0.0)) - p.radiusPx;
                float a = saturate(0.5 - d);   // 1px AA
                color.rgb *= a;
                color.a = a;
            }
            return color;
        }
        """

        let library = try device.makeLibrary(source: shaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "blitVertex")
        desc.fragmentFunction = library.makeFunction(name: "blitFragment")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm

        let ps = try device.makeRenderPipelineState(descriptor: desc)

        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.mipFilter = .notMipmapped
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        let s = device.makeSamplerState(descriptor: samplerDesc)!

        self.pipelineState = ps
        self.sampler = s
        return (ps, s)
    }
}

/// 출력 창 스타일
public enum OverlayStyleConstants {
    /// 표준 macOS 창 모서리 반경 (pt) — 검은 조각 경계 곡선 피팅으로 실측 14.1pt (2026-07-02)
    public nonisolated(unsafe) static var cornerRadius: CGFloat = 14
}

public enum OverlayStyle: Sendable {
    /// borderless 오버레이 — 대상 창 위를 덮음 (Cover Source)
    case overlay
    /// 일반 titled/resizable 창 — 사용자가 자유롭게 이동/리사이즈 (Separate Window)
    case viewer
}

/// 출력 창: borderless 오버레이 또는 이동 가능한 뷰어
@MainActor
public final class OverlayWindow: NSObject {
    public let style: OverlayStyle
    private let window: NSWindow
    private let metalLayer: CAMetalLayer
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLRenderPipelineState
    private let sampler: any MTLSamplerState
    private let logger = Logger(subsystem: "com.macfg", category: "OverlayWindow")
    private var appliedColorSpace: CGColorSpace?
    private var colorSpaceInitialized = false

    /// 뷰어 창을 사용자가 닫았을 때 (X 버튼)
    public var onUserClose: (() -> Void)?

    public init(device: any MTLDevice, style: OverlayStyle, title: String = "MacFG Output") throws {
        self.device = device
        self.style = style
        self.commandQueue = device.makeCommandQueue()!

        // 셰이더 캐시에서 가져오기 — 최초 1회만 컴파일
        let (ps, s) = try ShaderCache.shared.getOrCreate(device: device)
        self.pipelineState = ps
        self.sampler = s

        let window: NSWindow
        let metalLayer = CAMetalLayer()
        metalLayer.device = device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.isOpaque = true
        // vsync 정렬 present — 페이싱은 present(at: targetTimestamp)가 담당.
        // (displaySyncEnabled=false는 mid-refresh 표시로 judder를 유발했음)
        metalLayer.displaySyncEnabled = true
        metalLayer.maximumDrawableCount = 3
        metalLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0

        switch style {
        case .overlay:
            // Borderless, non-activating, topmost 윈도우
            window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 100, height: 100),
                styleMask: [.borderless],
                backing: .buffered,
                defer: false
            )
            window.level = .floating
            window.isOpaque = false
            window.backgroundColor = .clear
            window.ignoresMouseEvents = true
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            window.hasShadow = false
            window.alphaValue = 1.0

            // 모서리 마스킹은 셰이더 SDF로 수행 (CALayer.cornerRadius는 CAMetalLayer
            // 직접 스캔아웃에서 무시됨 — 실측). 컴포지터가 알파를 쓰도록 비불투명 레이어.
            metalLayer.isOpaque = false

            let contentView = NSView(frame: window.contentView!.bounds)
            contentView.wantsLayer = true
            contentView.layer = metalLayer
            window.contentView = contentView

        case .viewer:
            // 일반 창 — 자유 이동/리사이즈, 레터박스 aspect-fit
            window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 960, height: 540),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            window.title = title
            window.level = .normal
            window.isOpaque = true
            window.backgroundColor = .black
            window.isReleasedWhenClosed = false
            window.collectionBehavior = [.fullScreenAuxiliary]

            if let contentView = window.contentView {
                contentView.wantsLayer = true
                contentView.layer?.backgroundColor = NSColor.black.cgColor
                contentView.layer?.addSublayer(metalLayer)
            }
        }

        // .readOnly: 스크린샷/녹화에 출력이 보이도록 (검증 및 사용자 녹화용).
        // 캡처는 SCK desktopIndependentWindow(대상 창 백킹만 캡처)라 자기 캡처 루프가 생기지 않는다.
        window.sharingType = .readOnly

        self.window = window
        self.metalLayer = metalLayer
        super.init()

        if style == .viewer {
            window.delegate = self
        }

        logger.info("OverlayWindow created (style=\(String(describing: style)))")
    }

    /// 색 처리 정책.
    /// - passthrough(colorspace=nil): 컬러 매칭 없이 캡처 바이트를 그대로 패널에 전달.
    ///   소스 창과 같은 디스플레이에 출력할 때 바이트 단위 일치 (실측: 태깅 시 SCK 태그와
    ///   디스플레이 프로필 간 변환으로 어두운 채널이 +2~6 뜸 — 2026-07-02 검증).
    /// - 캡처 태그: 다른 디스플레이로 출력할 때만 사용 (프로필 차이 보정).
    public func setColorSpace(_ captureColorSpace: CGColorSpace?, sameDisplayAsSource: Bool) {
        let target: CGColorSpace? = sameDisplayAsSource ? nil : captureColorSpace
        if appliedColorSpace != target || !colorSpaceInitialized {
            colorSpaceInitialized = true
            appliedColorSpace = target
            metalLayer.colorspace = target
            let csName = target?.name.map { $0 as String } ?? "passthrough"
            logger.info("Overlay colorspace: \(csName)")
        }
    }

    /// macOS 오클루전 최적화 우회 (Cover 배치 전용).
    /// 완전 불투명 오버레이가 대상 창을 덮으면 window server가 대상 창을 "완전 가려짐"으로
    /// 판정 → 대상 앱(브라우저/플레이어)이 렌더링을 멈춰 캡처가 정지 프레임만 받는다.
    /// alpha < 1.0이면 오클루전 판정을 피한다. 0.99는 정확히 1% 어두워짐이 실측됐고
    /// (255→252; 컴포지터가 가려진 대상 창을 컬링해 1% 아래 성분이 검정이 됨),
    /// 0.999는 모든 8bit 값에서 round(v*0.999)=v 라 바이트 단위 무손실. (2026-07-02 실측)
    public func setOcclusionBypass(_ enabled: Bool) {
        guard style == .overlay else { return }
        window.alphaValue = enabled ? 0.999 : 1.0
    }

    /// 텍스처를 외부 commandBuffer에 렌더 인코딩.
    /// drawable를 반환 — 호출자가 present + commit 담당.
    public func encodeRender(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer) -> (any CAMetalDrawable)? {
        if style == .viewer {
            layoutViewerLayer(textureWidth: texture.width, textureHeight: texture.height)
        }

        let texW = CGFloat(texture.width)
        let texH = CGFloat(texture.height)
        if metalLayer.drawableSize.width != texW || metalLayer.drawableSize.height != texH {
            metalLayer.drawableSize = CGSize(width: texW, height: texH)
        }

        guard let drawable = metalLayer.nextDrawable() else { return nil }

        let renderPassDesc = MTLRenderPassDescriptor()
        renderPassDesc.colorAttachments[0].texture = drawable.texture
        renderPassDesc.colorAttachments[0].loadAction = .dontCare
        renderPassDesc.colorAttachments[0].storeAction = .store

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else { return nil }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentSamplerState(sampler, index: 0)
        // 모서리 마스킹: overlay 스타일만 (viewer는 레터박스 배경이 검정이라 불필요)
        var params = BlitParams(
            size: SIMD2<Float>(Float(texW), Float(texH)),
            radiusPx: style == .overlay ? Float(OverlayStyleConstants.cornerRadius * metalLayer.contentsScale) : 0
        )
        encoder.setFragmentBytes(&params, length: MemoryLayout<BlitParams>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        return drawable
    }

    private struct BlitParams {
        var size: SIMD2<Float>
        var radiusPx: Float
    }

    /// 뷰어: contentView 안에서 텍스처 종횡비 유지 레터박스 배치
    private func layoutViewerLayer(textureWidth: Int, textureHeight: Int) {
        guard let contentView = window.contentView, textureWidth > 0, textureHeight > 0 else { return }
        let bounds = contentView.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }

        let aspect = CGFloat(textureWidth) / CGFloat(textureHeight)
        var fitW = bounds.width
        var fitH = fitW / aspect
        if fitH > bounds.height {
            fitH = bounds.height
            fitW = fitH * aspect
        }
        let fit = CGRect(
            x: (bounds.width - fitW) / 2,
            y: (bounds.height - fitH) / 2,
            width: fitW,
            height: fitH
        )
        if metalLayer.frame != fit {
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            metalLayer.frame = fit
            CATransaction.commit()
        }
        if let scale = window.screen?.backingScaleFactor, metalLayer.contentsScale != scale {
            metalLayer.contentsScale = scale
        }
    }

    /// 오버레이 위치/크기 갱신 (overlay 스타일 전용 — 뷰어는 사용자가 제어)
    public func updateFrame(_ frame: CGRect) {
        guard style == .overlay else { return }
        window.setFrame(frame, display: false)
        metalLayer.frame = NSRect(origin: .zero, size: frame.size)
        // contentsScale을 대상 화면에 맞게 갱신
        if let screen = window.screen {
            metalLayer.contentsScale = screen.backingScaleFactor
        }
    }

    /// 뷰어 초기 위치/크기 (1회)
    public func setInitialViewerFrame(_ frame: CGRect) {
        guard style == .viewer else { return }
        window.setFrame(frame, display: true)
    }

    /// 표시/숨김
    public func setVisible(_ visible: Bool) {
        if visible {
            window.orderFront(nil)
        } else {
            window.orderOut(nil)
        }
    }

    /// 오버레이가 현재 위치한 화면
    public var currentScreen: NSScreen? {
        window.screen
    }
}

extension OverlayWindow: NSWindowDelegate {
    public func windowWillClose(_ notification: Notification) {
        onUserClose?()
    }
}
