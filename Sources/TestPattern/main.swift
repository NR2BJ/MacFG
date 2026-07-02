// MacFG 자체 검증용 60fps 테스트 패턴 창.
// - 수평 스윕 바 + 스크롤 체커보드: 보간 동작/부드러움 육안+캡처 확인
// - 하단 고정 색 스트립 (R/G/B/그레이): 오버레이 색 왜곡 픽셀 비교 기준
// - 정지 UI 영역 (좌측 고정 텍스트 블록): 정적 부들거림 감지
import AppKit
import Metal
import QuartzCore

final class PatternView: NSView {
    let metalLayer = CAMetalLayer()
    private var device: (any MTLDevice)!
    private var queue: (any MTLCommandQueue)!
    private var pso: (any MTLRenderPipelineState)!
    private var displayLink: CADisplayLink?
    private var frameIndex: UInt32 = 0
    private var lastAdvance: CFTimeInterval = 0
    var contentFPS: Double = 60.0

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        device = MTLCreateSystemDefaultDevice()
        queue = device.makeCommandQueue()
        metalLayer.device = device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.isOpaque = true
        layer = metalLayer

        let src = """
        #include <metal_stdlib>
        using namespace metal;
        struct VOut { float4 pos [[position]]; float2 uv; };
        vertex VOut vmain(uint vid [[vertex_id]]) {
            float2 p[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
            VOut o; o.pos = float4(p[vid],0,1); o.uv = (p[vid]+1.0)*0.5; o.uv.y = 1.0-o.uv.y; return o;
        }
        fragment float4 fmain(VOut in [[stage_in]], constant uint& frame [[buffer(0)]],
                              constant float2& res [[buffer(1)]]) {
            float2 px = in.uv * res;
            float3 c = float3(0.13, 0.13, 0.15); // 배경

            // 상단 1/4: 스크롤 체커보드 (프레임당 6px 이동)
            if (px.y < res.y * 0.25) {
                float sx = px.x + float(frame) * 6.0;
                bool a = (fmod(floor(sx / 40.0) + floor(px.y / 40.0), 2.0) < 1.0);
                c = a ? float3(0.85) : float3(0.25);
            }
            // 중단: 수평 스윕 바 (프레임당 8px — 60fps에서 480px/s)
            else if (px.y < res.y * 0.62) {
                float barX = fmod(float(frame) * 8.0, res.x);
                float d = abs(px.x - barX);
                if (d < 14.0) c = float3(0.95, 0.55, 0.1);
                else if (fmod(px.x, 120.0) < 2.0) c = float3(0.3); // 고정 눈금
            }
            // 정지 UI 영역
            else if (px.y < res.y * 0.80) {
                float2 lp = float2(fmod(px.x, 160.0), fmod(px.y - res.y * 0.62, 40.0));
                if (lp.y > 12.0 && lp.y < 28.0 && lp.x > 10.0 && lp.x < 130.0) c = float3(0.65);
            }
            // 하단: 고정 색 기준 스트립
            else {
                float seg = px.x / res.x;
                if (seg < 0.2) c = float3(1.0, 0.0, 0.0);
                else if (seg < 0.4) c = float3(0.0, 1.0, 0.0);
                else if (seg < 0.6) c = float3(0.0, 0.0, 1.0);
                else if (seg < 0.8) c = float3(0.5);
                else c = float3(1.0, 0.0, 1.0);
            }

            // 프레임 카운터: 좌상단 8비트 바이너리 블록 (freeze 감지)
            if (px.y < 24.0 && px.x < 8.0 * 28.0) {
                uint bit = uint(px.x / 28.0);
                bool on = ((frame >> bit) & 1u) == 1u;
                c = on ? float3(1.0, 1.0, 0.0) : float3(0.05);
            }
            return float4(c, 1.0);
        }
        """
        let lib = try! device.makeLibrary(source: src, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "vmain")
        desc.fragmentFunction = lib.makeFunction(name: "fmain")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        pso = try! device.makeRenderPipelineState(descriptor: desc)
    }

    required init?(coder: NSCoder) { fatalError() }

    func startRendering() {
        guard displayLink == nil, let screen = window?.screen ?? NSScreen.main else { return }
        let link = screen.displayLink(target: self, selector: #selector(tick(_:)))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @objc private func tick(_ link: CADisplayLink) {
        // 120Hz 틱에서 60fps로 프레임 전진
        let now = link.timestamp
        if now - lastAdvance >= (1.0 / contentFPS) - 0.002 {
            lastAdvance = now
            frameIndex &+= 1
            render()
        }
    }

    private func render() {
        let scale = window?.screen?.backingScaleFactor ?? 1.0
        let sz = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        if metalLayer.drawableSize != sz {
            metalLayer.drawableSize = sz
            metalLayer.contentsScale = scale
        }
        guard let drawable = metalLayer.nextDrawable(),
              let cb = queue.makeCommandBuffer() else { return }
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = drawable.texture
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return }
        enc.setRenderPipelineState(pso)
        var frame = frameIndex
        var res = SIMD2<Float>(Float(sz.width), Float(sz.height))
        enc.setFragmentBytes(&frame, length: 4, index: 0)
        enc.setFragmentBytes(&res, length: 8, index: 1)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

// MARK: - App bootstrap

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: PatternView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let args = ProcessInfo.processInfo.arguments
        var w: CGFloat = 1280, h: CGFloat = 720
        var px: CGFloat = 120, py: CGFloat = 120
        if let i = args.firstIndex(of: "--size"), i + 2 < args.count,
           let aw = Double(args[i + 1]), let ah = Double(args[i + 2]) {
            w = aw; h = ah
        }
        if let i = args.firstIndex(of: "--pos"), i + 2 < args.count,
           let ax = Double(args[i + 1]), let ay = Double(args[i + 2]) {
            px = ax; py = ay
        }
        var fps: Double = 60
        if let i = args.firstIndex(of: "--fps"), i + 1 < args.count, let f = Double(args[i + 1]) {
            fps = f
        }
        window = NSWindow(
            contentRect: NSRect(x: px, y: py, width: w, height: h),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered, defer: false
        )
        window.title = "MacFG Test Pattern"
        view = PatternView(frame: NSRect(x: 0, y: 0, width: w, height: h))
        view.contentFPS = fps
        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        view.startRendering()
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
