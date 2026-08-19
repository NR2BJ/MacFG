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
    /// 실콘텐츠(브라우저)의 PTS 지터 재현용 — 프레임별 ±jitterMs 랜덤 지연
    var jitterMs: Double = 0
    /// 고난도 콘텐츠 (회전/노이즈/밝기변화) — 보간 화질 검증용
    var complexMode: Bool = false

    // 마우스 수신 로깅 — MacFG 뷰어 이벤트 포워딩(postToPid) 수신 검증용
    override var acceptsFirstResponder: Bool { true }
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.activeAlways, .mouseMoved, .inVisibleRect], owner: self))
    }
    /// 마우스 이동에 반응해 즉시 한 프레임 진행 — **브라우저의 호버 리페인트 재현용**(--hover-repaint).
    ///
    /// 실사용 소스(브라우저)는 마우스가 움직이면 영상 프레임 사이에 창 갱신을 끼워 넣어,
    /// 우리가 재는 소스율이 60 → 90~110fps로 뛴다. 그 상승이 보간 예산을 잠식하는지를
    /// 재려면 그 조건을 결정적으로 만들 소스가 필요한데, TestPattern은 고정 fps라 마우스에
    /// 무반응이었다. 이 플래그가 그 간극을 메운다 — 실제 브라우저 없이, 콘텐츠 드리프트 없이
    /// "마우스 움직이면 소스율이 오르는" 상태를 정확히 재현한다.
    var hoverRepaint = false
    /// UI 마커 위치(드로어블 픽셀). 음수면 그리지 않는다.
    var uiMark = SIMD2<Float>(-1, -1)
    override func mouseMoved(with e: NSEvent)  {
        NSLog("[TP-MOUSE] moved (\(Int(e.locationInWindow.x)),\(Int(e.locationInWindow.y)))")
        if hoverRepaint {
            // **frameIndex를 진행시키지 않는다** — 영상은 멈춘 채 UI만 바뀌는 상태를 만든다.
            let scale = window?.backingScaleFactor ?? 1
            uiMark = SIMD2<Float>(Float(e.locationInWindow.x * scale),
                                  Float((bounds.height - e.locationInWindow.y) * scale))
            render()
        }
    }
    override func mouseDown(with e: NSEvent)   { NSLog("[TP-MOUSE] DOWN (\(Int(e.locationInWindow.x)),\(Int(e.locationInWindow.y)))") }
    override func mouseUp(with e: NSEvent)     { NSLog("[TP-MOUSE] UP (\(Int(e.locationInWindow.x)),\(Int(e.locationInWindow.y)))") }
    override func scrollWheel(with e: NSEvent) { NSLog("[TP-MOUSE] SCROLL (\(Int(e.locationInWindow.x)),\(Int(e.locationInWindow.y))) dy=\(Int(e.scrollingDeltaY))") }

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
        float vnoise(float2 p) {
            // 해시 기반 값 노이즈 (결정적)
            float2 i = floor(p);
            float2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            float a = fract(sin(dot(i, float2(127.1, 311.7))) * 43758.5453);
            float b = fract(sin(dot(i + float2(1, 0), float2(127.1, 311.7))) * 43758.5453);
            float c2 = fract(sin(dot(i + float2(0, 1), float2(127.1, 311.7))) * 43758.5453);
            float d = fract(sin(dot(i + float2(1, 1), float2(127.1, 311.7))) * 43758.5453);
            return mix(mix(a, b, f.x), mix(c2, d, f.x), f.y);
        }

        fragment float4 fmain(VOut in [[stage_in]], constant uint& frame [[buffer(0)]],
                              constant float2& res [[buffer(1)]],
                              constant uint& complexMode [[buffer(2)]],
                              constant float2& uiMark [[buffer(3)]]) {
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

            // 고난도 모드: 회전 휠(블록매칭 최약점) + 노이즈 텍스처 블록 + 밝기 펄스
            if (complexMode != 0) {
                // 회전 스포크 휠
                float2 wc = res * float2(0.72, 0.42);
                float2 d = px - wc;
                float r = length(d);
                float wheelR = min(res.x, res.y) * 0.16;
                if (r < wheelR) {
                    float ang = atan2(d.y, d.x) + float(frame) * 0.035;
                    float spokes = step(0.0, sin(ang * 9.0));
                    float rings = step(0.5, fract(r / (wheelR * 0.2)));
                    c = mix(float3(0.15, 0.2, 0.5), float3(0.95, 0.9, 0.3), spokes * 0.7 + rings * 0.3);
                    if (r > wheelR * 0.94) c = float3(0.9);
                }
                // 노이즈 텍스처 블록 (대각 이동)
                float2 bp = float2(fmod(float(frame) * 3.0, res.x * 0.6),
                                   res.y * 0.55 + sin(float(frame) * 0.05) * res.y * 0.1);
                float2 lp2 = px - bp;
                if (lp2.x >= 0.0 && lp2.x < 260.0 && lp2.y >= 0.0 && lp2.y < 200.0) {
                    float n = vnoise(lp2 * 0.08) * 0.6 + vnoise(lp2 * 0.31) * 0.4;
                    c = float3(n * 0.9, n * 0.75, n * 0.5);
                }
                // 체커 구역 밝기 펄스 (밝기 항등성 붕괴 재현)
                if (px.y < res.y * 0.25) {
                    c *= 0.85 + 0.15 * sin(float(frame) * 0.09);
                }
            }

            // 프레임 카운터: 좌상단 8비트 바이너리 블록 (freeze 감지)
            if (px.y < 24.0 && px.x < 8.0 * 28.0) {
                uint bit = uint(px.x / 28.0);
                bool on = ((frame >> bit) & 1u) == 1u;
                c = on ? float3(1.0, 1.0, 0.0) : float3(0.05);
            }
            // UI 마커 — **영상 프레임은 그대로 두고 화면의 아주 작은 영역만 바꾼다.**
            // 브라우저의 호버 리페인트가 정확히 이 모양이다: 영상은 같은 프레임인데
            // 창 표면이 갱신돼 우리 지문이 "새 프레임"으로 통과시킨다. 프레임을 통째로
            // 진행시키면(초기 재현) 추가 프레임이 진짜 모션을 담아 실제보다 유리해진다.
            if (uiMark.x >= 0.0) {
                float2 d = in.uv * res - uiMark;
                if (dot(d, d) < 900.0) return float4(1.0, 0.0, 0.0, 1.0);
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
        // 디스플레이 틱에서 contentFPS로 프레임 전진 (+옵션 지터)
        let now = link.timestamp
        var threshold = (1.0 / contentFPS) - 0.002
        if jitterMs > 0 {
            // frameIndex 기반 결정적 의사난수 (splitmix) — ±jitterMs
            var z = UInt64(frameIndex) &* 0x9e3779b97f4a7c15
            z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
            let r = Double(z % 2000) / 1000.0 - 1.0
            threshold += r * jitterMs / 1000.0
        }
        if now - lastAdvance >= threshold {
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
        var cm: UInt32 = complexMode ? 1 : 0
        enc.setFragmentBytes(&frame, length: 4, index: 0)
        enc.setFragmentBytes(&res, length: 8, index: 1)
        enc.setFragmentBytes(&cm, length: 4, index: 2)
        var mk = uiMark
        enc.setFragmentBytes(&mk, length: 8, index: 3)
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
        var jitter: Double = 0
        if let i = args.firstIndex(of: "--jitter"), i + 1 < args.count, let j = Double(args[i + 1]) {
            jitter = j
        }
        let complexFlag = args.contains("--complex")
        window = NSWindow(
            contentRect: NSRect(x: px, y: py, width: w, height: h),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false
        )
        window.title = "MacFG Test Pattern"
        view = PatternView(frame: NSRect(x: 0, y: 0, width: w, height: h))
        view.hoverRepaint = args.contains("--hover-repaint")
        view.contentFPS = fps
        view.jitterMs = jitter
        view.complexMode = complexFlag
        window.contentView = view
        window.acceptsMouseMovedEvents = true
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        view.startRendering()

        // 오클루전 상태 변화 로깅 (MacFG 뷰어가 덮었을 때 .visible 유지되는지 검증)
        NotificationCenter.default.addObserver(forName: NSWindow.didChangeOcclusionStateNotification, object: window, queue: .main) { [weak window] _ in
            let vis = window?.occlusionState.contains(.visible) ?? false
            NSLog("[TP-OCC] occlusionState.visible=\(vis)")
        }
        NSLog("[TP-OCC] initial visible=\(window.occlusionState.contains(.visible))")

        // 앱 레벨 이벤트 모니터 — 주입된 클릭이 뷰 이전에 앱에 도달하는지 판별용
        // + CGEvent 필드 덤프: 정상 클릭 vs 주입 클릭의 창-로컬 좌표 필드 규명용
        NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .leftMouseUp, .rightMouseDown]) { ev in
            NSLog("[TP-APPEV] \(ev.type.rawValue) win=\(ev.windowNumber) loc=(\(Int(ev.locationInWindow.x)),\(Int(ev.locationInWindow.y))) clicks=\(ev.clickCount)")
            if ev.type == .leftMouseDown, let cg = ev.cgEvent {
                var dump = "[TP-FIELDS] loc=(\(Int(cg.location.x)),\(Int(cg.location.y)))"
                for f in 0...200 {
                    guard let field = CGEventField(rawValue: UInt32(f)) else { continue }
                    let d = cg.getDoubleValueField(field)
                    let i = cg.getIntegerValueField(field)
                    if d != 0 && abs(d) < 1_000_000 { dump += " f\(f)=\(String(format: "%.1f", d))" }
                    else if i != 0 && abs(i) < 1_000_000 { dump += " f\(f)I=\(i)" }
                }
                NSLog("%@", dump)
            }
            return ev
        }

        // 자체 검증용: N초 후 전체화면/최대화 전환 (보간 유지 테스트)
        if let i = args.firstIndex(of: "--fullscreen-after"), i + 1 < args.count, let sec = Double(args[i + 1]) {
            DispatchQueue.main.asyncAfter(deadline: .now() + sec) { [weak window] in
                window?.toggleFullScreen(nil)
            }
        }
        if let i = args.firstIndex(of: "--zoom-after"), i + 1 < args.count, let sec = Double(args[i + 1]) {
            DispatchQueue.main.asyncAfter(deadline: .now() + sec) { [weak window] in
                window?.zoom(nil)
            }
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
