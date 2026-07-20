import AppKit
import Metal
import MetalKit
import QuartzCore
import Monitoring
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
            float sharpness;  // CAS 강도 0~1 (0=끔, 패스스루 바이트 보존)
        };

        // AMD FidelityFX CAS(Contrast Adaptive Sharpening) 단순화 — LS의 "1:1인데도
        // 선명해지는" 체감의 정체. 로컬 대비가 낮은 곳(브라우저가 늘려놓은 720p의
        // 뭉개진 디테일)을 강하게, 이미 최대 대비인 하드엣지는 약하게 → 헤일로 없음.
        float3 casSharpen(texture2d<float> tex, sampler samp, float2 uv, float2 texel, float sharpness) {
            float3 a = tex.sample(samp, uv + float2( 0, -1) * texel).rgb;
            float3 b = tex.sample(samp, uv + float2(-1,  0) * texel).rgb;
            float3 c = tex.sample(samp, uv).rgb;
            float3 d = tex.sample(samp, uv + float2( 1,  0) * texel).rgb;
            float3 e = tex.sample(samp, uv + float2( 0,  1) * texel).rgb;
            float3 mn = min(min(min(a, b), min(d, e)), c);
            float3 mx = max(max(max(a, b), max(d, e)), c);
            float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-4)));
            float peak = -1.0 / mix(8.0, 5.0, saturate(sharpness));
            float3 w = amp * peak;
            return saturate((c + (a + b + d + e) * w) / (1.0 + 4.0 * w));
        }

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
            if (p.sharpness > 0.01) {
                color.rgb = casSharpen(tex, samp, in.texCoord, 1.0 / p.size, p.sharpness);
            }
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

/// 업스케일 방식 (출력 > 소스일 때). CAS 샤픈은 이와 독립적으로 항상 적용 가능.
public enum UpscaleMode: String, CaseIterable, Identifiable, Sendable {
    case off          // 업스케일 없음 (컴포지터 이중선형)
    case ane          // ANE 신경망 2x만 (≤960 소스, 나머지는 컴포지터가 채움)
    case metalfx      // MetalFX Spatial로 한 번에 목표까지
    case aneMetalfx   // ANE 2x → MetalFX로 마저 (저해상도 소스 최고화질)

    public var id: String { rawValue }
    public var displayName: String {
        switch self {
        case .off: "Off"
        case .ane: "ANE"
        case .metalfx: "MetalFX"
        case .aneMetalfx: "ANE+FX"
        }
    }
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
    /// 스레드 무관 렌더 표면 — 인코딩 경로 전부 (A2: 렌더 스레드가 직접 사용)
    public private(set) var surface: RenderSurface!
    /// 업스케일 방식 (뷰어에서 출력>소스일 때). off면 이중선형.
    public var upscaleMode: UpscaleMode = .off {
        didSet { surface?.update { $0.upscaleMode = upscaleMode } }
    }
    /// CAS 샤프닝 강도 0~1 (0=끔 — 패스스루 바이트 보존 경로 유지). Cover 1:1에서도 유효.
    public var sharpness: Float = 0 {
        didSet { surface?.update { $0.sharpness = sharpness } }
    }
    /// 업스케일/샤픈 실동작 상태 (UI 표시용) — nil = 전부 off
    public var scaleStatus: String? { surface?.scaleStatus }

    /// 뷰어 창을 사용자가 닫았을 때 (X 버튼)
    public var onUserClose: (() -> Void)?

    /// 마우스 역매핑 (뷰어→소스): 소스 창 NS 프레임 + 소유 앱 PID + 창 ID.
    /// 설정되면 뷰어의 호버/클릭/스크롤을 업스케일 배율로 역산해 소스로 전달(CGEventPostToPid).
    /// windowID는 이벤트의 windowUnderMousePointer 필드에 박는다 — 수신 AppKit이 좌표 밑 창을
    /// 윈도우서버에 물으면 우리 뷰어(다른 앱)가 나와 자기 창을 못 찾고 클릭을 버리는 것 우회.
    public var sourceFrameNS: CGRect = .zero { didSet { pushPointerGeometry() } }
    public var sourcePID: pid_t = 0
    public var sourceWindowID: CGWindowID = 0
    private weak var interactionView: ViewerInteractionView?

    /// 정보 오버레이 (단축키 토글) — 좌상단에 소스/보간/업스케일 정보. metal 콘텐츠 위 서브뷰로 합성.
    private var infoLabel: NSTextField?

    /// 정보 오버레이 표시/갱신. nil이면 숨김. (메인 스레드)
    public func setInfoOverlay(_ text: String?) {
        guard let content = window.contentView else { return }
        guard let text, !text.isEmpty else { infoLabel?.isHidden = true; return }
        let label: NSTextField
        if let existing = infoLabel {
            label = existing
        } else {
            label = NSTextField(labelWithString: "")
            label.isEditable = false; label.isSelectable = false
            label.isBezeled = false; label.drawsBackground = true
            label.backgroundColor = NSColor.black.withAlphaComponent(0.55)
            label.textColor = .white
            label.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .medium)
            label.maximumNumberOfLines = 0
            label.lineBreakMode = .byWordWrapping
            label.wantsLayer = true
            label.layer?.cornerRadius = 6
            label.layer?.masksToBounds = true
            label.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(label)
            NSLayoutConstraint.activate([
                label.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
                label.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            ])
            infoLabel = label
        }
        // 인셋 여백 — attributed로 좌우 패딩
        let para = NSMutableParagraphStyle()
        para.firstLineHeadIndent = 8; para.headIndent = 8; para.tailIndent = -8
        para.lineSpacing = 2; para.paragraphSpacingBefore = 6; para.paragraphSpacing = 6
        label.attributedStringValue = NSAttributedString(string: text, attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: NSColor.white,
            .paragraphStyle: para,
        ])
        label.isHidden = false
        content.addSubview(label, positioned: .above, relativeTo: nil)   // 항상 최상단
    }

    /// 상대커서 모드 — 진짜 커서 위치를 소스로 재타깃 (호버/스크롤/클릭/드래그 전부 진짜 이벤트)
    private let relativePointer = RelativePointer()

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
            // 보더리스 전체화면 뷰어 — 소스 화면 전체(메뉴바·Dock 포함)를 덮는다.
            // 네이티브 전체화면(초록 버튼, 자기 Space)은 안 씀: 정지 시 검정 Space 잔존 +
            // Space 전환이 실시간 캡처 페이싱을 무너뜨림(실측). 대신 shielding 레벨 보더리스로
            // 메뉴바·Dock 위를 즉시(애니메이션·Space 없이) 덮어 초록버튼 전체화면과 같은 화면을 낸다.
            window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 960, height: 540),
                styleMask: [.borderless],
                backing: .buffered,
                defer: false
            )
            window.title = title
            // 메뉴바(24)·Dock(20) 위로 — 화면 전체를 덮음. Firefox PiP(.floating 3)도 당연히 아래.
            window.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
            window.isOpaque = true
            window.backgroundColor = .black
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

            // 마우스 이벤트를 소스로 포워딩하는 인터랙션 뷰 (호버/클릭/스크롤 역매핑)
            let iview = ViewerInteractionView(frame: window.contentView?.bounds ?? .zero)
            iview.wantsLayer = true
            iview.layer?.backgroundColor = NSColor.black.cgColor
            iview.layer?.addSublayer(metalLayer)
            iview.autoresizingMask = [.width, .height]
            window.contentView = iview
            self.interactionView = iview
        }

        // .readOnly: 스크린샷/녹화에 출력이 보이도록 (검증 및 사용자 녹화용).
        // 캡처는 SCK desktopIndependentWindow(대상 창 백킹만 캡처)라 자기 캡처 루프가 생기지 않는다.
        window.sharingType = .readOnly
        // 창 수명은 OverlayWindow가 강한 참조(let window)로 소유. close()가 창을 자동
        // 해제하면 dealloc 시 또 해제 → 이중 해제(EXC_BAD_ACCESS, 풀 드레인에서 objc_release).
        // cover(.borderless)는 기본 true였어서 단축키 정지 시 크래시했음 → 두 스타일 모두 false.
        window.isReleasedWhenClosed = false

        self.window = window
        self.metalLayer = metalLayer
        super.init()

        self.surface = RenderSurface(
            device: device, metalLayer: metalLayer,
            pipelineState: ps, sampler: s,
            params: RenderSurface.Params(
                isViewer: style == .viewer,
                sharpness: 0, upscaleMode: .off,
                contentBounds: CGRect(origin: .zero, size: window.frame.size),
                contentsScale: NSScreen.main?.backingScaleFactor ?? 2.0,
                cornerRadiusPt: OverlayStyleConstants.cornerRadius
            )
        )

        if style == .viewer {
            window.delegate = self
            window.acceptsMouseMovedEvents = true
            interactionView?.owner = self
        }

        logger.info("OverlayWindow created (style=\(String(describing: style)))")
    }

    // MARK: - 상대커서 모드 (진짜 커서 위치를 소스로 재타깃 — 링·숨김 없음)

    private var primaryScreenHeight: CGFloat {
        NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height
            ?? NSScreen.main?.frame.height ?? 0
    }

    /// 탭 스레드용 지오메트리 스냅샷을 메인에서 계산해 push (전부 값 타입 → 스레드 안전).
    /// sourceFrameNS 갱신(15Hz)·진입 시 호출 — 탭은 AppKit을 만지지 않고 이 스냅샷만 읽는다.
    private func pushPointerGeometry() {
        guard style == .viewer, relativePointer.active else { return }
        let displayID = (window.screen?.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
            ?? CGMainDisplayID()
        relativePointer.setGeometry(RelativePointer.Geometry(
            displayID: displayID,
            windowFrame: window.frame,
            letterbox: metalLayer.frame,
            sourceFrameNS: sourceFrameNS,
            primaryHeight: primaryScreenHeight,
            sourcePID: sourcePID
        ))
    }

    /// 뷰어 표시 시작 시 상대커서 진입 — 뷰어를 클릭투과로 만들고 커서를 소스에 상주.
    private func enterRelativePointer() {
        guard style == .viewer, !relativePointer.active else { return }
        window.ignoresMouseEvents = true            // 재게시 이벤트가 뷰어 통과 → 소스 도달
        // 소스를 키 창으로 — 비활성 창은 mouseMoved 로컬좌표가 붕괴해 호버가 죽음(실측 재확인).
        if sourcePID != 0 { NSRunningApplication(processIdentifier: sourcePID)?.activate() }
        // 옛 postToPid 포워딩 경로 차단 — 진짜 커서가 뷰어 위(fullscreen)에 있어 ViewerInteractionView
        // 트래킹영역이 발화하면 이벤트가 한 번 더 매핑·전달(이중 배달, 실측). 탭이 전담하므로 무력화.
        interactionView?.owner = nil
        relativePointer.enable()
        // 탭 생성 실패(접근성 미허가 등)면 클릭투과만 켜진 채 포워딩이 없어, 뷰어 위 클릭이
        // 아래 임의 창으로 새어 나간다 (리뷰 확정). 레거시 postToPid 경로로 되돌린다.
        guard relativePointer.active else {
            window.ignoresMouseEvents = false
            interactionView?.owner = self
            DiagnosticLog.shared.log("[RELPTR] tap 생성 실패 → 클릭투과 해제 + 레거시 포워딩 복원 (접근성 권한 확인 필요)")
            return
        }
        pushPointerGeometry()   // 초기 스냅샷 (이후 sourceFrameNS 갱신마다 push)
    }

    /// 뷰어 숨김/정지 시 상대커서 이탈 — 클릭투과 해제.
    private func exitRelativePointer() {
        guard style == .viewer else { return }
        relativePointer.disable()
        window.ignoresMouseEvents = false
        interactionView?.owner = self   // 옛 경로 복원 (폴백용)
    }

    // MARK: - 마우스 역매핑 (뷰어 → 소스) — 레거시 postToPid 경로(상대커서 모드에선 미사용)

    /// 뷰어 창 좌표(NS)의 마우스 위치를 소스 좌표로 역산.
    /// 반환: (global: 스크린 CG top-left, local: 소스 창 내부 top-left).
    /// windowNumber(필드51)를 박은 이벤트는 location을 창-로컬로 해석하므로 local이 필요.
    private func mapToSource(_ locInWindow: CGPoint) -> (global: CGPoint, local: CGPoint)? {
        guard style == .viewer, sourceFrameNS.width > 1, sourceFrameNS.height > 1 else { return nil }
        let lb = metalLayer.frame   // 레터박스 (contentView=window content 좌표, NS bottom-left)
        guard lb.width > 1, lb.height > 1 else { return nil }
        let nx = (locInWindow.x - lb.minX) / lb.width
        let ny = (locInWindow.y - lb.minY) / lb.height        // NS: 0=하단
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return nil }   // 레터박스 밖은 무시
        // 소스 창 NS 스크린 좌표 (소스 로컬 → 스크린)
        let sxNS = sourceFrameNS.minX + nx * sourceFrameNS.width
        let syNS = sourceFrameNS.minY + ny * sourceFrameNS.height
        // NS(bottom-left) → CG(top-left): 주 스크린 높이 기준 y 반전
        let primaryH = NSScreen.screens.first(where: { $0.frame.origin == .zero })?.frame.height
            ?? NSScreen.main?.frame.height ?? 0
        let global = CGPoint(x: sxNS, y: primaryH - syNS)
        let local = CGPoint(x: nx * sourceFrameNS.width, y: (1 - ny) * sourceFrameNS.height)
        return (global, local)
    }

    private let mouseEventSource = CGEventSource(stateID: .hidSystemState)
    private var mouseLogCount = 0
    /// 호버 전달 — 전역 좌표 + 창 필드 없이 postToPid. 키 창(소스는 캡처 시작 때 활성화됨)
    /// 라우팅으로 정확한 좌표로 도달(실측). ⚠️ f51(창번호)을 붙이면 오히려 locationInWindow가
    /// (0,height)로 붕괴해 호버가 구석 좌표로 감 — 창 필드는 moved에 쓰지 말 것.
    fileprivate func forwardMouse(_ e: NSEvent, type: CGEventType, button: CGMouseButton = .left) {
        guard sourcePID != 0, let m = mapToSource(e.locationInWindow),
              let ev = CGEvent(mouseEventSource: mouseEventSource, mouseType: type,
                               mouseCursorPosition: m.global,
                               mouseButton: button) else { return }
        ev.postToPid(sourcePID)
        if type != .mouseMoved || mouseLogCount < 3 {
            mouseLogCount += 1
            DiagnosticLog.shared.log("[MOUSE] \(type.rawValue) viewer=(\(Int(e.locationInWindow.x)),\(Int(e.locationInWindow.y))) → PID\(sourcePID) global(\(Int(m.global.x)),\(Int(m.global.y)))")
        }
    }

    /// 클릭 패스스루 리플레이 — postToPid 클릭은 창-로컬 위치(윈도우서버가 라우팅 중 채우는
    /// 내부 구조체)가 비어 수신 AppKit이 (0,height)로 해석·폐기함(실측). 대신:
    /// 뷰어를 이벤트 투과로 전환 → 커서를 소스 위치로 워프 → 진짜 클릭(세션 탭, 정상 라우팅)
    /// → 커서/투과 복구. 커서가 ~150ms 소스 위치로 깜빡이는 비용으로 100% 정상 배달.
    private var passthroughActive = false
    fileprivate func passthroughClick(_ e: NSEvent, button: CGMouseButton) {
        // 재진입 가드 — ignoresMouseEvents 반영 전(윈도우서버 비동기)에 자기 클릭이 뷰어에
        // 되튕겨 재귀 발화하던 것 차단 (실측: 30ms 내 재진입)
        guard !passthroughActive, let m = mapToSource(e.locationInWindow) else { return }
        passthroughActive = true
        let returnPos = CGEvent(source: nil)?.location ?? m.global   // 현재 실제 커서(CG)
        let target = m.global
        window.ignoresMouseEvents = true
        NSRunningApplication(processIdentifier: sourcePID)?.activate()
        CGWarpMouseCursorPosition(target)
        let downType: CGEventType = button == .right ? .rightMouseDown : .leftMouseDown
        let upType: CGEventType = button == .right ? .rightMouseUp : .leftMouseUp
        // ignore 전파 여유(80ms) 후 진짜 클릭 — 윈도우서버가 뷰어를 건너뛰고 소스로 정상 라우팅
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in
            if let d = CGEvent(mouseEventSource: self?.mouseEventSource, mouseType: downType, mouseCursorPosition: target, mouseButton: button) {
                d.setIntegerValueField(.mouseEventClickState, value: 1)
                d.post(tap: .cgSessionEventTap)
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.14) { [weak self] in
            if let u = CGEvent(mouseEventSource: self?.mouseEventSource, mouseType: upType, mouseCursorPosition: target, mouseButton: button) {
                u.setIntegerValueField(.mouseEventClickState, value: 1)
                u.post(tap: .cgSessionEventTap)
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) { [weak self] in
            guard let self else { return }
            CGWarpMouseCursorPosition(returnPos)
            // 이 0.22s 사이에 상대커서가 진입했다면 클릭투과는 그쪽이 소유한 상태 — 여기서
            // 되돌리면 재게시 이벤트가 뷰어에 막혀 뷰어 입력이 통째로 죽는다 (리뷰 확정).
            if !self.relativePointer.active { self.window.ignoresMouseEvents = false }
            self.passthroughActive = false
        }
        DiagnosticLog.shared.log("[MOUSE] passthrough click at CG(\(Int(target.x)),\(Int(target.y))) → return(\(Int(returnPos.x)),\(Int(returnPos.y)))")
    }

    fileprivate func forwardScroll(_ e: NSEvent) {
        guard sourcePID != 0, let m = mapToSource(e.locationInWindow),
              let ev = CGEvent(scrollWheelEvent2Source: mouseEventSource, units: .pixel, wheelCount: 2,
                               wheel1: Int32(e.scrollingDeltaY), wheel2: Int32(e.scrollingDeltaX), wheel3: 0) else { return }
        // moved와 동일: 전역 좌표 + 창 필드 없이 (f51은 좌표 붕괴 유발 — 위 참조)
        ev.location = m.global
        ev.postToPid(sourcePID)
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
    /// 오클루전 우회 — 이 창이 소스를 완전히 덮을 때, 소스가 occluded로 마킹돼
    /// 렌더링을 멈추는 것(Firefox PiP 등 가려진 창 페인팅 중단 → 캡처 정지 화면)을 막는다.
    /// alpha 0.999면 창이 "불투명 오클루더"가 아니게 돼 아래 창이 계속 그려짐. cover·전체화면 뷰어 공통.
    public func setOcclusionBypass(_ enabled: Bool) {
        window.alphaValue = enabled ? 0.999 : 1.0
    }

    /// 텍스처를 외부 commandBuffer에 렌더 인코딩. drawable 반환 — 호출자가 present + commit.
    /// (실제 인코딩은 RenderSurface — 스레드 무관. 여기는 하위호환 위임)
    public func encodeRender(texture: any MTLTexture, into commandBuffer: any MTLCommandBuffer) -> (any CAMetalDrawable)? {
        surface.encode(texture: texture, into: commandBuffer)
    }

    /// 창 기하/화면 변경을 surface 캐시에 반영 (메인) — 뷰어 레터박스 기준/배율.
    /// drawableSize가 0이면 시딩 — CAMetalDisplayLink는 0×0 레이어에선 영영 발화하지 않는다
    /// (기존 nextDrawable 경로는 첫 encode에서 lazy 설정이라 문제 없었음).
    public func refreshSurfaceParams() {
        let bounds = window.contentView?.bounds ?? CGRect(origin: .zero, size: window.frame.size)
        let scale = window.screen?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        surface.update {
            $0.contentBounds = bounds
            $0.contentsScale = scale
        }
        if metalLayer.drawableSize.width < 1 || metalLayer.drawableSize.height < 1 {
            metalLayer.drawableSize = CGSize(
                width: max(bounds.width * scale, 64),
                height: max(bounds.height * scale, 64)
            )
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
        refreshSurfaceParams()
    }

    /// 뷰어 초기 위치/크기 (1회)
    public func setInitialViewerFrame(_ frame: CGRect) {
        guard style == .viewer else { return }
        window.setFrame(frame, display: true)
        refreshSurfaceParams()
    }

    /// 화면 구성이 바뀌었을 때 뷰어를 화면 안으로 되돌린다 — 뷰어 프레임은 생성 시 1회만
    /// 설정되므로, 놓여 있던 디스플레이가 분리되거나 해상도가 줄면 창이 화면 밖에 남아
    /// 복구 경로가 없었다 (리뷰 확정). 이미 화면 안이면 건드리지 않아 사용자 배치를 보존.
    public func ensureViewerOnScreen() {
        guard style == .viewer, !window.styleMask.contains(.fullScreen) else { return }
        let frame = window.frame
        let visible = (window.screen ?? NSScreen.main)?.visibleFrame
            ?? NSScreen.screens.first?.visibleFrame
        guard let visible else { return }
        // 창의 상당 부분이 어느 화면에도 안 겹치면 재배치
        let onAnyScreen = NSScreen.screens.contains { $0.visibleFrame.intersects(frame) }
        guard !onAnyScreen else { return }
        var f = frame
        f.size.width = min(f.width, visible.width)
        f.size.height = min(f.height, visible.height)
        f.origin.x = visible.midX - f.width / 2
        f.origin.y = visible.midY - f.height / 2
        window.setFrame(f, display: true)
        refreshSurfaceParams()
        DiagnosticLog.shared.log("[VIEWER] 화면 구성 변경 → 뷰어 창을 화면 안으로 재배치")
    }

    /// 창을 확실히 닫는다 (정지 시). 전체화면 상태면 먼저 빠져나와 검정 Space 잔존 방지.
    public func close() {
        // 프로그램적 정지 — 델리게이트를 먼저 떼어 windowWillClose→onUserClose→stopCapture
        // 재진입을 차단 (이건 사용자가 X로 닫은 게 아님).
        exitRelativePointer()   // 커서 디커플/숨김 반드시 복구
        onUserClose = nil
        window.delegate = nil
        if window.styleMask.contains(.fullScreen) {
            window.toggleFullScreen(nil)
        }
        window.orderOut(nil)
        window.close()
    }

    /// 표시/숨김
    public func setVisible(_ visible: Bool) {
        if visible {
            window.orderFront(nil)
            if style == .viewer {
                // 첫 프레임 레이아웃(레터박스 확정) 후 진입 — 매핑 준비 완료 보장
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
                    guard let self, self.window.isVisible else { return }
                    self.enterRelativePointer()
                }
            }
        } else {
            exitRelativePointer()
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

    /// 창 크기가 바뀌면 서피스 파라미터(contentBounds/배율/드로어블)를 다시 읽는다.
    /// 없으면 contentBounds가 옛 크기에 머물러, 레터박스 fit이 그 작은 영역 안에서 계산되고
    /// 콘텐츠가 좌하단에 몰린다 — 전체화면 자동 전환(창이 커짐)에서 오른쪽·위쪽 패딩으로 발현.
    public func windowDidResize(_ notification: Notification) {
        refreshSurfaceParams()
        pushPointerGeometry()   // 상대커서 매핑 기준(레터박스)도 함께 갱신
    }

    /// 전체화면 진입/이탈은 리사이즈 통지가 애니메이션 도중 값으로 올 수 있어 완료 시점에 재갱신
    public func windowDidEnterFullScreen(_ notification: Notification) { refreshSurfaceParams(); pushPointerGeometry() }
    public func windowDidExitFullScreen(_ notification: Notification)  { refreshSurfaceParams(); pushPointerGeometry() }

    /// 다른 배율의 화면으로 옮겨가면 contentsScale이 달라진다
    public func windowDidChangeScreen(_ notification: Notification) { refreshSurfaceParams() }
}

/// 뷰어 contentView — 호버/클릭/스크롤을 OverlayWindow가 소스로 역매핑·전달하게 넘긴다.
/// 뷰어가 이벤트를 받아 먹기만 하던 것(조작 불가)을 소스로 포워딩. metalLayer는 서브레이어.
final class ViewerInteractionView: NSView {
    weak var owner: OverlayWindow?
    private var tracking: NSTrackingArea?

    override var isFlipped: Bool { false }   // NS bottom-left 유지 (매핑이 이 기준)
    override var acceptsFirstResponder: Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let tracking { removeTrackingArea(tracking) }
        let t = NSTrackingArea(rect: bounds, options: [.activeAlways, .mouseMoved, .inVisibleRect],
                               owner: self, userInfo: nil)
        addTrackingArea(t); tracking = t
    }

    // 호버 = postToPid moved(키 창 라우팅으로 정상 도달) · 클릭 = 패스스루 리플레이(진짜 이벤트)
    override func mouseMoved(with e: NSEvent)   { owner?.forwardMouse(e, type: .mouseMoved) }
    override func mouseDown(with e: NSEvent)    { owner?.passthroughClick(e, button: .left) }
    override func mouseUp(with e: NSEvent)      { }   // 클릭은 down에서 down+up 시퀀스로 처리
    override func mouseDragged(with e: NSEvent) { }
    override func rightMouseDown(with e: NSEvent) { owner?.passthroughClick(e, button: .right) }
    override func rightMouseUp(with e: NSEvent)   { }
    override func scrollWheel(with e: NSEvent)  { owner?.forwardScroll(e) }
}
