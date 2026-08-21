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

    /// 커버를 **완전 불투명**으로 두고, 대신 소스를 1pt 덜 덮어 오클루전 판정을 피한다.
    ///
    /// 지금 커버는 블렌딩을 세 겹으로 강요한다 — `window.isOpaque=false`,
    /// `metalLayer.isOpaque=false`(둥근 모서리 SDF), `alphaValue=0.999`(오클루전 우회).
    /// 그 대가가 실측됐다: 캡처 중 **WindowServer 77%**(최대 87%)로 기준선 43%의 두 배이고,
    /// 우리 프로세스는 31.7%에 불과하다. 그리고 미표시(present했는데 화면에 안 나옴)의 원인이
    /// 우리 쪽이 아님도 확정됐다 — GPU는 **100%** 목표 슬롯보다 일찍 끝나고(오버런 0건),
    /// 목표 슬롯 중복도 미표시 1789/s 중 5/s뿐이다. 제때 만들어 넘긴 프레임을 컴포지터가 버린다.
    ///
    /// alpha<1은 두 가지를 동시에 강요한다: (1) 우리 레이어를 매 프레임 블렌딩하고,
    /// (2) 우리가 불투명 오클루더가 아니므로 **가려진 아래 창들도 계속 그리게** 한다.
    /// 둘 다 픽셀 수에 비례해서 4K가 1080p의 4배다(붕괴 확률 1.5% vs 36.5%, 위험비 24배).
    ///
    /// 그런데 오클루전 판정은 "완전히 덮였는가"라서, **1pt만 덜 덮으면** 소스는 계속 렌더링하고
    /// 우리는 불투명일 수 있다. 그러면 컴포지터는 블렌딩도 건너뛰고 아래 창도 건너뛴다.
    /// 대가: 창 위쪽에 1pt 실소스가 보이고(브라우저면 툴바 영역), 모서리 SDF가 알파를 못 쓰므로
    /// 둥근 모서리가 검은 직각이 된다.
    static let opaqueCover = Knob.string("MACFG_OPAQUECOVER") == "1"

    /// 드로어블 backing 해상도 배율 (1.0 = 지금대로 창 크기 × 배율).
    ///
    /// **남은 단 하나의 변수를 가르기 위한 것.** 지금까지 배제된 것:
    /// 우리 GPU(슬롯 예산의 35%, 오버런 0건), 렌더 스레드 CPU(코어의 4.7%),
    /// 보간 연산 자체(blend 엔진으로 1/15로 줄여도 표시가 95.5 → 100.6/s로 오차 수준),
    /// 드로어블 풀(poolMiss=0), 목표 슬롯 중복(미표시 1789/s 중 5/s), 반투명 합성(OPAQUECOVER 무효).
    /// 남은 것은 **초당 114장의 8.0MP 표면을 컴포지터에 넘기는 행위 자체**다
    /// (한 실행 안 r(WindowServer CPU, 미표시) = +0.858).
    ///
    /// 0.5로 두면 장수는 그대로인데 픽셀은 1/4이 된다. 그래도 떨어지면 원인은 픽셀 처리량이
    /// 아니라 **present 횟수당 고정비**이고, 나아지면 픽셀 처리량이다. 둘은 해결책이 정반대다 —
    /// 전자는 장수를 줄여야 하고 후자는 해상도를 낮추면 된다.
    /// 화질은 컴포지터 업스케일만큼 나빠진다(진단용).
    static let drawScale = min(max(Knob.double("MACFG_DRAWSCALE") ?? 1.0, 0.25), 1.0)
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

    /// 이 오버레이/뷰어 창의 CGWindowID — 디스플레이 캡처 시 자기 출력을 되먹지 않도록
    /// 제외 목록에 넣기 위해 필요 (안 빼면 무한 거울). 미실현 창은 0/음수라 0을 반환.
    public var cgWindowID: CGWindowID {
        let n = window.windowNumber
        guard n > 0, n <= Int(UInt32.max) else { return 0 }
        return CGWindowID(n)
    }
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
    public var sourceWindowID: CGWindowID = 0

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
        // 이미 최상단이면 다시 넣지 않는다. 이 함수는 0.5초마다 불리는데, addSubview는
        // 이미 붙어 있는 뷰라도 **레이어 트리를 재정렬**한다 — 4K 창에서 초당 두 번 컴포지터를
        // 건드릴 이유가 없다. (렌더 런루프 점유가 vsync 콜백을 버리게 만드는 게 확인된 상황이라,
        //  주기적이고 불필요한 컴포지터 작업은 지운다.)
        if label.superview !== content || content.subviews.last !== label {
            content.addSubview(label, positioned: .above, relativeTo: nil)   // 항상 최상단
        }
    }

    /// 상대커서 모드 — 진짜 커서 위치를 소스로 재타깃 (호버/스크롤/클릭/드래그 전부 진짜 이벤트)

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
            window.isOpaque = OverlayStyleConstants.opaqueCover
            window.backgroundColor = OverlayStyleConstants.opaqueCover ? .black : .clear
            // **마우스 이벤트를 삼킨다.** 뷰어 마우스 재매핑은 2026-08-20에 제거됐다 —
            // 크래시 3계통의 근원이었고, 소스가 이벤트를 받아 호버 리페인트를 하면 입력
            // 케이던스가 오염돼 보간 예산이 좁아진다(worklog 참조). 통과시킬 이유가 없어졌다.
            // 소스를 조작하려면 단축키로 오버레이를 숨긴다.
            window.ignoresMouseEvents = false
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            window.hasShadow = false
            window.alphaValue = 1.0

            // 모서리 마스킹은 셰이더 SDF로 수행 (CALayer.cornerRadius는 CAMetalLayer
            // 직접 스캔아웃에서 무시됨 — 실측). 컴포지터가 알파를 쓰도록 비불투명 레이어.
            // opaqueCover에서는 그 알파를 포기한다 — 모서리가 검은 직각이 되는 대신
            // 컴포지터가 이 레이어를 블렌딩 없이 스캔아웃한다.
            metalLayer.isOpaque = OverlayStyleConstants.opaqueCover

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
            // 레벨은 노브로 바꿀 수 있다 (측정용). 기본은 shielding — 메뉴바(24)·Dock(20) 위를 덮기 위함.
            //   defaults write com.macfg.MacFG env.MACFG_VIEWERLEVEL -string 1000   (screenSaver)
            // **레벨은 전체화면 Space 문제의 원인이 아니다 — 2026-08-06에 기각됐다.**
            // shielding(2147483628)과 screenSaver(1000) 둘 다 onActiveSpace=false로 같았다.
            // 이 노브는 그 기각을 재확인하는 용도로만 남는다. 자세한 것은 setVisible의 주석 참조.
            window.level = NSWindow.Level(rawValue: Knob.int("MACFG_VIEWERLEVEL") ?? Int(CGShieldingWindowLevel()))
            window.isOpaque = true
            window.backgroundColor = .black
            window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

            // 마우스 이벤트를 소스로 포워딩하는 인터랙션 뷰 (호버/클릭/스크롤 역매핑)
            let iview = NSView(frame: window.contentView?.bounds ?? .zero)
            iview.wantsLayer = true
            iview.layer?.backgroundColor = NSColor.black.cgColor
            iview.layer?.addSublayer(metalLayer)
            iview.autoresizingMask = [.width, .height]
            window.contentView = iview
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
                // opaqueCover면 모서리 SDF를 끈다. 불투명 레이어에서는 알파가 무시되고
                // rgb만 0으로 곱해져 모서리가 **검게** 남는다 — 마스킹을 아예 안 하면
                // 직각이 되는 대신 소스 픽셀이 그대로 보여 그쪽이 덜 눈에 띈다.
                cornerRadiusPt: OverlayStyleConstants.opaqueCover ? 0 : OverlayStyleConstants.cornerRadius
            )
        )

        if style == .viewer {
            window.delegate = self
        }

        logger.info("OverlayWindow created (style=\(String(describing: style)))")
    }

    /// 뷰어 기하 로그 중복 억제 키 — 마우스 재매핑 제거 후 이 블록에서 유일한 잔존 사용처.
    private var lastLoggedViewerFrame = ""

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
        // MACFG_NOOCCBYPASS=1: 우회를 끄고 완전 불투명으로 둔다. **측정 전용.**
        // 가르려는 것: 표시 천장(~48장/s)이 반투명 합성 비용 때문인가.
        // alpha < 1.0이면 window server가 이 창을 불투명 오클루더로 보지 않아 아래 창들을
        // 계속 그리고 매 프레임 블렌딩한다. 불투명이면 그 아래를 통째로 건너뛸 수 있다.
        // 부작용(의도됨): 소스 창이 occluded 판정돼 렌더링을 멈춘다 — 그래서 이 측정은
        // MACFG_ALWAYSPRESENT=1과 함께 써서 마지막 프레임을 계속 내보내며 효율만 본다.
        if Knob.string("MACFG_NOOCCBYPASS") == "1" || OverlayStyleConstants.opaqueCover {
            // opaqueCover는 alpha 대신 **기하**로 오클루전을 피한다(updateFrame의 1pt 인셋).
            window.alphaValue = 1.0
            return
        }
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
        if style == .viewer {
            // 창이 요청한 프레임대로 놓였는지(macOS constrainFrameRect가 줄이거나 옮기지 않았는지)
            // 확인용 — 뷰어 여백 진단의 출발점.
            let scr = window.screen?.frame ?? .zero
            DiagnosticLog.shared.log(String(
                format: "[VIEWER-WIN] frame=(%.0f,%.0f %.0fx%.0f) content=%.0fx%.0f screen=(%.0f,%.0f %.0fx%.0f) scale=%.1f",
                window.frame.minX, window.frame.minY, window.frame.width, window.frame.height,
                bounds.width, bounds.height, scr.minX, scr.minY, scr.width, scr.height, scale))
        }
        surface.update {
            $0.contentBounds = bounds
            $0.contentsScale = scale
        }
        let ds = OverlayStyleConstants.drawScale
        if ds < 0.999 {
            // 배율이 걸리면 시딩 조건(<1)이 아니라 **매번** 맞춘다 — 다른 경로가 드로어블을
            // 풀사이즈로 되돌려도 다음 갱신에서 되잡는다.
            let w = max(bounds.width * scale * ds, 64)
            let h = max(bounds.height * scale * ds, 64)
            if abs(metalLayer.drawableSize.width - w) > 1 || abs(metalLayer.drawableSize.height - h) > 1 {
                metalLayer.drawableSize = CGSize(width: w, height: h)
                DiagnosticLog.shared.log("[DRAWSCALE] 드로어블 \(Int(w))x\(Int(h)) (배율 \(ds))")
            }
        } else if metalLayer.drawableSize.width < 1 || metalLayer.drawableSize.height < 1 {
            metalLayer.drawableSize = CGSize(
                width: max(bounds.width * scale, 64),
                height: max(bounds.height * scale, 64)
            )
        }
    }

    /// 오버레이 위치/크기 갱신 (overlay 스타일 전용 — 뷰어는 사용자가 제어)
    public func updateFrame(_ frame: CGRect) {
        guard style == .overlay else { return }
        var frame = frame
        if OverlayStyleConstants.opaqueCover, frame.height > 2 {
            // 소스를 1pt 덜 덮어 "완전히 가려짐" 판정을 피한다 — 그래야 소스 앱이 계속
            // 렌더링한다(alpha<1이 하던 일). AppKit 원점은 좌하단이라 height를 줄이면
            // 위쪽에 1pt가 남는다. 브라우저면 툴바 영역이라 눈에 띄지 않는다.
            frame.size.height -= 1
        }
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
        // macOS는 setFrame을 constrainFrameRect로 제약할 수 있다(메뉴바 아래로 밀기 등).
        // 요청과 결과가 다르면 그대로 두면 화면 일부가 안 덮인다 — 어긋나면 기록하고,
        // 화면 전체를 요구한 경우에는 제약을 우회해 다시 설정한다.
        if abs(window.frame.width - frame.width) > 1 || abs(window.frame.height - frame.height) > 1
            || abs(window.frame.minX - frame.minX) > 1 || abs(window.frame.minY - frame.minY) > 1 {
            DiagnosticLog.shared.log(String(
                format: "[VIEWER-WIN] 요청=(%.0f,%.0f %.0fx%.0f) → 제약됨=(%.0f,%.0f %.0fx%.0f) — 재설정 시도",
                frame.minX, frame.minY, frame.width, frame.height,
                window.frame.minX, window.frame.minY, window.frame.width, window.frame.height))
            window.setFrameOrigin(frame.origin)
            window.setContentSize(frame.size)
            window.setFrameOrigin(frame.origin)
        }
        refreshSurfaceParams()
    }

    /// 뷰어 창 기하가 바뀌었으면 기록 (추적 틱에서 호출) — setFrame 직후가 아니라 나중에
    /// 윈도우서버가 창을 옮기거나 줄이는 경우를 잡기 위한 감시.
    public func logViewerGeometryIfChanged() {
        guard style == .viewer else { return }
        let f = window.frame
        let key = "\(Int(f.minX)),\(Int(f.minY)),\(Int(f.width)),\(Int(f.height))"
        guard key != lastLoggedViewerFrame else { return }
        lastLoggedViewerFrame = key
        let scr = window.screen?.frame ?? .zero
        DiagnosticLog.shared.log(String(
            format: "[VIEWER-WIN*] frame=(%.0f,%.0f %.0fx%.0f) content=%.0fx%.0f screen=(%.0f,%.0f %.0fx%.0f)",
            f.minX, f.minY, f.width, f.height,
            window.contentView?.bounds.width ?? 0, window.contentView?.bounds.height ?? 0,
            scr.minX, scr.minY, scr.width, scr.height))
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
            // **orderFrontRegardless** — orderFront가 아니다.
            // MacFG는 LSUIElement 백그라운드 앱이라 스스로 활성화되지 않는다. 그 상태에서
            // orderFront(nil)는 "우리 앱이 활성화될 때" 앞으로 나오는 것으로 지연될 수 있고,
            // 다른 앱이 전체화면 Space를 쥐고 있으면 창이 데스크톱 Space에 남는다.
            // 실측 2026-08-06(소스 전체화면): isVisible=true인데 onActiveSpace=false,
            // occlusion=hidden → 화면에 안 보이고 CAMetalDisplayLink도 발화하지 않아
            // 파이프라인 전체가 조용히 멈췄다([DRIVER] update 0줄, interpEnc=0).
            // Regardless는 앱 활성화와 무관하게 즉시 올린다 — 전체화면 위에 뜨는 유틸리티들이
            // 쓰는 방식이다.
            window.orderFrontRegardless()
            // **창이 실제로 보이는 Space에 올라갔는지 확인한다.**
            // 소스가 macOS 전체화면(자기 Space)이면 우리 창이 데스크톱 Space에 남아
            // 화면에 안 나오고, 그러면 CAMetalDisplayLink도 발화하지 않아 파이프라인 전체가
            // 조용히 멈춘다([DRIVER] update 0줄, [SCHED] 0줄 — 실측 2026-08-06).
            // collectionBehavior에 canJoinAllSpaces가 걸려 있는데도 그렇다면 그 사실 자체가
            // 필요한 정보다. orderFront 직후는 아직 반영 전이라 한 박자 뒤에 본다.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                guard let self else { return }
                let occ = self.window.occlusionState.contains(.visible) ? "visible" : "hidden"
                DiagnosticLog.shared.log(
                    "[WIN] style=\(self.style == .viewer ? "viewer" : "cover") "
                    + "isVisible=\(self.window.isVisible) onActiveSpace=\(self.window.isOnActiveSpace) "
                    + "occlusion=\(occ) level=\(self.window.level.rawValue) behavior=\(self.window.collectionBehavior.rawValue) "
                    + "screen=\(self.window.screen?.localizedName ?? "nil") frame=\(self.window.frame)")
                // **활성 Space에 못 올라갔으면 조용히 두지 않는다.**
                //
                // 다른 앱이 macOS 전체화면(자기 Space)을 쥐고 있으면 우리 출력 창은 그 Space에
                // 올라가지 못한다. 그러면 화면에 아무것도 안 보이고, 레이어가 화면에 없으니
                // CAMetalDisplayLink도 발화하지 않아 파이프라인 전체가 침묵한다
                // ([DRIVER] update 0줄, interpEnc=0). 증상만 보면 "보간이 갑자기 안 된다"이다.
                //
                // **원인은 창 속성이 아니라 activation policy였다 (2026-08-07 확정).**
                //
                // 2026-08-06에 창 속성 세 가지를 시험했고 전부 실패했다:
                //   · orderFront(nil) → orderFrontRegardless()
                //   · 창 레벨 shielding(2147483628) → screenSaver(1000)
                //   · collectionBehavior 재적용 후 재-orderFront
                // collectionBehavior는 내내 257(canJoinAllSpaces|fullScreenAuxiliary)로 살아 있었다.
                // 그래서 "창 속성으로는 넘을 수 없는 벽"이라고 결론냈는데, **틀렸다.**
                //
                // 그 측정들은 전부 `s.menubaronly = false` 상태에서 이뤄졌다 — 자동화 편의로 꺼둔
                // 테스트 설정이고 되돌리는 걸 잊었다. 그 값은 AppState가 앱을 `.regular`로 올리고
                // `activate(ignoringOtherApps: true)`까지 부르게 한다. 그 상태에서는 우리 창이
                // 다른 앱의 전체화면 Space에 올라가지 못한다.
                //
                // `s.menubaronly = true`(=기본값, LSUIElement 상주)로 되돌리고 같은 재현을 하면:
                //   onActiveSpace=true / occlusion=visible / SCShareableContent 목록에도 등장
                //   ("제외 창 0/2개" → "1/2개") / CAMetalDisplayLink 정상 발화.
                // 7월 커밋(b82551a, 정책이 .accessory 하드코딩이던 시기)에 "뷰어가 전체화면 소스
                // 위에 정상 합성됨"이 기록돼 있던 것과도 일치한다 — 같은 창 속성인데 7월엔 됐다.
                //
                // 교훈: 배경 상주 앱(LSUIElement)이어야 다른 앱의 전체화면 Space 위에 뜬다.
                // 메뉴바 전용 설정을 끄면 그 능력을 잃는다.
                if !self.window.isOnActiveSpace {
                    DiagnosticLog.shared.log(
                        "[WIN] ⚠︎ 출력 창이 활성 Space에 없다 — 화면에 안 나오고 링크도 발화하지 않는다. "
                        + "가장 흔한 원인: 메뉴바 전용 설정이 꺼져 있음(앱이 .regular라 다른 앱의 "
                        + "전체화면 Space에 못 올라간다). 확인: defaults read com.macfg.MacFG s.menubaronly")
                }
            }
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

    /// 창 크기가 바뀌면 서피스 파라미터(contentBounds/배율/드로어블)를 다시 읽는다.
    /// 없으면 contentBounds가 옛 크기에 머물러, 레터박스 fit이 그 작은 영역 안에서 계산되고
    /// 콘텐츠가 좌하단에 몰린다 — 전체화면 자동 전환(창이 커짐)에서 오른쪽·위쪽 패딩으로 발현.
    public func windowDidResize(_ notification: Notification) {
        refreshSurfaceParams()
    }

    /// 전체화면 진입/이탈은 리사이즈 통지가 애니메이션 도중 값으로 올 수 있어 완료 시점에 재갱신
    public func windowDidEnterFullScreen(_ notification: Notification) { refreshSurfaceParams() }
    public func windowDidExitFullScreen(_ notification: Notification)  { refreshSurfaceParams() }

    /// 다른 배율의 화면으로 옮겨가면 contentsScale이 달라진다
    public func windowDidChangeScreen(_ notification: Notification) { refreshSurfaceParams() }
}

