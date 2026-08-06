@preconcurrency import CoreVideo
import Metal
import CoreGraphics

/// 프레임 버퍼의 단일 슬롯
public struct FrameSlot: @unchecked Sendable {
    public let texture: (any MTLTexture)?
    public let timestamp: CFTimeInterval
    public let width: Int
    public let height: Int
    /// SCK 프레임 상태 기반 콘텐츠 변화 여부 (SCFrameStatus.complete = true)
    public let contentChanged: Bool
    /// 콘텐츠 변화 감지용 fingerprint (몇 개 샘플 픽셀 해시)
    public let contentFingerprint: UInt64
    /// 캡처 프레임의 색공간 (CMSampleBuffer 어태치먼트 기반). 출력 레이어가 동일하게 태깅해야 색 왜곡이 없다.
    public let colorSpace: CGColorSpace?
    /// **texture가 감싸고 있는 IOSurface의 생산자 버퍼.** 소비가 끝날 때까지 반드시 살려둬야 한다.
    ///
    /// texture는 CVPixelBuffer의 IOSurface를 제로카피로 감싼 것이다. Metal은 IOSurface를
    /// 붙잡지만, **CVPixelBufferPool의 재활용 판정은 IOSurface가 아니라 CVPixelBuffer의
    /// 참조수를 본다.** 그래서 이 버퍼를 놓아버리면 풀이 같은 표면을 다음 프레임에 다시 내주고,
    /// SCK가 거기에 새 프레임을 쓰는 동안 우리 blit이 읽어 **한 장에 두 프레임이 섞인다**.
    /// 콜백에서 인제스트까지 5~15ms가 걸리는데 소스는 10ms마다 오므로 창이 넓다.
    public let pixelBuffer: CVPixelBuffer?

    public init(texture: (any MTLTexture)?, timestamp: CFTimeInterval, width: Int = 0, height: Int = 0, contentChanged: Bool = true, contentFingerprint: UInt64 = 0, colorSpace: CGColorSpace? = nil, pixelBuffer: CVPixelBuffer? = nil) {
        self.texture = texture
        self.pixelBuffer = pixelBuffer
        self.timestamp = timestamp
        self.width = width
        self.height = height
        self.contentChanged = contentChanged
        self.contentFingerprint = contentFingerprint
        self.colorSpace = colorSpace
    }

    public static let empty = FrameSlot(texture: nil, timestamp: 0, contentChanged: false)
}
