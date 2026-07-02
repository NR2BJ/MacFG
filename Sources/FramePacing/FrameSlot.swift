@preconcurrency import Metal
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

    public init(texture: (any MTLTexture)?, timestamp: CFTimeInterval, width: Int = 0, height: Int = 0, contentChanged: Bool = true, contentFingerprint: UInt64 = 0, colorSpace: CGColorSpace? = nil) {
        self.texture = texture
        self.timestamp = timestamp
        self.width = width
        self.height = height
        self.contentChanged = contentChanged
        self.contentFingerprint = contentFingerprint
        self.colorSpace = colorSpace
    }

    public static let empty = FrameSlot(texture: nil, timestamp: 0, contentChanged: false)
}
