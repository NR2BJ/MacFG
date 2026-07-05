import Metal
import CoreGraphics
import FramePacing

/// 캡처 소스 종류
public enum CaptureMethod: String, Sendable {
    case ioSurface = "IOSurface"
    case screenCaptureKit = "ScreenCaptureKit"
}

/// 프레임 캡처 프로토콜
public protocol FrameSource: AnyObject, Sendable {
    /// 캡처 시작. 창 ID 지정. captureRect(창 상대 pt)면 그 영역만 크롭 캡처(SCK 전용, nil=전체).
    func startCapture(windowID: CGWindowID, device: any MTLDevice, captureRect: CGRect?) async throws
    /// 캡처 중지
    func stopCapture() async
    /// 최신 프레임 가져오기
    func latestFrame() -> FrameSlot?
    /// 마지막 drain 이후 도착한 모든 프레임을 타임스탬프 순으로 반환.
    /// 타임스탬프 기반 출력 스케줄러는 이 경로를 사용한다 (latest-only는 프레임 드랍 유발).
    func drainFrames() -> [FrameSlot]
    /// 캡처 방식
    var method: CaptureMethod { get }
    /// 사용 가능 여부
    var isAvailable: Bool { get }
}

// 기본 구현: latestFrame 폴백 (큐 미지원 소스용)
extension FrameSource {
    public func drainFrames() -> [FrameSlot] {
        guard let slot = latestFrame() else { return [] }
        return [slot]
    }
}
