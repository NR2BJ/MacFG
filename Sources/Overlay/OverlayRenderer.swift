import Metal

/// 오버레이 렌더러 프로토콜
public protocol OverlayRenderer: AnyObject {
    /// 텍스처를 오버레이에 렌더링
    func render(texture: any MTLTexture, commandBuffer: any MTLCommandBuffer)
    /// 오버레이 위치/크기 갱신
    func updateFrame(_ frame: CGRect)
    /// 오버레이 표시/숨김
    func setVisible(_ visible: Bool)
}
