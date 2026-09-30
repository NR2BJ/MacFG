import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
let W = 1920, H = 1080
let cs = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.clear(CGRect(x: 0, y: 0, width: W, height: H))
let text = "흔들림 테스트 자막" as CFString
let font = CTFontCreateWithName("AppleSDGothicNeo-Heavy" as CFString, 92, nil)
func line(_ fill: CGColor, stroke: CGColor?, width: CGFloat) -> CTLine {
    var attrs: [CFString: Any] = [kCTFontAttributeName: font, kCTForegroundColorAttributeName: fill]
    if let st = stroke { attrs[kCTStrokeColorAttributeName] = st; attrs[kCTStrokeWidthAttributeName] = width }
    let a = CFAttributedStringCreate(nil, text, attrs as CFDictionary)!
    return CTLineCreateWithAttributedString(a)
}
let fillLine = line(CGColor(red: 1, green: 0, blue: 0, alpha: 1), stroke: nil, width: 0)
let bounds = CTLineGetImageBounds(fillLine, ctx)
let x = CGFloat(W) / 2 - bounds.width / 2 - bounds.origin.x
let y = CGFloat(H) - 820   // 좌하 원점: 화면 위에서 ~760px 근처 기준선
// 검은 외곽선(두꺼운 스트로크) 먼저, 그 위에 빨간 채움
ctx.textPosition = CGPoint(x: x, y: y)
CTLineDraw(line(CGColor(red: 0, green: 0, blue: 0, alpha: 1), stroke: CGColor(red: 0, green: 0, blue: 0, alpha: 1), width: 22), ctx)
ctx.textPosition = CGPoint(x: x, y: y)
CTLineDraw(fillLine, ctx)
let img = ctx.makeImage()!
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dst, img, nil); CGImageDestinationFinalize(dst)
print("subtitle bounds", bounds, "x", x, "y", y)
