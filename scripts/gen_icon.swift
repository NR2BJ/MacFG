// 앱 아이콘 생성 — 프레임 두 장이 겹치며 배가 되는 모티프
// 사용: swift scripts/gen_icon.swift <출력.png>
import AppKit

let size: CGFloat = 1024
guard CommandLine.arguments.count > 1 else { fatalError("usage: gen_icon.swift out.png") }
let outPath = CommandLine.arguments[1]

let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()
guard let ctx = NSGraphicsContext.current?.cgContext else { fatalError() }

// 배경: 딥 네이비 → 퍼플 대각 그라디언트, 라운드 사각 (macOS 스쿼클 근사)
let bgRect = CGRect(x: 64, y: 64, width: size - 128, height: size - 128)
let bgPath = CGPath(roundedRect: bgRect, cornerWidth: 200, cornerHeight: 200, transform: nil)
ctx.addPath(bgPath)
ctx.clip()
let colors = [
    CGColor(red: 0.07, green: 0.09, blue: 0.20, alpha: 1),
    CGColor(red: 0.22, green: 0.12, blue: 0.42, alpha: 1),
] as CFArray
let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 1])!
ctx.drawLinearGradient(grad, start: CGPoint(x: 64, y: size - 64), end: CGPoint(x: size - 64, y: 64), options: [])

// 뒤 프레임 (희미함 — 원본 프레임)
func frame(_ rect: CGRect, alpha: CGFloat, lineWidth: CGFloat, color: CGColor) {
    let p = CGPath(roundedRect: rect, cornerWidth: 48, cornerHeight: 48, transform: nil)
    ctx.setStrokeColor(color.copy(alpha: alpha)!)
    ctx.setLineWidth(lineWidth)
    ctx.addPath(p)
    ctx.strokePath()
}
let cyan = CGColor(red: 0.35, green: 0.85, blue: 1.0, alpha: 1)
let white = CGColor(red: 1, green: 1, blue: 1, alpha: 1)

frame(CGRect(x: 240, y: 330, width: 400, height: 400), alpha: 0.35, lineWidth: 26, color: white)
// 중간 프레임 (보간 프레임 — 반투명 시안)
frame(CGRect(x: 330, y: 300, width: 400, height: 400), alpha: 0.65, lineWidth: 26, color: cyan)
// 앞 프레임 (선명)
frame(CGRect(x: 420, y: 270, width: 400, height: 400), alpha: 1.0, lineWidth: 30, color: white)

// 앞 프레임 안에 재생/전진 화살표
ctx.setFillColor(cyan)
ctx.beginPath()
ctx.move(to: CGPoint(x: 540, y: 380))
ctx.addLine(to: CGPoint(x: 540, y: 560))
ctx.addLine(to: CGPoint(x: 690, y: 470))
ctx.closePath()
ctx.fillPath()

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let png = rep.representation(using: .png, properties: [:]) else { fatalError("png encode fail") }
try! png.write(to: URL(fileURLWithPath: outPath))
print("icon written: \(outPath)")
