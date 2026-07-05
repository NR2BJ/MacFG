/// PairIO — 품질 A/B용 PNG 쌍 입출력 + 페어모드 러너.
///
/// 실영상 삼중항(im0, im1=GT, im2)에서 각 엔진이 im0+im2로 중간프레임을 예측하고
/// PNG로 덤프한다. PSNR/SSIM은 Python 한 곳에서 계산(공정성). 합성 패턴이 아닌
/// 실콘텐츠에서 RIFE(신경망) vs MetalFlow(고전) vs AppleFI(ANE) 화질 비교가 목적.
///
/// 사용: swift run -c release InterpBench --pair-dir DIR --engine metalflow
///   DIR/<triplet>/a.png, b.png 를 읽어 DIR/<triplet>/pred_<engine>.png 생성.
///   orientation은 로드 시 Y-flip으로 top-down 통일 → 덤프 CGImage와 왕복 일치.

import Metal
import Interpolation
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// PNG → BGRA8 top-down MTLTexture (CoreGraphics 바텀레프트 보정 위해 Y-flip 후 그림).
func loadPNGTexture(path: String, device: any MTLDevice) -> (any MTLTexture)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = img.width, h = img.height
    let bpr = w * 4
    var data = [UInt8](repeating: 0, count: bpr * h)
    // 이미지 자신의 컬러스페이스로 그리기 (색변환 배제)
    let cs = img.colorSpace ?? CGColorSpaceCreateDeviceRGB()
    let bmp = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: bpr, space: cs, bitmapInfo: bmp) else { return nil }
    // 주의: CGBitmapContext에 draw(image:)는 이미 row0=이미지 상단(top-down) — 플립 변환을
    // 넣으면 오히려 bottom-up이 된다. (플립 관용구는 텍스트 등 드로잉 프리미티브용.)
    // 이전에 로더 플립+라이터 행플립이 서로 상쇄돼 등변 엔진(blend 등)은 통과했지만,
    // RIFE는 학습 모델이라 뒤집힌 입력에서 flow 품질이 달라져 패리티가 깨졌었다.
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))

    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.usage = [.shaderRead, .shaderWrite]
    desc.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: desc) else { return nil }
    tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                withBytes: data, bytesPerRow: bpr)
    return tex
}

/// top-down BGRA8 텍스처 → PNG (블릿으로 shared 복사 후 CGImage 직접 생성, 재flip 없음).
func writePNGTexture(_ tex: any MTLTexture, path: String,
                     device: any MTLDevice, queue: any MTLCommandQueue) -> Bool {
    let w = tex.width, h = tex.height, bpr = w * 4
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    desc.storageMode = .shared
    desc.usage = [.shaderRead]
    guard let shared = device.makeTexture(descriptor: desc),
          let cb = queue.makeCommandBuffer(),
          let blit = cb.makeBlitCommandEncoder() else { return false }
    blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: shared, destinationSlice: 0, destinationLevel: 0,
              destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    var data = [UInt8](repeating: 0, count: bpr * h)
    shared.getBytes(&data, bytesPerRow: bpr,
                    from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    // 텍스처는 top-down, CGImage도 row0=상단 — 그대로 쓴다 (플립 불필요)
    let cs = CGColorSpaceCreateDeviceRGB()
    let bmp = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    guard let provider = CGDataProvider(data: Data(data) as CFData),
          let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: bpr, space: cs, bitmapInfo: CGBitmapInfo(rawValue: bmp),
                           provider: provider, decode: nil, shouldInterpolate: false,
                           intent: .defaultIntent) else { return false }
    guard let dest = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) else { return false }
    CGImageDestinationAddImage(dest, cg, nil)
    return CGImageDestinationFinalize(dest)
}

/// 한 엔진으로 삼중항 디렉터리 전체 처리: 각 <triplet>/a,b.png → pred_<engine>.png
func runPairMode(engineKey: String, dir: String,
                 device: any MTLDevice, queue: any MTLCommandQueue) async {
    let engine: any PairInterpolationEngine
    switch engineKey {
    case "metalflow": engine = MetalFlowEngine()
    case "applefi":
        guard AppleFIEngine.isSupported else { print("applefi 미지원 — 스킵"); return }
        engine = AppleFIEngine()
    case "blend": engine = LegacyPairEngine(BlendInterpolator())
    case "rife":
        guard RIFEEngine.modelAvailable(short: RIFEEngine.flowShortSide) else {
            print("rife\(RIFEEngine.flowShortSide) 모델 없음 — 스킵"); return
        }
        engine = RIFEEngine()
    default: print("알 수 없는 엔진 \(engineKey)"); return
    }
    do { try await engine.prepare(device: device) }
    catch { print("prepare 실패: \(error)"); return }

    let fm = FileManager.default
    guard let subs = try? fm.contentsOfDirectory(atPath: dir).sorted() else {
        print("디렉터리 읽기 실패: \(dir)"); return
    }
    var ts = 100.0
    let dt = 1.0 / 60.0
    var done = 0
    for name in subs {
        let tdir = "\(dir)/\(name)"
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: tdir, isDirectory: &isDir), isDir.boolValue else { continue }
        let aPath = "\(tdir)/a.png", bPath = "\(tdir)/b.png"
        guard fm.fileExists(atPath: aPath), fm.fileExists(atPath: bPath),
              let a = loadPNGTexture(path: aPath, device: device),
              let b = loadPNGTexture(path: bPath, device: device) else { continue }

        engine.reset()
        var out: (any MTLTexture)?
        // 워밍업 3 + 최종 1 (MetalFlow 시간적 prior / AppleFI 세션 정착; ts 단조증가)
        for i in 0..<4 {
            guard let cb = queue.makeCommandBuffer() else { break }
            let r = engine.encodePair(stableA: a, stableB: b, tsA: ts, tsB: ts + dt,
                                      tValues: [0.5], into: cb)
            cb.commit(); await cb.completed()
            ts += dt
            if i == 3 { out = r?.frames.first?.texture }
        }
        guard let out else { print("  \(name): 보간 실패"); continue }
        let outPath = "\(tdir)/pred_\(engineKey).png"
        if writePNGTexture(out, path: outPath, device: device, queue: queue) { done += 1 }
        else { print("  \(name): PNG 쓰기 실패") }
    }
    engine.shutdown()
    print("✅ \(engineKey): \(done)개 삼중항 예측 완료 → \(dir)/*/pred_\(engineKey).png")
}
