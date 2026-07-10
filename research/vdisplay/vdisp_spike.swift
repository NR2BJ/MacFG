// vdisp_spike.swift — 가상 디스플레이 타당성 스파이크 (U4).
// 목적: 단일 모니터에서 "소스는 (가상)전체화면 4K 렌더 + 실제 모니터엔 보간 출력 전체화면"을
// 성립시키는 유일 경로인 CGVirtualDisplay(비공개 API, BetterDisplay/DeskPad 계열 기법)가
// 이 macOS에서 동작하는지 실측:
//   ① CGVirtualDisplay로 3840×2160@60 가상 디스플레이 생성 (온라인 확인)
//   ② 그 위에 60fps 애니메이션 창(vdispwin) 배치
//   ③ (별도 프로세스) MacFG가 그 창을 4K/60으로 캡처하는지 diag 로그로 확인
// 사용: ./vdispspike [유지초=120]  — 종료 시 가상 디스플레이 자동 소멸(프로세스 소유)
import AppKit
import CoreGraphics

let SECS = CommandLine.arguments.count > 1 ? (Double(CommandLine.arguments[1]) ?? 120) : 120
let HIDPI = CommandLine.arguments.count > 2 ? (UInt32(CommandLine.arguments[2]) ?? 1) : 1
let MW = CommandLine.arguments.count > 3 ? (UInt32(CommandLine.arguments[3]) ?? 1920) : 1920
let MH = CommandLine.arguments.count > 4 ? (UInt32(CommandLine.arguments[4]) ?? 1080) : 1080

func listDisplays(_ tag: String) {
    var ids = [CGDirectDisplayID](repeating: 0, count: 16); var n: UInt32 = 0
    CGGetOnlineDisplayList(16, &ids, &n)
    let desc = (0..<Int(n)).map { i -> String in
        let id = ids[i]; let b = CGDisplayBounds(id)
        return "id=\(id) \(Int(b.width))x\(Int(b.height))@(\(Int(b.minX)),\(Int(b.minY)))\(CGDisplayIsBuiltin(id) != 0 ? " builtin" : "")"
    }
    print("[\(tag)] displays(\(n)): " + desc.joined(separator: " | "))
    fflush(stdout)
}

listDisplays("before")

// ── 비공개 클래스 로드 (런타임 조회 — 헤더 불필요)
guard let descCls = NSClassFromString("CGVirtualDisplayDescriptor") as? NSObject.Type,
      let dispCls = NSClassFromString("CGVirtualDisplay"),
      let setCls = NSClassFromString("CGVirtualDisplaySettings") as? NSObject.Type,
      let modeCls = NSClassFromString("CGVirtualDisplayMode") else {
    print("VERDICT=FAIL reason=CGVirtualDisplay 클래스 없음 (이 macOS에서 제거/개명됨)")
    exit(1)
}

let d = descCls.init()
d.setValue("MacFG Virtual 4K", forKey: "name")
d.setValue(NSNumber(value: 0x4D46), forKey: "vendorID")   // 'MF'
d.setValue(NSNumber(value: 1), forKey: "productID")
d.setValue(NSNumber(value: 1), forKey: "serialNum")
d.setValue(NSNumber(value: 3840), forKey: "maxPixelsWide")
d.setValue(NSNumber(value: 2160), forKey: "maxPixelsHigh")
d.setValue(NSValue(size: NSSize(width: 600, height: 340)), forKey: "sizeInMillimeters")
d.setValue(NSValue(point: NSPoint(x: 0.3127, y: 0.3290)), forKey: "whitePoint")
d.setValue(NSValue(point: NSPoint(x: 0.64, y: 0.33)), forKey: "redPrimary")
d.setValue(NSValue(point: NSPoint(x: 0.30, y: 0.60)), forKey: "greenPrimary")
d.setValue(NSValue(point: NSPoint(x: 0.15, y: 0.06)), forKey: "bluePrimary")
d.setValue(DispatchQueue.main, forKey: "queue")

let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend")!
let allocFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector) -> Unmanaged<AnyObject>).self)
let init1Fn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, AnyObject) -> Unmanaged<AnyObject>?).self)
let modeFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, UInt32, UInt32, Double) -> Unmanaged<AnyObject>).self)
let applyFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, AnyObject) -> Bool).self)

// display = [[CGVirtualDisplay alloc] initWithDescriptor:d]  (참조 유지 — 소멸 시 디스플레이 사라짐)
let dispAlloc = allocFn(dispCls, NSSelectorFromString("alloc")).takeUnretainedValue()
guard let display = init1Fn(dispAlloc, NSSelectorFromString("initWithDescriptor:"), d)?.takeUnretainedValue() else {
    print("VERDICT=FAIL reason=initWithDescriptor nil")
    exit(1)
}

let modeAlloc = allocFn(modeCls, NSSelectorFromString("alloc")).takeUnretainedValue()
let mode = modeFn(modeAlloc, NSSelectorFromString("initWithWidth:height:refreshRate:"), MW, MH, 60).takeUnretainedValue()

print("mode readback: w=\((mode as AnyObject).value(forKey: "width") ?? "?") h=\((mode as AnyObject).value(forKey: "height") ?? "?") hz=\((mode as AnyObject).value(forKey: "refreshRate") ?? "?")"); fflush(stdout)

let st = setCls.init()
st.setValue(NSNumber(value: HIDPI), forKey: "hiDPI")  // 1=레티나(pt×2=px) — 진짜 4K 디스플레이와 동일
st.setValue([mode], forKey: "modes")

let ok = applyFn(display, NSSelectorFromString("applySettings:"), st)
let did = (display as AnyObject).value(forKey: "displayID") as? UInt32 ?? 0
print("applySettings=\(ok) displayID=\(did)"); fflush(stdout)
if !ok || did == 0 { print("VERDICT=FAIL reason=활성화 실패"); exit(1) }

// ── AppKit: 가상 디스플레이 위 애니메이션 창
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
    listDisplays("after")
    guard let scr = NSScreen.screens.first(where: {
        ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value == did
    }) else {
        print("VERDICT=FAIL reason=NSScreen에 가상 디스플레이 미등장"); exit(1)
    }
    // 등록은 되지만 기본 1920x1080로 켜짐 → 공개 API로 목표 모드 전환
    if let all = CGDisplayCopyAllDisplayModes(did, nil) as? [CGDisplayMode] {
        print("디스플레이 노출 모드: " + all.map { "\($0.pixelWidth)x\($0.pixelHeight)@\(Int($0.refreshRate))" }.joined(separator: " "))
        if let target = all.first(where: { $0.pixelWidth == Int(MW) && $0.pixelHeight == Int(MH) }) {
            let err = CGDisplaySetDisplayMode(did, target, nil)
            print("CGDisplaySetDisplayMode(\(MW)x\(MH)) → \(err == .success ? "성공" : "실패 \(err.rawValue)")"); fflush(stdout)
        } else {
            print("목표 모드 \(MW)x\(MH) 미노출"); fflush(stdout)
        }
    }
    let pxW = CGDisplayPixelsWide(did), pxH = CGDisplayPixelsHigh(did)
    var pxMode = "?"
    if let m = CGDisplayCopyDisplayMode(did) { pxMode = "\(m.pixelWidth)x\(m.pixelHeight)@\(m.refreshRate)Hz" }
    print("vscreen frame=\(scr.frame) scale=\(scr.backingScaleFactor) px=\(pxW)x\(pxH) mode=\(pxMode)"); fflush(stdout)

    let w = NSWindow(contentRect: scr.frame, styleMask: [.titled], backing: .buffered, defer: false)
    w.title = "vdispwin"
    w.setFrame(scr.frame, display: true)
    let v = NSView(frame: w.contentView!.bounds)
    v.wantsLayer = true
    v.layer?.backgroundColor = NSColor.black.cgColor
    let bar = CALayer()
    bar.backgroundColor = NSColor.systemOrange.cgColor
    bar.frame = CGRect(x: 0, y: 0, width: 300, height: v.bounds.height)
    v.layer?.addSublayer(bar)
    let txt = CATextLayer()
    txt.fontSize = 300
    txt.foregroundColor = NSColor.white.cgColor
    txt.frame = CGRect(x: 400, y: v.bounds.height / 2 - 200, width: 2600, height: 420)
    v.layer?.addSublayer(txt)
    w.contentView = v
    w.orderFrontRegardless()

    var tick = 0
    Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { _ in
        tick += 1
        CATransaction.begin(); CATransaction.setDisableActions(true)
        let x = CGFloat(tick % 118) / 118.0 * (v.bounds.width - 300)
        bar.frame.origin.x = x
        txt.string = "vdisp \(tick)"
        CATransaction.commit()
    }
    print("READY window=vdispwin on displayID=\(did)"); fflush(stdout)
}

DispatchQueue.main.asyncAfter(deadline: .now() + SECS) {
    print("VERDICT=DONE (유지시간 종료 — 가상 디스플레이 해제)")
    exit(0)
}
app.run()
