import AppKit
import CoreGraphics
import Monitoring

/// 가상 디스플레이 관리 (U4 "가상 전체화면") — CGVirtualDisplay 비공개 API를 런타임 조회로 사용.
/// 스파이크 실측(research/vdisplay/vdisp_spike.swift): 요청 모드는 '등록'만 되고 기본 1920x1080로
/// 켜짐(hiDPI KVC 무시) → 온라인 후 공개 API CGDisplaySetDisplayMode로 전환해야 실제 4K@60이 됨.
/// NSScreen은 전환 직후 stale일 수 있으나 창 이동은 CG 좌표(CGDisplayBounds)만 쓰므로 무관.
/// 프로세스 종료 시 디스플레이는 자동 소멸(객체 소유) — 크래시에도 시스템에 잔재 없음.
@MainActor
final class VirtualDisplayManager {
    private var display: AnyObject?          // 참조 유지 = 디스플레이 생존
    private(set) var displayID: CGDirectDisplayID = 0
    var isActive: Bool { display != nil && displayID != 0 }
    /// 가상 디스플레이의 글로벌 CG 좌표 bounds (창 이동 목적지)
    var cgBounds: CGRect { CGDisplayBounds(displayID) }

    /// 생성 → 온라인 대기 → 목표 모드 전환. 실패 시 false (비공개 API 소실/거부 대비).
    func create(pixelsWide: Int, pixelsHigh: Int, refresh: Double = 60) async -> Bool {
        guard display == nil else { return true }
        guard let descCls = NSClassFromString("CGVirtualDisplayDescriptor") as? NSObject.Type,
              let dispCls = NSClassFromString("CGVirtualDisplay"),
              let setCls = NSClassFromString("CGVirtualDisplaySettings") as? NSObject.Type,
              let modeCls = NSClassFromString("CGVirtualDisplayMode") else {
            DiagnosticLog.shared.log("[VFS] CGVirtualDisplay 클래스 없음 — 이 macOS에서 사용 불가")
            return false
        }

        let d = descCls.init()
        d.setValue("MacFG Virtual Display", forKey: "name")
        d.setValue(NSNumber(value: 0x4D46), forKey: "vendorID")   // 'MF'
        d.setValue(NSNumber(value: 1), forKey: "productID")
        d.setValue(NSNumber(value: 1), forKey: "serialNum")
        d.setValue(NSNumber(value: pixelsWide), forKey: "maxPixelsWide")
        d.setValue(NSNumber(value: pixelsHigh), forKey: "maxPixelsHigh")
        d.setValue(NSValue(size: NSSize(width: 600, height: 340)), forKey: "sizeInMillimeters")
        d.setValue(NSValue(point: NSPoint(x: 0.3127, y: 0.3290)), forKey: "whitePoint")
        d.setValue(NSValue(point: NSPoint(x: 0.64, y: 0.33)), forKey: "redPrimary")
        d.setValue(NSValue(point: NSPoint(x: 0.30, y: 0.60)), forKey: "greenPrimary")
        d.setValue(NSValue(point: NSPoint(x: 0.15, y: 0.06)), forKey: "bluePrimary")
        d.setValue(DispatchQueue.main, forKey: "queue")

        guard let msgSend = dlsym(dlopen(nil, RTLD_LAZY), "objc_msgSend") else { return false }
        let allocFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector) -> Unmanaged<AnyObject>).self)
        let init1Fn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, AnyObject) -> Unmanaged<AnyObject>?).self)
        let modeFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, UInt32, UInt32, Double) -> Unmanaged<AnyObject>).self)
        let applyFn = unsafeBitCast(msgSend, to: (@convention(c) (AnyObject, Selector, AnyObject) -> Bool).self)

        let dispAlloc = allocFn(dispCls, NSSelectorFromString("alloc")).takeUnretainedValue()
        guard let disp = init1Fn(dispAlloc, NSSelectorFromString("initWithDescriptor:"), d)?.takeUnretainedValue() else {
            DiagnosticLog.shared.log("[VFS] initWithDescriptor 실패")
            return false
        }
        let modeAlloc = allocFn(modeCls, NSSelectorFromString("alloc")).takeUnretainedValue()
        let mode = modeFn(modeAlloc, NSSelectorFromString("initWithWidth:height:refreshRate:"),
                          UInt32(pixelsWide), UInt32(pixelsHigh), refresh).takeUnretainedValue()
        let st = setCls.init()
        st.setValue(NSNumber(value: 0), forKey: "hiDPI")   // 1x — 모드 px 그대로 (실측 경로)
        st.setValue([mode], forKey: "modes")

        let ok = applyFn(disp, NSSelectorFromString("applySettings:"), st)
        let did = (disp as AnyObject).value(forKey: "displayID") as? UInt32 ?? 0
        guard ok, did != 0 else {
            DiagnosticLog.shared.log("[VFS] applySettings 실패 (ok=\(ok) id=\(did))")
            return false
        }
        display = disp
        displayID = did

        // 온라인 대기 (≤2s) — WindowServer 등록 비동기
        for _ in 0..<20 {
            if CGDisplayIsOnline(did) != 0 { break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        // 기본 1920x1080로 켜짐 → 등록된 목표 모드로 전환 (공개 API).
        // 커스텀 모드가 온라인 직후엔 목록에 아직 없을 수 있어 재시도 (E2E 실측: 즉시 조회는 미등장)
        var switched = false
        for _ in 0..<20 {
            if let all = CGDisplayCopyAllDisplayModes(did, nil) as? [CGDisplayMode],
               let target = all.first(where: { $0.pixelWidth == pixelsWide && $0.pixelHeight == pixelsHigh }) {
                switched = CGDisplaySetDisplayMode(did, target, nil) == .success
                break
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        // 모드 전환 반영 대기 (bounds가 목표 크기가 될 때까지 ≤2s)
        for _ in 0..<20 where Int(cgBounds.width) != pixelsWide {
            try? await Task.sleep(for: .milliseconds(100))
        }
        DiagnosticLog.shared.log("[VFS] 생성 id=\(did) 목표=\(pixelsWide)x\(pixelsHigh) 전환=\(switched) bounds=\(Int(cgBounds.width))x\(Int(cgBounds.height))@(\(Int(cgBounds.minX)),\(Int(cgBounds.minY)))")
        return true
    }

    /// 해제 — 참조를 놓으면 WindowServer가 디스플레이를 내리고 그 위 창들은 실제 화면으로 돌아온다.
    func destroy() {
        guard display != nil else { return }
        DiagnosticLog.shared.log("[VFS] 해제 id=\(displayID)")
        display = nil
        displayID = 0
    }
}
