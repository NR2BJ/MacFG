import AppKit
import ApplicationServices
import CoreGraphics
import os

// _AXUIElementGetWindow — AXUIElement에서 CGWindowID를 얻는 private API
private typealias AXUIElementGetWindowFunc = @convention(c) (AXUIElement, UnsafeMutablePointer<CGWindowID>) -> AXError
private let _AXUIElementGetWindow: AXUIElementGetWindowFunc? = {
    guard let sym = dlsym(dlopen(nil, RTLD_LAZY), "_AXUIElementGetWindow") else { return nil }
    return unsafeBitCast(sym, to: AXUIElementGetWindowFunc.self)
}()

/// 창 위치/크기 정보
public struct WindowGeometry: Equatable, Sendable {
    public let origin: CGPoint
    public let size: CGSize

    public var frame: CGRect {
        // macOS 좌표계 변환: CG(좌상단 원점) → NS(좌하단 원점)
        // 기준 높이는 NSScreen.main이 아닌 좌표계 원점 화면 (origin == .zero)
        let primaryHeight = NSScreen.screens
            .first(where: { $0.frame.origin == .zero })?
            .frame.height ?? NSScreen.main?.frame.height ?? 0
        return CGRect(
            x: origin.x,
            y: primaryHeight - origin.y - size.height,
            width: size.width,
            height: size.height
        )
    }
}

/// 창 추적 방식
public enum TrackingMethod: String, Sendable {
    case accessibility = "Accessibility"
    case cgWindowList = "CGWindowList"
}

/// 대상 창 추적 — Accessibility API 우선, CGWindowList 폴백
@MainActor
public final class WindowTracker {
    private let logger = Logger(subsystem: "com.macfg", category: "WindowTracker")

    private var windowID: CGWindowID = 0
    private var pid: pid_t = 0
    private var axElement: AXUIElement?
    private var axObserver: AXObserver?
    private var trackingMethod: TrackingMethod = .accessibility
    private var pollingTimer: Timer?

    /// 마지막 감지된 창 위치
    private(set) public var lastGeometry: WindowGeometry?

    /// 현재 추적 방식
    public var currentMethod: TrackingMethod { trackingMethod }

    /// 위치 변경 콜백
    public var onGeometryChanged: ((WindowGeometry) -> Void)?

    /// 접근성 타임아웃 (초) — 이 시간 동안 이벤트 없으면 CGWindowList로 전환
    private let axTimeoutSeconds: TimeInterval = 3.0
    private var lastAXEventTime: Date = .distantPast

    public init() {}

    /// 추적 시작
    public func startTracking(windowID: CGWindowID) {
        self.windowID = windowID

        // 창의 PID 찾기
        guard let pid = findPID(for: windowID) else {
            logger.warning("Could not find PID for window \(windowID), using CGWindowList only")
            startCGWindowListPolling()
            return
        }
        self.pid = pid

        // Accessibility API 시도
        if AXIsProcessTrusted() {
            setupAccessibilityTracking(pid: pid, windowID: windowID)
        } else {
            logger.info("Accessibility not trusted, using CGWindowList")
            startCGWindowListPolling()
        }
    }

    /// 추적 중지
    public func stopTracking() {
        pollingTimer?.invalidate()
        pollingTimer = nil

        if let observer = axObserver {
            CFRunLoopRemoveSource(
                CFRunLoopGetMain(),
                AXObserverGetRunLoopSource(observer),
                .defaultMode
            )
        }
        axObserver = nil
        axElement = nil
        windowID = 0
    }

    // MARK: - Accessibility API

    private func setupAccessibilityTracking(pid: pid_t, windowID: CGWindowID) {
        let appElement = AXUIElementCreateApplication(pid)

        // 앱의 윈도우 목록에서 대상 찾기
        var windowsRef: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(appElement, kAXWindowsAttribute as CFString, &windowsRef)
        guard err == .success, let windows = windowsRef as? [AXUIElement] else {
            logger.info("Could not get AX windows, falling back to CGWindowList")
            startCGWindowListPolling()
            return
        }

        // _AXUIElementGetWindow로 정확한 CGWindowID 매칭
        var targetWindow: AXUIElement?
        if let getWindow = _AXUIElementGetWindow {
            for win in windows {
                var wid: CGWindowID = 0
                if getWindow(win, &wid) == .success && wid == windowID {
                    targetWindow = win
                    break
                }
            }
        }

        // private API 실패 시 CGWindowList 위치 기반 매칭 폴백
        if targetWindow == nil {
            if let cgGeom = readCGWindowListGeometry() {
                for win in windows {
                    if let axGeom = readAXGeometry(win),
                       abs(axGeom.origin.x - cgGeom.origin.x) < 2,
                       abs(axGeom.origin.y - cgGeom.origin.y) < 2 {
                        targetWindow = win
                        break
                    }
                }
            }
        }

        guard let targetWindow else {
            logger.info("Could not match AX window to ID \(windowID), falling back to CGWindowList")
            startCGWindowListPolling()
            return
        }

        self.axElement = targetWindow
        self.trackingMethod = .accessibility

        // 초기 위치 읽기 — CGWindowList 우선 (정확한 windowID 기반)
        if let geom = readCGWindowListGeometry() ?? readAXGeometry(targetWindow) {
            lastGeometry = geom
            onGeometryChanged?(geom)
        }

        // AXObserver 설정
        var observer: AXObserver?
        let callback: AXObserverCallback = { _, element, notification, refcon in
            guard let refcon else { return }
            let tracker = Unmanaged<WindowTracker>.fromOpaque(refcon).takeUnretainedValue()
            Task { @MainActor in
                tracker.handleAXNotification(element: element, notification: notification as String)
            }
        }

        AXObserverCreate(pid, callback, &observer)
        guard let observer else {
            startCGWindowListPolling()
            return
        }

        self.axObserver = observer
        let refcon = Unmanaged.passUnretained(self).toOpaque()

        AXObserverAddNotification(observer, targetWindow, kAXMovedNotification as CFString, refcon)
        AXObserverAddNotification(observer, targetWindow, kAXResizedNotification as CFString, refcon)

        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(observer),
            .defaultMode
        )

        lastAXEventTime = Date()

        // 타임아웃 감지용 폴링도 병행
        startFallbackDetectionTimer()

        logger.info("Accessibility tracking started for PID \(pid)")
    }

    private func handleAXNotification(element: AXUIElement, notification: String) {
        lastAXEventTime = Date()
        // AX 이벤트 트리거로 사용하되, 실제 geometry는 CGWindowList에서 읽음 (정확한 windowID 기반)
        if let geom = readCGWindowListGeometry() ?? readAXGeometry(element) {
            if geom != lastGeometry {
                lastGeometry = geom
                onGeometryChanged?(geom)
            }
        }
    }

    private func readAXGeometry(_ element: AXUIElement) -> WindowGeometry? {
        var posRef: CFTypeRef?
        var sizeRef: CFTypeRef?

        guard AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &posRef) == .success,
              AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef) == .success
        else { return nil }

        var point = CGPoint.zero
        var size = CGSize.zero
        AXValueGetValue(posRef as! AXValue, .cgPoint, &point)
        AXValueGetValue(sizeRef as! AXValue, .cgSize, &size)

        return WindowGeometry(origin: point, size: size)
    }

    // MARK: - CGWindowList Fallback

    private func startFallbackDetectionTimer() {
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.checkFallback()
            }
        }
    }

    private func checkFallback() {
        // AX 이벤트가 N초간 없으면 CGWindowList로 전환
        if trackingMethod == .accessibility &&
           Date().timeIntervalSince(lastAXEventTime) > axTimeoutSeconds {

            // CGWindowList에서 위치 변화가 있는지 확인
            if let cgGeom = readCGWindowListGeometry(), cgGeom != lastGeometry {
                logger.info("AX timeout with position change detected, switching to CGWindowList")
                trackingMethod = .cgWindowList
                pollingTimer?.invalidate()
                startCGWindowListPolling()
            }
        }

        // CGWindowList 모드에서도 위치 갱신
        if trackingMethod == .cgWindowList {
            pollCGWindowList()
        }
    }

    private func startCGWindowListPolling() {
        trackingMethod = .cgWindowList
        pollingTimer?.invalidate()
        pollingTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pollCGWindowList()
            }
        }
        logger.info("CGWindowList polling started")
    }

    private func pollCGWindowList() {
        if let geom = readCGWindowListGeometry(), geom != lastGeometry {
            lastGeometry = geom
            onGeometryChanged?(geom)
        }
    }

    private func readCGWindowListGeometry() -> WindowGeometry? {
        let options: CGWindowListOption = [.optionIncludingWindow]
        guard let infoList = CGWindowListCopyWindowInfo(options, windowID) as? [[String: Any]],
              let info = infoList.first,
              let bounds = info[kCGWindowBounds as String] as? [String: CGFloat] else {
            return nil
        }

        let x = bounds["X"] ?? 0
        let y = bounds["Y"] ?? 0
        let w = bounds["Width"] ?? 0
        let h = bounds["Height"] ?? 0

        return WindowGeometry(origin: CGPoint(x: x, y: y), size: CGSize(width: w, height: h))
    }

    // MARK: - Public Polling

    /// 매 렌더 틱에서 호출 — CGWindowList로 최신 geometry 반환 (경량)
    public func pollGeometry() -> WindowGeometry? {
        return readCGWindowListGeometry()
    }

    // MARK: - Helpers

    private func findPID(for windowID: CGWindowID) -> pid_t? {
        let options: CGWindowListOption = [.optionIncludingWindow]
        guard let infoList = CGWindowListCopyWindowInfo(options, windowID) as? [[String: Any]],
              let info = infoList.first,
              let pid = info[kCGWindowOwnerPID as String] as? pid_t else {
            return nil
        }
        return pid
    }
}
