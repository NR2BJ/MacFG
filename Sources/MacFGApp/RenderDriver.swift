import AppKit
import QuartzCore
import os
import Monitoring

/// 전용 렌더 스레드 + CAMetalDisplayLink 드라이버 (A2의 심장).
///
/// 목적 — 실측으로 확정된 두 구조 병목을 동시에 제거:
///  1. NSScreen.displayLink는 AppKit UpdateCycle의 deferred 배달이라 앱이 한가해도
///     vsync 콜백이 스킵됨 (tick 117Hz 천장, sample로 확증) → 링크를 전용 스레드
///     런루프에 붙여 AppKit/SwiftUI와 완전 분리.
///  2. nextDrawable 세마포어 대기 (10s 중 0.7-1.4s 실측) → CAMetalDisplayLink는
///     vsync마다 드로어블을 콜백으로 직접 배달 (대기 개념 자체가 없음).
///
/// handler는 렌더 스레드에서 실행된다. 스레드는 최초 attach에서 기동 후 재사용.
final class RenderDriver: NSObject, CAMetalDisplayLinkDelegate, @unchecked Sendable {
    struct Tick {
        let drawable: any CAMetalDrawable
        /// 이 업데이트의 기준 시각 (콜백 발화 기준)
        let timestamp: CFTimeInterval
        /// 이 드로어블이 실제 표시될 vsync 시각 — 스케줄러의 targetTimestamp
        let targetPresentTimestamp: CFTimeInterval
    }

    private let logger = Logger(subsystem: "com.macfg", category: "RenderDriver")
    private let lock = NSLock()
    private var thread: Thread?
    private var runLoop: CFRunLoop?
    private let threadReady = DispatchSemaphore(value: 0)
    private var link: CAMetalDisplayLink?
    private var handler: ((Tick) -> Void)?

    /// 렌더 스레드 기동 (1회) — 런루프를 더미 소스로 유지
    private func ensureThread() {
        lock.lock()
        let started = thread != nil
        lock.unlock()
        guard !started else { return }

        let t = Thread { [weak self] in
            guard let self else { return }
            let rl = CFRunLoopGetCurrent()
            self.lock.lock(); self.runLoop = rl; self.lock.unlock()
            self.threadReady.signal()
            var ctx = CFRunLoopSourceContext()
            if let src = CFRunLoopSourceCreate(nil, 0, &ctx) {
                CFRunLoopAddSource(rl, src, .defaultMode)
            }
            CFRunLoopRun()
        }
        t.name = "MacFG.Render"
        t.qualityOfService = .userInteractive
        lock.lock(); thread = t; lock.unlock()
        t.start()
        threadReady.wait()
        logger.info("Render thread started")
    }

    /// 렌더 스레드에서 블록 동기 실행 — 스케줄러 상태 리셋을 틱과 직렬화하는 데 사용.
    /// 틱(metalDisplayLink 콜백)과 이 블록은 같은 렌더 스레드 런루프에서 순차 실행되므로
    /// 절대 겹치지 않는다 → timeline 등 렌더 상태를 메인에서 직접 비우던 레이스(크래시) 제거.
    func perform(_ block: @escaping @Sendable () -> Void) {
        performSync(block)
    }

    /// 렌더 런루프 기동 여부 — perform()이 no-op일지 판별용 (미기동이면 틱이 없어
    /// 호출측이 직접 상태를 만져도 안전). 런루프는 한 번 설정되면 해제되지 않는다.
    var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return runLoop != nil
    }

    /// 렌더 스레드에서 블록 실행 (완료 대기 없음) — 캡처 콜백처럼 "빨리 태우고 빠지는" 경로용.
    ///
    /// performSync와 같은 런루프에 올라가므로 **틱과 절대 겹치지 않는다** → 렌더 상태의
    /// 무락 전제(timeline/prevStable/snap 링 등을 렌더 스레드만 만진다)가 그대로 보존된다.
    /// 대기하지 않으므로 캡처 스레드를 막지 않고, 실패(런루프 미기동) 시 조용히 버린다 —
    /// 그 경우 렌더 틱의 기존 drain 경로가 같은 프레임을 가져간다.
    /// @discardableResult로 "태워졌는지"를 알려 호출측이 폴백을 판단할 수 있게 한다.
    @discardableResult
    func performAsync(_ block: @escaping @Sendable () -> Void) -> Bool {
        lock.lock(); let rl = runLoop; lock.unlock()
        guard let rl else { return false }
        CFRunLoopPerformBlock(rl, CFRunLoopMode.defaultMode.rawValue, block)
        CFRunLoopWakeUp(rl)
        return true
    }

    /// 렌더 스레드에서 블록 실행 (완료 대기)
    private func performSync(_ block: @escaping () -> Void) {
        lock.lock(); let rl = runLoop; lock.unlock()
        guard let rl else { return }
        if CFRunLoopGetCurrent() === rl { block(); return }
        let done = DispatchSemaphore(value: 0)
        CFRunLoopPerformBlock(rl, CFRunLoopMode.defaultMode.rawValue) {
            block()
            done.signal()
        }
        CFRunLoopWakeUp(rl)
        done.wait()
    }

    /// 레이어에 링크 부착 (기존 링크는 교체). handler는 렌더 스레드에서 vsync마다 호출.
    func attach(layer: CAMetalLayer, handler newHandler: @escaping (Tick) -> Void) {
        ensureThread()
        performSync { [weak self] in
            guard let self else { return }
            self.link?.invalidate()
            self.lock.lock(); self.handler = newHandler; self.lock.unlock()
            let link = CAMetalDisplayLink(metalLayer: layer)
            link.delegate = self
            link.add(to: RunLoop.current, forMode: .default)
            self.link = link
            DiagnosticLog.shared.log("[DRIVER] link attached (layer=\(Int(layer.drawableSize.width))x\(Int(layer.drawableSize.height)), thread=\(Thread.current.name ?? "?"))")
        }
    }

    /// 링크 해제 — 반환 시점 이후 handler 호출 없음 보장 (동기)
    func detach() {
        performSync { [weak self] in
            guard let self else { return }
            self.link?.invalidate()
            self.link = nil
            self.lock.lock(); self.handler = nil; self.lock.unlock()
            self.logger.info("CAMetalDisplayLink detached")
        }
    }

    // MARK: - CAMetalDisplayLinkDelegate (렌더 스레드에서 호출됨)

    private var cbCount = 0
    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        cbCount += 1
        if cbCount <= 3 || cbCount % 600 == 0 {
            DiagnosticLog.shared.log("[DRIVER] update #\(cbCount) target=\(update.targetPresentationTimestamp)")
        }
        lock.lock(); let h = handler; lock.unlock()
        h?(Tick(
            drawable: update.drawable,
            timestamp: update.targetTimestamp,
            targetPresentTimestamp: update.targetPresentationTimestamp
        ))
    }
}
