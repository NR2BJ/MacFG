import SwiftUI
import AppKit
import Monitoring

/// 메뉴바 상주 앱.
///
/// **왜 SwiftUI MenuBarExtra를 쓰지 않는가 (2026-07-25 실측):**
/// MenuBarExtra는 앱의 유일한 Scene이 되는데, macOS가 그 상태항목 씬을 파괴하기로 결정하면
/// (아이콘이 "숨김"으로 기록됐거나 메뉴바가 포화일 때) SwiftUI 구현이 그대로 `NSApplication.terminate:`를
/// 호출해 **앱이 시작 직후 조용히 종료**된다. 실제로 그 상태에 빠져 앱이 아예 실행 불가가 됐고,
/// 스택으로 확인했다:
///     -[NSSceneStatusItem scene:handleActions:] → -[NSApplication terminate:] → applicationShouldTerminate
/// 크래시도 로그도 없이 exit(0)이라 원인 파악이 어려웠다. 사용자 설정
/// (`NSStatusItem VisibleCC Item-0`)을 1로 되돌리거나 재부팅해도 복구되지 않았다.
/// 그래서 상태항목을 **직접 만들어 소유**한다 — 씬 생명주기에 종속되지 않아 이 실패 모드가 사라진다.
@main
struct MacFGApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        // 표시되지 않는 빈 씬 — SwiftUI App은 Scene이 최소 하나 필요하다.
        // 실제 UI는 AppDelegate가 소유한 NSStatusItem + NSPopover가 담당한다.
        Settings { EmptyView() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate {
    let appState = AppState()
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var iconTimer: Timer?
    /// 사용자가 Quit을 눌렀는가 — 시스템發 종료 요청과 구분한다.
    private var userRequestedQuit = false
    private var systemIsPoweringOff = false

    override init() {
        super.init()
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willPowerOffNotification, object: nil, queue: .main
        ) { [weak self] _ in MainActor.assumeIsolated { self?.systemIsPoweringOff = true } }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // 메뉴바 전용 — Dock 아이콘·⌘Tab 제거. 창 없이 상주하므로 닫기로 종료되지 않는다.
        NSApplication.shared.setActivationPolicy(.accessory)
        setUpStatusItem()

        // 접근성 권한 (마우스 역매핑용) — 없으면 프롬프트
        if !AXIsProcessTrusted() {
            AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary)
        }

        appState.onQuitRequested = { [weak self] in self?.quitFromUser() }
        appState.registerHotKeys()
        Task { await appState.processAutoStartArguments() }
    }

    // MARK: - 상태 항목 (직접 소유)

    private func setUpStatusItem() {
        // **번들 밖(.build/release/MacFGApp 직접 실행)에서는 상태항목을 만들지 않는다.**
        // macOS 26의 ControlCenter는 메뉴바 항목 허용 목록을 앱이 아니라 **띄운 프로세스**
        // (responsible process) 기준으로 기록한다. 번들 없이 셸에서 실행하면 그룹 컨테이너
        // group.com.apple.controlcenter의 trackedApplications에 `adhocBinary .build/...` 소유자
        // 기록이 새로 생기고, 같은 메뉴 항목을 여러 소유자가 상충하는 허용 상태로 주장하게 되면
        // ControlCenter가 **아무것도 채택하지 않는다** — 그러면 아이콘이 영영 안 뜬다.
        // (실측 2026-07-25: 디버깅 중 셸/에이전트에서 반복 실행해 소유자 기록이 6개까지 늘었고,
        //  코드는 그대로인데 같은 커밋이 30분 만에 정상→고장으로 바뀌었다.)
        // 개발 중 셸 실행은 ⌃⌥⌘M(설정 창)으로 쓴다.
        guard Bundle.main.bundleIdentifier != nil else {
            DiagnosticLog.shared.log("[SI] 번들 밖 실행 — 상태항목 생성 생략 (ControlCenter 기록 오염 방지). ⌃⌥⌘M 사용")
            return
        }
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.image = NSImage(systemSymbolName: "display", accessibilityDescription: "MacFG")
        item.button?.target = self
        item.button?.action = #selector(togglePopover(_:))
        // **고유 autosaveName 필수** — 지정하지 않으면 macOS가 익명 슬롯(Item-N)을 배정하는데,
        // 제어센터가 그 슬롯들을 전부 "숨김"으로 기록해 두면(실측: Item-0~Item-103 전부 0)
        // 아이콘이 영영 안 보인다. 시스템 설정 메뉴막대에 같은 앱이 여러 개로 뜨는 것도 같은 원인.
        // 이름을 주면 자기 항목("MacFG")을 갖고 표시 상태가 그 이름으로 저장된다.
        item.autosaveName = "MacFG"
        item.isVisible = true      // 숨김으로 기록돼 있던 상태를 매 실행 되돌린다
        statusItem = item

        let pop = NSPopover()
        pop.behavior = .transient
        pop.animates = false
        pop.delegate = self
        pop.contentSize = NSSize(width: 440, height: 592)
        pop.contentViewController = NSHostingController(rootView: WindowPickerView(appState: appState))
        popover = pop

        verifyStatusItemAdopted()

        // 캡처 상태를 아이콘에 반영 — 상태 변화가 드물어 1초 폴링으로 충분
        iconTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let btn = self.statusItem?.button else { return }
                let name = self.appState.isCapturing ? "display.trianglebadge.exclamationmark" : "display"
                if btn.image?.accessibilityDescription != name {
                    let img = NSImage(systemSymbolName: name, accessibilityDescription: name)
                    btn.image = img
                }
            }
        }
    }

    /// 상태항목이 **실제로 메뉴바에 올라갔는지** 확인하고, 아니면 설정 창을 대신 띄운다.
    ///
    /// 왜 필요한가: macOS 26에서 메뉴바 상태항목 창은 ControlCenter가 소유·호스팅한다. 우리
    /// NSStatusItem은 그 채택을 신청하는 프록시일 뿐이고, 채택이 거부돼도 **AppKit은 아무것도
    /// 알려주지 않는다** — isVisible·window·onScreen 전부 true를 계속 반환한다(프록시 자신을
    /// 설명하는 값이라서). 실제로 2026-07-25에 이 상태에 빠졌고, 아이콘이 없으니 사용자에겐
    /// "앱이 아예 안 켜진다"로 보였다. 조용한 실패를 눈에 보이는 실패로 바꾼다.
    ///
    /// 판정: 창 서버에 우리 pid 소유의 상태항목 레이어(25) 창이 있는가. 채택되면 창은
    /// ControlCenter 소유가 되므로 0개가 정상… 이 아니라, **채택 실패 시에도 0개**다.
    /// 그래서 창 서버가 아니라 프록시 창의 기하로 가른다: 채택 못 받은 항목은 화면 오른쪽
    /// 끝에 딱 붙고(가장자리로부터 자기 폭만큼), 높이가 구형 22pt로 남는다 — Tahoe의 실제
    /// 메뉴바 띠는 30pt 이상이다.
    private func verifyStatusItemAdopted() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self, let win = self.statusItem?.button?.window, let screen = win.screen else { return }
            let menuBarHeight = screen.frame.height - screen.visibleFrame.height - (screen.visibleFrame.origin.y - screen.frame.origin.y)
            let flushToRightEdge = (screen.frame.maxX - win.frame.maxX) < 1.0
            let legacyHeight = menuBarHeight > 0 && win.frame.height < menuBarHeight - 1.0
            guard flushToRightEdge && legacyHeight else { return }

            DiagnosticLog.shared.log("""
                [SI] 메뉴바 채택 실패 — ControlCenter가 상태항목을 받아주지 않았다 \
                (frame=\(win.frame) 메뉴바높이=\(menuBarHeight)). 설정 창으로 대체한다. \
                복구: scripts/repair_menubar_allowlist.py --apply && killall ControlCenter
                """)
            self.appState.openSettingsWindow()
        }
    }

    @objc private func togglePopover(_ sender: Any?) {
        guard let pop = popover, let btn = statusItem?.button else { return }
        if pop.isShown {
            pop.performClose(sender)
        } else {
            pop.show(relativeTo: btn.bounds, of: btn, preferredEdge: .minY)
            pop.contentViewController?.view.window?.makeKey()
        }
    }

    func popoverDidShow(_ notification: Notification) { appState.popoverVisible = true }
    func popoverDidClose(_ notification: Notification) { appState.popoverVisible = false }

    // MARK: - 종료 정책

    func quitFromUser() {
        userRequestedQuit = true
        NSApplication.shared.terminate(nil)
    }

    /// 사용자 Quit과 시스템 로그아웃/재시동만 허용한다. 상태항목 씬 파괴 같은 이유로 오는
    /// 종료 요청은 거부해 앱을 살려둔다(위 주석의 실패 모드 방어).
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if userRequestedQuit || systemIsPoweringOff { return .terminateNow }
        DiagnosticLog.shared.log("[APP] 시스템發 종료 요청 무시 — 메뉴바 상주 유지")
        return .terminateCancel
    }

    // 마지막 창(뷰어)을 닫아도 앱은 메뉴바에 상주 — 종료는 팝오버의 Quit 버튼으로만.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}
