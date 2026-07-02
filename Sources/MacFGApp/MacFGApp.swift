import SwiftUI
import AppKit

@main
struct MacFGApp: App {
    @State private var appState = AppState()
    @State private var menuBarManager: MenuBarManager?
    @Environment(\.openWindow) private var openWindow

    init() {
        // 번들 앱이 아닌 경우에도 Dock + 메뉴바에 표시되도록
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup(id: "main") {
            WindowPickerView(appState: appState)
                .onAppear {
                    setupMenuBar()
                    checkPermissions()
                    Task { await appState.processAutoStartArguments() }
                }
                .onChange(of: appState.isCapturing) {
                    menuBarManager?.updateMenu()
                }
        }
        .windowResizability(.contentSize)
    }

    private func setupMenuBar() {
        guard menuBarManager == nil else { return }
        let manager = MenuBarManager(appState: appState) { [openWindow] in
            openWindow(id: "main")
        }
        manager.setup()
        menuBarManager = manager
    }

    private func checkPermissions() {
        // 접근성 권한 확인
        let trusted = AXIsProcessTrusted()
        if !trusted {
            // 권한 요청 프롬프트 트리거
            let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            AXIsProcessTrustedWithOptions(options)
        }
    }
}
