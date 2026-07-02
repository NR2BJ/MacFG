import AppKit
import SwiftUI

/// NSStatusItem 메뉴바 관리
@MainActor
final class MenuBarManager {
    private var statusItem: NSStatusItem?
    private let appState: AppState
    private let openWindowAction: () -> Void

    init(appState: AppState, openWindow: @escaping () -> Void) {
        self.appState = appState
        self.openWindowAction = openWindow
    }

    func setup() {
        let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "display", accessibilityDescription: "MacFG")
        }
        self.statusItem = statusItem
        updateMenu()
    }

    func updateMenu() {
        let menu = NSMenu()

        if appState.isCapturing {
            let statusItem = NSMenuItem(title: "Capturing: \(appState.selectedWindowName)", action: nil, keyEquivalent: "")
            statusItem.isEnabled = false
            menu.addItem(statusItem)

            let methodItem = NSMenuItem(title: "Method: \(appState.captureMethod)", action: nil, keyEquivalent: "")
            methodItem.isEnabled = false
            menu.addItem(methodItem)

            let fpsItem = NSMenuItem(
                title: String(format: "FPS: %.1f → %.1f", appState.inputFPS, appState.outputFPS),
                action: nil, keyEquivalent: ""
            )
            fpsItem.isEnabled = false
            menu.addItem(fpsItem)

            let modeItem = NSMenuItem(title: "Mode: \(appState.selectedRenderMode.displayName)", action: nil, keyEquivalent: "")
            modeItem.isEnabled = false
            menu.addItem(modeItem)

            menu.addItem(.separator())

            let stopItem = NSMenuItem(title: "Stop Capture", action: #selector(stopCapture), keyEquivalent: "s")
            stopItem.target = self
            menu.addItem(stopItem)
        } else {
            let startItem = NSMenuItem(title: "Select Window...", action: #selector(showWindowPicker), keyEquivalent: "w")
            startItem.target = self
            menu.addItem(startItem)
        }

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit MacFG", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem?.menu = menu
    }

    @objc private func showWindowPicker() {
        openWindowAction()
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func stopCapture() {
        Task {
            await appState.stopCapture()
            updateMenu()
        }
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }
}
