import SwiftUI
import AppKit

@main
struct MacFGApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        // 메뉴바 상주 팝오버 (Tailscale/AlDente 스타일) — 아이콘 클릭 시 설정/대시보드가
        // 바로 아래로 펼쳐진다. 별도 창 없음 → 빨간 닫기 버튼으로 종료되는 문제 자체가 사라짐.
        MenuBarExtra {
            WindowPickerView(appState: delegate.appState)
        } label: {
            // 캡처 중이면 배지 아이콘으로 상태 표시 (관찰 뷰)
            MenuBarLabel(appState: delegate.appState)
        }
        .menuBarExtraStyle(.window)
    }
}

/// 메뉴바 아이콘 — 캡처 상태를 반영
private struct MenuBarLabel: View {
    @Bindable var appState: AppState
    var body: some View {
        Image(systemName: appState.isCapturing ? "display.trianglebadge.exclamationmark" : "display")
    }
}

/// 런치타임 셋업은 AppDelegate에서 (MenuBarExtra 콘텐츠 onAppear는 첫 클릭 시에야 실행되므로
/// 핫키·auto-start를 여기서 처리해야 앱 시작 즉시 동작한다). AppState도 여기 소유해 뷰와 공유.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState = AppState()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // 메뉴바 전용 — Dock 아이콘·⌘Tab 제거. 창 없이 상주하므로 닫기로 종료되지 않는다.
        NSApplication.shared.setActivationPolicy(.accessory)

        // 접근성 권한 (마우스 역매핑용) — 없으면 프롬프트
        if !AXIsProcessTrusted() {
            AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary)
        }

        appState.registerHotKeys()
        Task { await appState.processAutoStartArguments() }
    }

    // 마지막 창(뷰어)을 닫아도 앱은 메뉴바에 상주 — 종료는 팝오버의 Quit 버튼으로만.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}
