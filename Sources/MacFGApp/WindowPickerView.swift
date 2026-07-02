import SwiftUI
import Overlay

/// 캡처 대상 창 선택 화면
struct WindowPickerView: View {
    @Bindable var appState: AppState

    var body: some View {
        VStack(spacing: 0) {
            // 헤더
            HStack {
                Text("MacFG")
                    .font(.title2)
                    .fontWeight(.bold)

                Spacer()

                if appState.isCapturing {
                    statusBadge
                }

                Button {
                    appState.refreshWindowList()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
            }
            .padding()

            Divider()

            if appState.isCapturing {
                captureStatusView
            } else {
                windowListView
            }
        }
        .frame(width: 400, height: 400)
        .onAppear {
            appState.refreshWindowList()
        }
    }

    // MARK: - Capture Status

    private var captureStatusView: some View {
        VStack(spacing: 14) {
            // 정지 버튼을 최상단에 — 콘텐츠가 길어져도 항상 보이도록
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(appState.selectedWindowName)
                        .font(.headline)
                        .lineLimit(1)
                    Text("\(appState.captureMethod) · \(appState.interpolationEngine)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Stop Capture") {
                    Task {
                        await appState.stopCapture()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            }

            Divider()

            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
                GridRow {
                    Text("FPS")
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.1f → %.1f", appState.inputFPS, appState.outputFPS))
                        .fontWeight(.medium)
                        .monospacedDigit()
                }
                GridRow {
                    Text("Latency")
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.0f ms", appState.latencyMs))
                        .fontWeight(.medium)
                        .monospacedDigit()
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Divider()

            VStack(alignment: .leading, spacing: 10) {
                Toggle("Frame Interpolation", isOn: $appState.isInterpolationEnabled)
                    .toggleStyle(.switch)
                    .onChange(of: appState.isInterpolationEnabled) {
                        appState.updateInterpolationEnabled()
                    }

                Picker("Engine", selection: $appState.selectedRenderMode) {
                    ForEach(RenderMode.userSelectable) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: appState.selectedRenderMode) {
                    appState.updateRenderMode()
                }

                Picker("Output", selection: $appState.selectedOverlayPlacement) {
                    ForEach(OverlayPlacement.allCases) { placement in
                        Text(placement.displayName).tag(placement)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: appState.selectedOverlayPlacement) {
                    appState.updateOverlayPlacement()
                }
            }

            Spacer()

            // 맨 아래 고정 안내 — Apple FI는 2배 보간 고정이라 주사율 정합이 중요.
            // (Metal Flow는 갭 채움으로 임의 주사율 대응 → 안내 불필요)
            if appState.isInterpolationEnabled && appState.selectedRenderMode == .appleFI {
                Text("Apple FI는 2배 보간 고정입니다. 모니터 주사율을 원본 fps×2의 정수배로 설정하면 가장 부드럽습니다 (60fps→120Hz, 24fps→144Hz).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }

    // MARK: - Window List

    private var windowListView: some View {
        Group {
            if appState.availableWindows.isEmpty {
                ContentUnavailableView(
                    "No Windows Found",
                    systemImage: "macwindow",
                    description: Text("Open an application window and click refresh.")
                )
            } else {
                List(appState.availableWindows) { window in
                    windowRow(window)
                }
            }
        }
    }

    private func windowRow(_ window: WindowInfo) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text(window.displayName)
                    .fontWeight(.medium)
                    .lineLimit(1)
                Text("\(window.width) x \(window.height)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Capture") {
                appState.selectedWindowID = window.windowID
                appState.selectedWindowName = window.displayName
                Task {
                    await appState.startCapture()
                }
            }
            .buttonStyle(.bordered)
        }
        .padding(.vertical, 2)
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(.green)
                .frame(width: 8, height: 8)
            Text("Active")
                .font(.caption)
                .foregroundStyle(.green)
        }
    }
}
