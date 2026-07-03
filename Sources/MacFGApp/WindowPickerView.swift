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
                if let scale = appState.upscaleStatus {
                    GridRow {
                        Text("Scale")
                            .foregroundStyle(.secondary)
                        Text(scale)
                            .fontWeight(.medium)
                            .font(.caption)
                    }
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

                // 배율: Auto = 디스플레이 슬롯 전부 채움 (30fps→120Hz면 4x),
                // ×N = 소스 fps × N 상한 (예: 30fps ×2 → 60fps에서 멈춤)
                Picker("Multiplier", selection: $appState.frameMultiplier) {
                    Text("Auto").tag(0)
                    Text("×2").tag(2)
                    Text("×3").tag(3)
                    Text("×4").tag(4)
                    Text("×5").tag(5)
                }
                .pickerStyle(.segmented)

                Picker("Output", selection: $appState.selectedOverlayPlacement) {
                    ForEach(OverlayPlacement.allCases) { placement in
                        Text(placement.displayName).tag(placement)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: appState.selectedOverlayPlacement) {
                    appState.updateOverlayPlacement()
                }

                // 업스케일 방식: Off / ANE(신경망 2x) / MetalFX / ANE+FX(체이닝).
                // 뷰어를 소스보다 크게(최대화·전체화면) 했을 때만 실효.
                Picker("Upscale", selection: $appState.upscaleMode) {
                    ForEach(UpscaleMode.allCases) { m in
                        Text(m.displayName).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: appState.upscaleMode) {
                    appState.updateUpscale()
                }

                // CAS 샤픈은 업스케일과 독립 (Cover 1:1 포함 어디서나 — 늘어난 저해상도 영상 복원)
                Toggle("Sharpen (CAS)", isOn: $appState.casEnabled)
                    .toggleStyle(.switch)
                    .onChange(of: appState.casEnabled) {
                        appState.updateUpscale()
                    }
                if appState.casEnabled {
                    HStack(spacing: 8) {
                        Text("Sharpness")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Slider(value: $appState.sharpness, in: 0...1)
                            .onChange(of: appState.sharpness) {
                                appState.updateUpscale()
                            }
                        Text(String(format: "%.1f", appState.sharpness))
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                Text("Upscale needs Separate Window enlarged past the source (≤960px sources use the Neural Engine). Maximize or fullscreen the viewer to fill your display. The Scale row shows what's active.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                // 전역 단축키 안내 (Cover 배치 전용 — 전체화면에서 창 없이 제어).
                // 오버레이는 소스 앱이 최전면일 때만 표시되고, 벗어나면 자동으로 숨는다.
                if appState.selectedOverlayPlacement == .coverSource {
                    Label("⌃⌥⌘I toggle overlay · ⌃⌥⌘. stop · hides when you switch apps",
                          systemImage: "keyboard")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            Spacer()

            // 맨 아래 고정 안내 — Apple FI는 2배 보간 고정이라 주사율 정합이 중요.
            // (Metal Flow는 갭 채움으로 임의 주사율 대응 → 안내 불필요)
            if appState.isInterpolationEnabled && appState.selectedRenderMode == .appleFI {
                Text("Apple FI is fixed 2× interpolation. For smoothest playback, set your display refresh rate to an integer multiple of source fps × 2 (60fps → 120Hz, 24fps → 144Hz).")
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
