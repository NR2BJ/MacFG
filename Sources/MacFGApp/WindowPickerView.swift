import SwiftUI
import Overlay

/// 설정 우선 화면 (Lossless Scaling 방식): 설정은 앱에서 미리, 시작은 포커스 창에 ⌃⌥⌘U.
/// 특정(비포커스) 창 캡처는 하단 "Capture a specific window"로.
struct WindowPickerView: View {
    @Bindable var appState: AppState
    @State private var showWindowList = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    startStatusCard
                    Divider()
                    settingsControls
                    if appState.isCapturing {
                        sourcePresetsRow
                        statusGrid
                    }
                    Divider()
                    specificWindowDisclosure
                    shortcutsSection
                }
                .padding()
            }
        }
        .frame(width: 420, height: 560)
        .onAppear { appState.refreshWindowList() }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("MacFG").font(.title2).fontWeight(.bold)
            Spacer()
            if appState.isCapturing { statusBadge }
        }
        .padding()
    }

    private var statusBadge: some View {
        HStack(spacing: 4) {
            Circle().fill(.green).frame(width: 8, height: 8)
            Text("Active").font(.caption).foregroundStyle(.green)
        }
    }

    // MARK: - Start / Status

    private var startStatusCard: some View {
        Group {
            if appState.isCapturing {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(appState.selectedWindowName).font(.headline).lineLimit(1)
                        Text("\(appState.captureMethod) · \(appState.interpolationEngine)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Stop") { Task { await appState.stopCapture() } }
                        .buttonStyle(.borderedProminent).tint(.red)
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Button {
                        appState.toggleCaptureFocused()
                    } label: {
                        Label("Capture Focused Window", systemImage: "viewfinder")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    Text("Focus the window you want, then click above or press \(appState.hotCapture.label). Settings below apply.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Settings (항상 편집 가능)

    private var settingsControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("Frame Interpolation", isOn: $appState.isInterpolationEnabled)
                .toggleStyle(.switch)
                .onChange(of: appState.isInterpolationEnabled) { appState.updateInterpolationEnabled() }

            Picker("Engine", selection: $appState.selectedRenderMode) {
                ForEach(RenderMode.userSelectable) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)
            .onChange(of: appState.selectedRenderMode) { appState.updateRenderMode() }

            Picker("Multiplier", selection: $appState.frameMultiplier) {
                Text("Auto").tag(0); Text("×2").tag(2); Text("×3").tag(3); Text("×4").tag(4); Text("×5").tag(5)
            }
            .pickerStyle(.segmented)

            Picker("Upscale", selection: $appState.upscaleMode) {
                ForEach(UpscaleMode.allCases) { Text($0.displayName).tag($0) }
            }
            .pickerStyle(.segmented)
            .onChange(of: appState.upscaleMode) {
                appState.updateUpscale()
                appState.autoSelectPlacementForUpscale()
            }
            Text(appState.upscaleMode == .off
                 ? "Off: overlay sits on the source window (interpolation only)."
                 : "Upscaling shows a separate maximized window (small source → big sharp output).")
                .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)

            Toggle("Sharpen (CAS)", isOn: $appState.casEnabled)
                .toggleStyle(.switch)
                .onChange(of: appState.casEnabled) { appState.updateUpscale() }
            if appState.casEnabled {
                HStack(spacing: 8) {
                    Text("Sharpness").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $appState.sharpness, in: 0...1)
                        .onChange(of: appState.sharpness) { appState.updateUpscale() }
                    Text(String(format: "%.1f", appState.sharpness))
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }

            if appState.isInterpolationEnabled && appState.selectedRenderMode == .appleFI {
                Text("Apple FI is fixed 2× interpolation. Set the display to an integer multiple of source fps × 2 (60→120Hz, 24→144Hz).")
                    .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: - During capture

    private var sourcePresetsRow: some View {
        HStack(spacing: 6) {
            Text("Source").font(.caption).foregroundStyle(.secondary)
            ForEach([360, 480, 540, 720, 1080], id: \.self) { h in
                Button("\(h)p") { appState.resizeSourceToHeight(h) }
                    .buttonStyle(.bordered).controlSize(.small)
            }
        }
    }

    private var statusGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 6) {
            GridRow {
                Text("FPS").foregroundStyle(.secondary)
                Text(String(format: "%.1f → %.1f", appState.inputFPS, appState.outputFPS))
                    .fontWeight(.medium).monospacedDigit()
            }
            GridRow {
                Text("Latency").foregroundStyle(.secondary)
                Text(String(format: "%.0f ms", appState.latencyMs)).fontWeight(.medium).monospacedDigit()
            }
            if let scale = appState.upscaleStatus {
                GridRow {
                    Text("Scale").foregroundStyle(.secondary)
                    Text(scale).font(.caption)
                }
            }
        }
        .font(.callout)
    }

    // MARK: - Secondary: specific window

    private var specificWindowDisclosure: some View {
        DisclosureGroup(isExpanded: $showWindowList) {
            VStack(spacing: 2) {
                Button {
                    appState.refreshWindowList()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise").font(.caption)
                }
                .buttonStyle(.borderless)
                .frame(maxWidth: .infinity, alignment: .trailing)

                ForEach(appState.availableWindows) { window in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(window.displayName).font(.caption).fontWeight(.medium).lineLimit(1)
                            Text("\(window.width) x \(window.height)").font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Capture") {
                            appState.selectedWindowID = window.windowID
                            appState.selectedWindowName = window.displayName
                            Task { await appState.startCapture() }
                        }
                        .buttonStyle(.bordered).controlSize(.small)
                    }
                    .padding(.vertical, 1)
                }
            }
            .padding(.top, 4)
        } label: {
            Text("Capture a specific window").font(.caption)
        }
    }

    // MARK: - Shortcuts (녹화식 커스터마이징)

    private var shortcutsSection: some View {
        DisclosureGroup("Shortcuts") {
            VStack(spacing: 6) {
                shortcutRow("Capture focused (toggle)", binding: $appState.hotCapture)
                shortcutRow("Toggle overlay", binding: $appState.hotToggle)
                shortcutRow("Stop capture", binding: $appState.hotStop)
            }
            .padding(.top, 4)
        }
        .font(.caption)
    }

    private func shortcutRow(_ label: String, binding: Binding<HotKeyBinding>) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            ShortcutRecorder(binding: binding)
                .frame(width: 96, height: 22)
                .onChange(of: binding.wrappedValue) { appState.updateHotKeys() }
        }
    }
}
