import SwiftUI
import Overlay

/// 설정 우선 화면 (Lossless Scaling 방식): 설정은 앱에서 미리, 시작은 포커스 창에 ⌃⌥⌘U.
/// 특정(비포커스) 창 캡처는 하단 "Capture a specific window"로.
struct WindowPickerView: View {
    @Bindable var appState: AppState

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
                        statusGrid
                    }
                    Divider()
                    shortcutsSection
                }
                .padding()
            }
        }
        .frame(width: 420, height: 520)
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
                HStack(spacing: 10) {
                    Image(systemName: "viewfinder").font(.title2).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Focus a window, press \(appState.hotCapture.label.isEmpty ? "the shortcut" : appState.hotCapture.label)")
                            .font(.headline)
                        Text("Settings below apply. Press again to stop.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
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

            if appState.selectedRenderMode == .metalFlow {
                Toggle("Occlusion warp (experimental)", isOn: $appState.occlusionDirectional)
                    .toggleStyle(.switch)
                    .onChange(of: appState.occlusionDirectional) { appState.updateOcclusionDirectional() }
                Text("Directional warp at reveal/cover edges — try on fast motion. Applies instantly, so in Cover mode you can toggle while watching; in fullscreen, set it before capture and rewind to compare.")
                    .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }

            Picker("Multiplier", selection: $appState.frameMultiplier) {
                Text("Auto").tag(0); Text("×2").tag(2); Text("×3").tag(3); Text("×4").tag(4); Text("×5").tag(5)
            }
            .pickerStyle(.segmented)
            .onChange(of: appState.frameMultiplier) { appState.persistSettings() }
            if appState.frameMultiplier >= 3 {
                Text("Capped at your display's refresh rate — 60fps ×3 = 180 needs a 180Hz+ display (120Hz shows 120).")
                    .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }

            Picker("Source", selection: $appState.sourcePreset) {
                Text("Off").tag(0); Text("360").tag(360); Text("480").tag(480)
                Text("540").tag(540); Text("720").tag(720); Text("1080").tag(1080)
            }
            .pickerStyle(.segmented)
            .onChange(of: appState.sourcePreset) {
                appState.persistSettings()
                if appState.isCapturing && appState.sourcePreset != 0 {
                    appState.resizeSourceToPreset(appState.sourcePreset)
                }
            }
            if appState.sourcePreset != 0 {
                Text("Resizes the source to this short side on capture (landscape: height, portrait: width). Set it here beforehand — the viewer covers this panel while capturing.")
                    .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }

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

    // MARK: - Shortcut (커스터마이징 — 캡처 토글 하나)

    private var shortcutsSection: some View {
        HStack {
            Text("Capture shortcut").font(.caption).foregroundStyle(.secondary)
            Spacer()
            ShortcutRecorder(binding: $appState.hotCapture)
                .frame(width: 96, height: 22)
                .onChange(of: appState.hotCapture) { appState.updateHotKeys() }
        }
    }
}
