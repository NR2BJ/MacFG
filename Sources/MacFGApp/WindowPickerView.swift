import SwiftUI
import Overlay

/// 설정 우선 화면 (Lossless Scaling 방식): 설정은 앱에서 미리, 시작은 포커스 창에 단축키.
/// 섹션 그룹 + 상세 설명은 컨트롤 `.help()` 툴팁(호버)으로 — 패널을 깔끔하게 유지.
struct WindowPickerView: View {
    @Bindable var appState: AppState
    @State private var showAdvanced = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                statusCard
                interpolationSection
                upscalingSection
                if appState.isCapturing { liveSection }
                shortcutSection
            }
            .padding(18)
        }
        .frame(width: 430, height: 540)
        .background(.background)
    }

    // MARK: - Reusable section shell

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.caption).fontWeight(.semibold)
                .foregroundStyle(.secondary).textCase(.uppercase)
                .tracking(0.5)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    /// 라벨 + 컨트롤 한 줄 (라벨 고정폭으로 정렬)
    @ViewBuilder
    private func row<Control: View>(_ label: String, help: String? = nil, @ViewBuilder _ control: () -> Control) -> some View {
        HStack(spacing: 10) {
            Text(label).frame(width: 74, alignment: .leading)
            control()
        }
        .help(help ?? "")
    }

    // MARK: - Status hero

    private var statusCard: some View {
        Group {
            if appState.isCapturing {
                HStack(spacing: 10) {
                    Circle().fill(.green).frame(width: 9, height: 9)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(appState.selectedWindowName).font(.headline).lineLimit(1)
                        Text("\(appState.captureMethod) · \(appState.interpolationEngine)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Stop") { Task { await appState.stopCapture() } }
                        .buttonStyle(.borderedProminent).tint(.red).controlSize(.large)
                }
            } else {
                HStack(spacing: 12) {
                    Image(systemName: "viewfinder")
                        .font(.system(size: 26)).foregroundStyle(.tint)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Focus a window, press \(appState.hotCapture.label.isEmpty ? "the shortcut" : appState.hotCapture.label)")
                            .font(.headline)
                        Text("Set it up below · press again to stop")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(appState.isCapturing ? AnyShapeStyle(.green.opacity(0.12)) : AnyShapeStyle(.tint.opacity(0.10)),
                    in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Interpolation

    private var interpolationSection: some View {
        section("Interpolation") {
            Toggle("Frame interpolation", isOn: $appState.isInterpolationEnabled)
                .onChange(of: appState.isInterpolationEnabled) { appState.updateInterpolationEnabled() }

            row("Engine", help: "Metal Flow: our GPU pipeline, any multiplier, keeps native sharpness. Apple FI: Apple's ANE model, fixed 2× at 720p — set the display to fps×2 (60→120, 24→144).") {
                Picker("", selection: $appState.selectedRenderMode) {
                    ForEach(RenderMode.userSelectable) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.selectedRenderMode) { appState.updateRenderMode() }
            }

            row("Multiplier", help: "Output = source fps × N, capped at your display refresh (60 ×3 = 180 needs 180Hz+; 120Hz shows 120). Auto fills every display slot.") {
                Picker("", selection: $appState.frameMultiplier) {
                    Text("Auto").tag(0); Text("×2").tag(2); Text("×3").tag(3); Text("×4").tag(4); Text("×5").tag(5)
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.frameMultiplier) { appState.persistSettings() }
            }

            if appState.selectedRenderMode == .metalFlow {
                Divider().padding(.vertical, 2)
                sliderRow("Motion", low: "sharp", high: "smooth", value: $appState.motionSmoothness,
                          help: "Flow character. Sharp keeps motion detail (can shimmer); smooth is gentler/softer. Taste, not quality.") {
                    appState.updateMotionSmoothness()
                }
                sliderRow("Edges", low: "crisp", high: "soft", value: $appState.boundarySoftness,
                          help: "Object-boundary handling. Crisp = less ghosting, slight judder (games/fast action); soft = smoother, slight ghosting (film/slow pans).") {
                    appState.updateBoundarySoftness()
                }

                DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                    Toggle("Occlusion warp (experimental)", isOn: $appState.occlusionDirectional)
                        .onChange(of: appState.occlusionDirectional) { appState.updateOcclusionDirectional() }
                        .help("Directional warp at reveal/cover edges. Off by default — helps some fast motion but can shimmer on repetitive patterns.")
                        .padding(.top, 4)
                }
                .font(.callout)
            }
        }
    }

    /// low↔high 라벨이 붙은 슬라이더 한 줄
    @ViewBuilder
    private func sliderRow(_ label: String, low: String, high: String, value: Binding<Double>, help: String, _ onChange: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Text(label).frame(width: 52, alignment: .leading)
            Text(low).font(.caption2).foregroundStyle(.secondary)
            Slider(value: value, in: 0...1).onChange(of: value.wrappedValue) { onChange() }
            Text(high).font(.caption2).foregroundStyle(.secondary)
        }
        .help(help)
    }

    // MARK: - Upscaling

    private var upscalingSection: some View {
        section("Upscaling & sharpness") {
            row("Upscale", help: "Off: overlay sits 1:1 on the source (interpolation only). ANE/MetalFX/ANE+FX: show a fullscreen viewer that scales a small source up. ANE needs a ≤960px source.") {
                Picker("", selection: $appState.upscaleMode) {
                    ForEach(UpscaleMode.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.upscaleMode) {
                    appState.updateUpscale()
                    appState.autoSelectPlacementForUpscale()
                }
            }

            row("Source", help: "Resize the source to this short side on capture (landscape: height, portrait: width) so it renders at native res for a clean 1:1 capture. Great for browser PiP / IINA.") {
                Picker("", selection: $appState.sourcePreset) {
                    Text("Off").tag(0); Text("360").tag(360); Text("480").tag(480)
                    Text("540").tag(540); Text("720").tag(720); Text("1080").tag(1080)
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.sourcePreset) {
                    appState.persistSettings()
                    if appState.isCapturing && appState.sourcePreset != 0 {
                        appState.resizeSourceToPreset(appState.sourcePreset)
                    }
                }
            }

            Toggle("Sharpen (CAS)", isOn: $appState.casEnabled)
                .onChange(of: appState.casEnabled) { appState.updateUpscale() }
                .help("Contrast-adaptive sharpening — restores crispness on stretched low-res video. Works even at 1:1.")
            if appState.casEnabled {
                HStack(spacing: 8) {
                    Text("Sharpness").frame(width: 74, alignment: .leading).foregroundStyle(.secondary)
                    Slider(value: $appState.sharpness, in: 0...1)
                        .onChange(of: appState.sharpness) { appState.updateUpscale() }
                    Text(String(format: "%.1f", appState.sharpness))
                        .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                        .frame(width: 26, alignment: .trailing)
                }
            }
        }
    }

    // MARK: - Live stats (capturing)

    // 값 텍스트는 고정 폭 — 자릿수 변화(99→100)가 창 오토레이아웃 연쇄를 일으켜 메인 스레드를
    // 블록하던 것 방지 (렌더는 이제 전용 스레드지만 메인 부하는 여전히 줄이는 게 이득).
    private var liveSection: some View {
        section("Live") {
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                GridRow {
                    Label("FPS", systemImage: "speedometer").foregroundStyle(.secondary).gridColumnAlignment(.leading)
                    Text(String(format: "%.0f → %.0f", appState.inputFPS, appState.outputFPS))
                        .fontWeight(.semibold).monospacedDigit().frame(width: 110, alignment: .leading)
                }
                GridRow {
                    Label("Latency", systemImage: "timer").foregroundStyle(.secondary)
                    Text(String(format: "%.0f ms", appState.latencyMs))
                        .fontWeight(.semibold).monospacedDigit().frame(width: 110, alignment: .leading)
                }
                if let scale = appState.upscaleStatus {
                    GridRow {
                        Label("Scale", systemImage: "arrow.up.left.and.arrow.down.right").foregroundStyle(.secondary)
                        Text(scale).font(.caption).lineLimit(1).frame(width: 240, alignment: .leading)
                    }
                }
            }
            .font(.callout)
        }
    }

    // MARK: - Shortcut

    private var shortcutSection: some View {
        section("Shortcut") {
            HStack {
                Text("Capture toggle").foregroundStyle(.secondary)
                Spacer()
                ShortcutRecorder(binding: $appState.hotCapture)
                    .frame(width: 96, height: 22)
                    .onChange(of: appState.hotCapture) { appState.updateHotKeys() }
            }
            .help("Focus any window and press this to start/stop capture. Click the field to record a new combo.")
        }
    }
}
