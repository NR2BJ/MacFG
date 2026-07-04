import SwiftUI
import Overlay

/// 설정 우선 화면 (Lossless Scaling 방식): 설정은 앱에서 미리, 시작은 포커스 창에 단축키.
/// 각 항목에 한 줄 설명 + (?) 아이콘(호버/클릭 상세 팝오버)으로 기술 용어를 풀어준다.
struct WindowPickerView: View {
    @Bindable var appState: AppState

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
        .frame(width: 440, height: 560)
        .background(.background)
    }

    // MARK: - Reusable shells

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.caption).fontWeight(.semibold)
                .foregroundStyle(.secondary).textCase(.uppercase).tracking(0.5)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
    }

    /// 라벨(+도움말 아이콘) · 컨트롤 · 한 줄 부연 설명을 묶은 항목
    @ViewBuilder
    private func field<Control: View>(_ label: String, hint: String, detail: String? = nil,
                                      @ViewBuilder _ control: () -> Control) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                if let detail { HelpButton(title: label, text: detail) }
                Spacer()
            }
            control()
            Text(hint).font(.caption2).foregroundStyle(.secondary)
        }
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
                    Image(systemName: "viewfinder").font(.system(size: 26)).foregroundStyle(.tint)
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

            field("Engine", hint: "Not sure? Try both and pick what looks better.",
                  detail: "Metal Flow — our GPU interpolator: any multiplier (×2–×5), keeps native sharpness, works on all Apple Silicon.\n\nApple FI — Apple's Neural Engine model: fixed 2× at 720p, sometimes handles complex motion more gently. Needs the display at fps × 2 (60→120, 24→144).") {
                Picker("", selection: $appState.selectedRenderMode) {
                    ForEach(RenderMode.userSelectable) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.selectedRenderMode) { appState.updateRenderMode() }
            }

            field("Multiplier", hint: "Output frames = source fps × N. Auto fills your refresh rate.",
                  detail: "Caps at your display's refresh rate: 60fps ×3 = 180 needs a 180Hz+ display (a 120Hz display shows 120). Auto picks the most your display can show.") {
                Picker("", selection: $appState.frameMultiplier) {
                    Text("Auto").tag(0); Text("×2").tag(2); Text("×3").tag(3); Text("×4").tag(4); Text("×5").tag(5)
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.frameMultiplier) { appState.persistSettings() }
            }

            if appState.selectedRenderMode == .metalFlow {
                Divider().padding(.vertical, 2)

                sliderField("Motion", low: "sharp", high: "smooth", value: $appState.motionSmoothness,
                            hint: "How the motion looks — taste, not quality.",
                            detail: "Sharp keeps more motion detail but can shimmer. Smooth is gentler and softer (closer to Apple FI's feel). Slide it while watching.") {
                    appState.updateMotionSmoothness()
                }

                sliderField("Edges", low: "crisp", high: "soft", value: $appState.boundarySoftness,
                            hint: "Object boundaries — pick by content.",
                            detail: "The ghosting-vs-judder trade at moving edges. Crisp = less ghosting with a slight step (good for games / fast action). Soft = smoother with slight ghosting (good for film / slow pans).") {
                    appState.updateBoundarySoftness()
                }

                Divider().padding(.vertical, 2)

                field("Occlusion warp", hint: "Experimental — off is fine for most content.",
                      detail: "A directional warp at reveal/cover edges. Can help some fast motion, but may shimmer on repetitive patterns (grids, text). Off by default; toggle while watching to compare.") {
                    Toggle("Enable", isOn: $appState.occlusionDirectional)
                        .toggleStyle(.switch).labelsHidden()
                        .onChange(of: appState.occlusionDirectional) { appState.updateOcclusionDirectional() }
                }
            }
        }
    }

    @ViewBuilder
    private func sliderField(_ label: String, low: String, high: String, value: Binding<Double>,
                            hint: String, detail: String, _ onChange: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(label)
                HelpButton(title: label, text: detail)
                Spacer()
            }
            HStack(spacing: 8) {
                Text(low).font(.caption2).foregroundStyle(.secondary)
                Slider(value: value, in: 0...1).onChange(of: value.wrappedValue) { onChange() }
                Text(high).font(.caption2).foregroundStyle(.secondary)
            }
            Text(hint).font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - Upscaling

    private var upscalingSection: some View {
        section("Upscaling & sharpness") {
            field("Upscale", hint: "Blow a small source up to a sharp fullscreen viewer.",
                  detail: "Off — a 1:1 overlay on the source (interpolation only).\n\nANE — Apple's Neural Engine 2× upscaler (needs a ≤960px source, e.g. a small PiP).\nMetalFX — GPU spatial upscaler, any size.\nANE+FX — ANE then MetalFX, best for tiny sources up to 4K.") {
                Picker("", selection: $appState.upscaleMode) {
                    ForEach(UpscaleMode.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.upscaleMode) {
                    appState.updateUpscale()
                    appState.autoSelectPlacementForUpscale()
                }
            }

            field("Source", hint: "Resize the source to a clean native resolution first.",
                  detail: "On capture, resizes the source window so its short side hits this (landscape: height, portrait: width). A native-res source gives a clean 1:1 grab — ideal for browser Picture-in-Picture and IINA (both are chrome-free 16:9). Set it here before capturing.") {
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

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Toggle("Sharpen (CAS)", isOn: $appState.casEnabled)
                        .onChange(of: appState.casEnabled) { appState.updateUpscale() }
                    HelpButton(title: "Sharpen (CAS)", text: "Contrast-Adaptive Sharpening — the 'looks crisper' feel. Restores detail on stretched or soft video and works even at 1:1. Strong on soft areas, gentle on hard edges (no halos).")
                    Spacer()
                }
                if appState.casEnabled {
                    HStack(spacing: 8) {
                        Text("Strength").font(.caption2).foregroundStyle(.secondary)
                        Slider(value: $appState.sharpness, in: 0...1)
                            .onChange(of: appState.sharpness) { appState.updateUpscale() }
                        Text(String(format: "%.1f", appState.sharpness))
                            .font(.callout).monospacedDigit().foregroundStyle(.secondary)
                            .frame(width: 26, alignment: .trailing)
                    }
                }
            }
        }
    }

    // MARK: - Live stats

    // 값 텍스트 고정 폭 — 자릿수 변화(99→100)가 창 오토레이아웃 연쇄로 메인 스레드 블록하던 것 방지.
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
                        Text(scale).font(.caption).lineLimit(1).frame(width: 250, alignment: .leading)
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
                HelpButton(title: "Capture toggle", text: "Focus any window and press this to start or stop capture. Click the field to record a new combo (must include a modifier).")
                Spacer()
                ShortcutRecorder(binding: $appState.hotCapture)
                    .frame(width: 96, height: 22)
                    .onChange(of: appState.hotCapture) { appState.updateHotKeys() }
            }
        }
    }
}

/// (?) 도움말 버튼 — 클릭 시 팝오버로 상세 설명. 발견 가능한 인라인 헬프.
private struct HelpButton: View {
    let title: String
    let text: String
    @State private var show = false

    var body: some View {
        Button { show.toggle() } label: {
            Image(systemName: "questionmark.circle")
                .font(.caption).foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help(text)   // 호버 시 기본 툴팁도 함께
        .popover(isPresented: $show, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.headline)
                Text(text).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(14).frame(width: 300)
        }
    }
}
