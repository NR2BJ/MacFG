import SwiftUI
import Overlay

/// 설정 우선 화면 (Lossless Scaling 방식): 설정은 앱에서 미리, 시작은 포커스 창에 단축키.
/// 각 항목에 한 줄 설명 + (?) 아이콘(클릭 팝오버 상세)으로 기술 용어를 풀어준다. en/ko/ja는 L().
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
                appSection
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
                    Button(L("Stop", "정지", "停止")) { Task { await appState.stopCapture() } }
                        .buttonStyle(.borderedProminent).tint(.red).controlSize(.large)
                }
            } else {
                HStack(spacing: 12) {
                    Image(systemName: "viewfinder").font(.system(size: 26)).foregroundStyle(.tint)
                    VStack(alignment: .leading, spacing: 2) {
                        let key = appState.hotCapture.label.isEmpty ? L("the shortcut", "단축키", "ショートカット") : appState.hotCapture.label
                        Text(L("Focus a window, press \(key)",
                               "창을 포커스하고 \(key) 누르기",
                               "ウィンドウをフォーカスして \(key) を押す"))
                            .font(.headline)
                        Text(L("Set it up below · press again to stop",
                               "아래에서 설정 · 다시 누르면 정지",
                               "下で設定 · もう一度で停止"))
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
        section(L("Interpolation", "프레임 보간", "フレーム補間")) {
            Toggle(L("Frame interpolation", "프레임 보간", "フレーム補間"), isOn: $appState.isInterpolationEnabled)
                .onChange(of: appState.isInterpolationEnabled) { appState.updateInterpolationEnabled() }

            field(L("Engine", "엔진", "エンジン"),
                  hint: L("Not sure? Try each and pick what looks better.",
                          "잘 모르겠으면 하나씩 써보고 더 나은 걸 고르세요.",
                          "迷ったら試して好みの方を。"),
                  detail: L("Metal Flow — our GPU interpolator: any multiplier (×2–×5), keeps native sharpness, lightest.\n\nNeural — learned optical flow (RIFE): cleanest fast motion and object edges (great for anime/film), uses more GPU and adds a little latency.\n\nApple FI — Apple's ANE model: fixed 2× at 720p, gentle look. Needs the display at fps × 2 (60→120, 24→144).",
                            "Metal Flow — 자체 GPU 보간기: 임의 배율(×2–×5), 원본 선명도 유지, 가장 가벼움.\n\nNeural — 학습된 옵티컬 플로우(RIFE): 빠른 모션·물체 경계가 가장 깨끗(애니/영화에 강함), GPU를 더 쓰고 지연이 약간 늘어남.\n\nApple FI — 애플 ANE 모델: 720p·2배 고정, 부드러운 느낌. 디스플레이를 fps×2로(60→120, 24→144).",
                            "Metal Flow — 自作GPU補間: 任意倍率(×2–×5)、元の鮮明さを維持、最軽量。\n\nNeural — 学習オプティカルフロー(RIFE): 速い動きと物体の輪郭が最もきれい(アニメ/映画向き)、GPU使用量と遅延が少し増える。\n\nApple FI — AppleのANEモデル: 720p・2倍固定、柔らかい印象。ディスプレイをfps×2に(60→120, 24→144)。")) {
                Picker("", selection: $appState.selectedRenderMode) {
                    ForEach(RenderMode.userSelectable) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.selectedRenderMode) { appState.updateRenderMode() }
            }

            field(L("Multiplier", "배율", "倍率"),
                  hint: L("Output frames = source fps × N. Auto fills your refresh rate.",
                          "출력 = 소스 fps × N. Auto는 주사율만큼 채움.",
                          "出力 = ソースfps × N。Autoはリフレッシュレートまで。"),
                  detail: L("Caps at your display's refresh rate: 60fps ×3 = 180 needs a 180Hz+ display (a 120Hz display shows 120). Auto picks the most your display can show.",
                            "디스플레이 주사율이 상한: 60fps ×3 = 180은 180Hz+ 필요(120Hz는 120까지). Auto는 디스플레이가 낼 수 있는 최대로.",
                            "上限はディスプレイのリフレッシュレート: 60fps ×3 = 180には180Hz+が必要(120Hzなら120)。Autoは表示可能な最大に。")) {
                Picker("", selection: $appState.frameMultiplier) {
                    Text("Auto").tag(0); Text("×2").tag(2); Text("×3").tag(3); Text("×4").tag(4); Text("×5").tag(5)
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.frameMultiplier) { appState.persistSettings() }
            }

            if appState.selectedRenderMode == .metalFlow {
                Divider().padding(.vertical, 2)

                sliderField(L("Motion", "모션", "モーション"),
                            low: L("sharp", "예리", "シャープ"), high: L("smooth", "부드러움", "スムーズ"),
                            value: $appState.motionSmoothness,
                            hint: L("How the motion looks — taste, not quality.",
                                    "움직임 느낌 — 화질이 아니라 취향.",
                                    "動きの質感 — 画質でなく好み。"),
                            detail: L("Sharp keeps more motion detail but can shimmer. Smooth is gentler and softer (closer to Apple FI's feel). Slide it while watching.",
                                      "예리 = 모션 디테일↑(어른거릴 수 있음). 부드러움 = 완만하고 소프트(Apple FI 느낌). 보면서 조절하세요.",
                                      "シャープ = 動きのディテール↑(ちらつくことも)。スムーズ = 穏やかで柔らか(Apple FI寄り)。見ながら調整。")) {
                    appState.updateMotionSmoothness()
                }

                sliderField(L("Edges", "경계", "エッジ"),
                            low: L("crisp", "선명", "くっきり"), high: L("soft", "부드러움", "やわらか"),
                            value: $appState.boundarySoftness,
                            hint: L("Object boundaries — pick by content.",
                                    "물체 경계 — 콘텐츠에 맞춰.",
                                    "物体の境界 — コンテンツに合わせて。"),
                            detail: L("The ghosting-vs-judder trade at moving edges. Crisp = less ghosting with a slight step (good for games / fast action). Soft = smoother with slight ghosting (good for film / slow pans).",
                                      "움직이는 경계의 고스팅↔저더 맞바꿈. 선명 = 고스팅↓·미세 저더(게임/빠른 액션). 부드러움 = 매끄럽지만 약간 고스팅(영화/느린 팬).",
                                      "動く境界のゴースト↔ジャダーのトレードオフ。くっきり = ゴースト↓·わずかなカクつき(ゲーム/速い動き)。やわらか = 滑らかだが少しゴースト(映画/ゆっくりパン)。")) {
                    appState.updateBoundarySoftness()
                }

                Divider().padding(.vertical, 2)

                field(L("Occlusion warp", "오클루전 워프", "オクルージョンワープ"),
                      hint: L("Experimental — off is fine for most content.",
                              "실험 기능 — 대부분 꺼둬도 됩니다.",
                              "実験機能 — 通常はオフでOK。"),
                      detail: L("A directional warp at reveal/cover edges. Can help some fast motion, but may shimmer on repetitive patterns (grids, text). Off by default; toggle while watching to compare.",
                                "가림/드러남 경계의 방향별 워프. 빠른 모션에 도움될 수 있으나 반복 패턴(격자·텍스트)에서 어른거릴 수 있음. 기본 off; 보면서 켜보고 비교.",
                                "隠れ/現れる境界の方向別ワープ。速い動きに効くことがあるが、繰り返しパターン(格子·文字)でちらつくことも。既定オフ; 見ながら切り替えて比較。")) {
                    Toggle(L("Enable", "켜기", "有効"), isOn: $appState.occlusionDirectional)
                        .toggleStyle(.switch).labelsHidden()
                        .onChange(of: appState.occlusionDirectional) { appState.updateOcclusionDirectional() }
                }

                field(L("Keep overlay while multitasking", "멀티태스킹 중 오버레이 유지", "マルチタスク中もオーバーレイ維持"),
                      hint: L("Cover mode: don't hide when you click another app.",
                              "Cover 모드: 다른 앱을 클릭해도 숨기지 않음.",
                              "Coverモード: 他アプリをクリックしても隠さない。"),
                      detail: L("By default the Cover overlay hides when you switch to another app so it doesn't block it (single-monitor safety). Turn on to keep watching the interpolated output while working in other windows — it stays on top of the source area.",
                                "기본값은 다른 앱으로 전환하면 Cover 오버레이가 그 앱을 안 가리려고 숨습니다(단일 모니터 안전장치). 켜면 다른 창에서 작업하면서도 보간 출력을 계속 볼 수 있습니다 — 소스 영역 위에 유지됩니다.",
                                "既定では他アプリに切り替えるとCoverオーバーレイが非表示になります(単一モニタの安全策)。オンにすると他ウィンドウで作業しながら補間出力を見続けられます。")) {
                    Toggle(L("Enable", "켜기", "有効"), isOn: $appState.coverKeepVisible)
                        .toggleStyle(.switch).labelsHidden()
                        .onChange(of: appState.coverKeepVisible) { appState.refreshOverlayVisibility() }
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
        section(L("Upscaling & sharpness", "업스케일 & 샤픈", "アップスケール & シャープ")) {
            field(L("Upscale", "업스케일", "アップスケール"),
                  hint: L("Blow a small source up to a sharp fullscreen viewer.",
                          "작은 소스를 선명한 전체화면으로 확대.",
                          "小さいソースを鮮明な全画面に拡大。"),
                  detail: L("Off — a 1:1 overlay on the source (interpolation only).\n\nANE — Apple's Neural Engine 2× upscaler (needs a ≤960px source, e.g. a small PiP).\nMetalFX — GPU spatial upscaler, any size.\nANE+FX — ANE then MetalFX, best for tiny sources up to 4K.",
                            "Off — 소스 위 1:1 오버레이(보간만).\n\nANE — 애플 뉴럴 엔진 2배 업스케일(소스 ≤960px 필요, 작은 PiP 등).\nMetalFX — GPU 공간 업스케일, 크기 무관.\nANE+FX — ANE 후 MetalFX, 아주 작은 소스를 4K까지.",
                            "Off — ソース上の1:1オーバーレイ(補間のみ)。\n\nANE — AppleのNeural Engine 2倍アップスケール(ソース≤960px、小さいPiPなど)。\nMetalFX — GPU空間アップスケール、サイズ問わず。\nANE+FX — ANEの後MetalFX、極小ソースを4Kまで。")) {
                Picker("", selection: $appState.upscaleMode) {
                    ForEach(UpscaleMode.allCases) { Text($0.displayName).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented)
                .onChange(of: appState.upscaleMode) {
                    appState.updateUpscale()
                    appState.autoSelectPlacementForUpscale()
                }
            }

            field(L("Source", "소스", "ソース"),
                  hint: L("Resize the source to a clean native resolution first.",
                          "소스를 깔끔한 네이티브 해상도로 먼저 리사이즈.",
                          "ソースをまず綺麗なネイティブ解像度に。"),
                  detail: L("On capture, resizes the source window so its short side hits this (landscape: height, portrait: width). A native-res source gives a clean 1:1 grab — ideal for browser Picture-in-Picture and IINA (both are chrome-free 16:9). Set it here before capturing.",
                            "캡처 시 소스 창의 짧은 변을 이 값으로(가로: 높이, 세로: 너비). 네이티브 해상도 소스는 1:1로 깨끗하게 잡힘 — 브라우저 PiP·IINA(둘 다 크롬 없는 16:9)에 이상적. 캡처 전에 설정.",
                            "キャプチャ時にソースの短辺をこの値に(横: 高さ、縦: 幅)。ネイティブ解像度なら1:1で綺麗に取得 — ブラウザPiP·IINA(共にクローム無し16:9)に最適。キャプチャ前に設定。")) {
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
                    Toggle(L("Sharpen (CAS)", "샤픈 (CAS)", "シャープ (CAS)"), isOn: $appState.casEnabled)
                        .onChange(of: appState.casEnabled) { appState.updateUpscale() }
                    HelpButton(title: L("Sharpen (CAS)", "샤픈 (CAS)", "シャープ (CAS)"),
                               text: L("Contrast-Adaptive Sharpening — the 'looks crisper' feel. Restores detail on stretched or soft video and works even at 1:1. Strong on soft areas, gentle on hard edges (no halos).",
                                       "대비 적응 샤프닝 — '선명해 보이는' 느낌. 늘어나거나 뭉개진 영상 디테일을 살리고 1:1에서도 유효. 부드러운 곳은 강하게, 하드 엣지는 약하게(헤일로 없음).",
                                       "コントラスト適応シャープ — 「くっきり感」。引き伸ばした/眠い映像のディテールを復元、1:1でも有効。柔らかい所は強く、硬いエッジは弱く(ハロー無し)。"))
                    Spacer()
                }
                if appState.casEnabled {
                    HStack(spacing: 8) {
                        Text(L("Strength", "강도", "強度")).font(.caption2).foregroundStyle(.secondary)
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
        section(L("Live", "실시간", "リアルタイム")) {
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                GridRow {
                    Label(L("Latency", "지연", "遅延"), systemImage: "timer").foregroundStyle(.secondary).gridColumnAlignment(.leading)
                    Text(String(format: "%.0f ms", appState.latencyMs))
                        .fontWeight(.semibold).monospacedDigit().frame(width: 110, alignment: .leading)
                }
            }
            .font(.callout)
        }
    }

    // MARK: - Shortcut

    private var shortcutSection: some View {
        section(L("Shortcut", "단축키", "ショートカット")) {
            shortcutRow(L("Capture toggle", "캡처 토글", "キャプチャ切替"),
                        L("Focus any window and press this to start or stop capture. Click the field to record a new combo (must include a modifier); the ✕ clears it.",
                          "아무 창이나 포커스하고 이 키를 눌러 캡처 시작/정지. 필드를 클릭해 새 조합 녹화(수정키 필수). ✕로 지웁니다.",
                          "任意のウィンドウをフォーカスしてこのキーでキャプチャ開始/停止。フィールドをクリックして記録(修飾キー必須)。✕で消去。"),
                        $appState.hotCapture)
            shortcutRow(L("Interpolation toggle", "보간 토글", "補間切替"),
                        L("Toggle frame interpolation on/off globally. Keep it ON for video; press to turn OFF for text/interactive apps. Leave blank (✕) if unused.",
                          "프레임 보간 on/off 전역 토글. 동영상은 켠 채로, 텍스트/인터랙티브는 끄기. 안 쓰면 ✕로 비워둡니다.",
                          "フレーム補間のオン/オフを全体切替。動画はオン、テキスト/操作系はオフ。使わなければ✕で空に。"),
                        $appState.hotInterp)
            shortcutRow(L("Info overlay", "정보 오버레이", "情報オーバーレイ"),
                        L("Show source / interpolated frame rate and upscale info in the top-left of the viewer.",
                          "뷰어 좌상단에 소스/보간 프레임레이트와 업스케일 정보를 표시합니다.",
                          "ビューアー左上にソース/補間フレームレートとアップスケール情報を表示。"),
                        $appState.hotInfo)
        }
    }

    /// 단축키 행 — 라벨 + 도움말 + 녹화 필드 + ✕(지우기, 미등록으로).
    @ViewBuilder
    private func shortcutRow(_ label: String, _ help: String, _ binding: Binding<HotKeyBinding>) -> some View {
        HStack(spacing: 6) {
            Text(label).foregroundStyle(.secondary)
            HelpButton(title: label, text: help)
            Spacer()
            ShortcutRecorder(binding: binding)
                .frame(width: 96, height: 22)
                .onChange(of: binding.wrappedValue) { appState.updateHotKeys() }
            Button {
                binding.wrappedValue = HotKeyBinding(keyCode: 0, modifiers: 0, label: "")
                appState.updateHotKeys()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(binding.wrappedValue.keyCode == 0 ? .quaternary : .tertiary)
            }
            .buttonStyle(.plain)
            .disabled(binding.wrappedValue.keyCode == 0)
            .help(L("Clear (no shortcut)", "지우기 (단축키 없음)", "消去(ショートカットなし)"))
        }
    }

    private var appSection: some View {
        section(L("App", "앱", "アプリ")) {
            field(L("Language", "언어", "言語"),
                  hint: L("Applies immediately — no restart.",
                          "즉시 적용 — 재시작 불필요.",
                          "即時適用 — 再起動不要。"),
                  detail: L("Overrides the auto-detected system language (default: System).",
                            "시스템 언어 자동 감지를 덮어씁니다 (기본: 시스템).",
                            "システム言語の自動検出を上書きします(既定: システム)。")) {
                Picker("", selection: Binding(
                    get: { appState.uiLanguage },
                    set: { appState.setLanguage($0) }   // current를 재구성 전에 갱신 → 즉시 반영
                )) {
                    Text(L("System", "시스템", "システム")).tag("system")
                    Text("한국어").tag("ko")
                    Text("English").tag("en")
                    Text("日本語").tag("ja")
                }
                .labelsHidden().pickerStyle(.menu).fixedSize()
            }

            field(L("Developer logging", "개발자 로그", "開発者ログ"),
                  hint: L("Off by default — records only when on.",
                          "기본 꺼짐 — 켤 때만 기록.",
                          "既定オフ — オン時のみ記録。"),
                  detail: L("When on, writes /tmp/MacFG_diag.log for troubleshooting. Turning it off deletes the file and stops recording.",
                            "켜면 문제 진단용 /tmp/MacFG_diag.log를 기록합니다. 끄면 파일을 삭제하고 기록을 멈춥니다.",
                            "オンで /tmp/MacFG_diag.log を記録します。オフでファイルを削除し記録を停止します。")) {
                Toggle("", isOn: $appState.devLoggingEnabled).labelsHidden()
                    .onChange(of: appState.devLoggingEnabled) { appState.updateDevLogging() }
            }

            Divider().padding(.top, 2)
            HStack {
                Button {
                    appState.openSettingsWindow()
                } label: {
                    Label(L("Open as window", "창으로 열기", "ウィンドウで開く"), systemImage: "macwindow")
                }
                .controlSize(.small)
                Spacer()
                Button(role: .destructive) {
                    NSApplication.shared.terminate(nil)
                } label: {
                    Label(L("Quit", "종료", "終了"), systemImage: "power")
                }
                .controlSize(.small)
            }
            Text(L("MacFG lives in the menu bar.", "MacFG는 메뉴바에 상주합니다.", "MacFGはメニューバーに常駐します。"))
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }
}

/// (?) 도움말 버튼 — 클릭 시 팝오버로 상세 설명.
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
        .popover(isPresented: $show, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(.headline)
                Text(text).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(14).frame(width: 300)
        }
    }
}
