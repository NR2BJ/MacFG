# MacFG

> **English** · [한국어](#macfg-한국어)

Real-time frame **interpolation + upscaling** overlay for macOS — a personal take on Lossless Scaling for the Mac, mainly for watching video (streams, PiP, local players) on Apple Silicon.

Set your options once, focus any window, and press the shortcut. MacFG captures that window, interpolates it to a rock-solid 120 Hz, optionally upscales a small source to fullscreen, and shows it either 1:1 over the source or in a fullscreen viewer. Verified on a base M4 at 4K 60→120 with every presented frame exactly one vsync apart (glass-time σ = 0.00).

The app UI is localized to **English / 한국어 / 日本語** (auto-detected from your system language, English default).

## Requirements

- Apple Silicon, macOS 26 (Tahoe) or later
- **Screen Recording** + **Accessibility** permission (prompted on first run — Accessibility is for window tracking/resize)

## Usage

1. Set your options in the panel (they persist).
2. **Focus** the window you want, and **press the capture shortcut** (default ⌃⌥⌘U, customizable). Press again to stop.
3. Placement is automatic: **Upscale off** → a 1:1 overlay on the source (interpolation only); **Upscale on** → a fullscreen viewer on the source's screen.

### Settings

- **Engine** — *Metal Flow* (default): our GPU pipeline (pyramid optical flow + full-res warp), any multiplier, keeps native sharpness, lightest. *Neural*: learned optical flow (RIFE v4.25 via CoreML) + full-resolution Metal warp — cleanest fast motion and object edges (great for anime/film), uses more GPU and adds a little latency. *Apple FI*: Apple's ANE model, fixed 2× at 720p; set the display to fps × 2 (60→120, 24→144).
- **Multiplier** — Auto, or ×2–×5 (capped at your display's refresh rate).
- **Motion** / **Edges** sliders (Metal Flow) — taste, not quality. *Motion* sharp↔smooth (flow detail vs gentleness). *Edges* crisp↔soft (the ghosting-vs-judder trade at object boundaries; crisp for games, soft for film).
- **Upscale** — Off / ANE (neural 2×, ≤960px source) / MetalFX / ANE+FX. **Sharpen (CAS)** restores crispness on stretched video.
- **Source** — resize the source to a native resolution on capture (360–1080p short side) for a clean 1:1 grab. Ideal for browser Picture-in-Picture and IINA (both are chrome-free 16:9).

## Install

Download the `.dmg` from [Releases](https://github.com/NR2BJ/MacFG/releases). It's self-signed; on first launch right-click → **Open** once (or `xattr -dr com.apple.quarantine MacFG.app`).

## Build

```sh
swift build                    # debug
scripts/make_app.sh 1.0.6      # release .app + .dmg (in dist/)
```

`make_app.sh` signs with a local **"MacFG Dev"** identity if present (so Screen Recording / Accessibility grants survive rebuilds), else ad-hoc.

## Architecture

- `Sources/CaptureKit` — ScreenCaptureKit capture (frame queue, fingerprint dedup, seamless resize)
- `Sources/Interpolation` — engines (`MetalFlowEngine`, `RIFEEngine` — CoreML flow + Metal warp via MTLSharedEvent async pipeline, `AppleFIEngine`, `PairEngine` protocol) + `InterpBench` (PSNR/timing regression + real-content triplet A/B)
- `Models/` — RIFE v4.25 CoreML flow networks (288/360/432p short side, MIT — [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)), compiled and bundled into the app
- `Sources/Overlay` — overlay/viewer windows, `RenderSurface` (thread-agnostic encode), window tracking, same-display color passthrough, shader-side rounded corners
- `Sources/MacFGApp` — `RenderDriver` (**dedicated render thread + CAMetalDisplayLink** — true 120 Hz), timestamp output scheduler (cadence snap, vsync-grid phases, adaptive latency), SwiftUI panel, in-code localization (`Localization.swift`)
- `Sources/TestPattern` — self-verification source (`--fps N --jitter MS --complex`)

## Notes & limitations

- Interpolated 120 fps is inherently softer in motion than native 120 — a limit shared by all real-time interpolation. The Motion/Edges sliders tune *how* it degrades, not the ceiling; occlusion quality beyond hand-tuned flow needs a learned model (planned).
- **DRM** content (Netflix etc.) captures black by design (macOS protected-frame path) — out of scope.
- Apple FI is macOS 26-only and fixed at 720p / 2× (Apple-side session limits, measured).
- HDR capture/display is not implemented yet (SDR pipeline).

---

# MacFG (한국어)

> [English](#macfg) · **한국어**

macOS용 실시간 프레임 **보간 + 업스케일링** 오버레이 — 애플 실리콘에서 주로 영상(스트리밍, PiP, 로컬 플레이어)을 볼 때 쓰는, Lossless Scaling의 맥 버전 같은 개인 프로젝트입니다.

옵션을 한 번 정해두고, 원하는 창을 포커스한 뒤 단축키만 누르면 됩니다. MacFG가 그 창을 캡처해 흔들림 없는 120 Hz로 보간하고, 작은 소스는 전체화면으로 업스케일하며, 소스 위에 1:1로 겹치거나 전체화면 뷰어로 보여줍니다. 기본형 M4에서 4K 60→120으로, 표시되는 모든 프레임이 정확히 vsync 하나 간격(glass-time σ = 0.00)임을 실측했습니다.

앱 UI는 **English / 한국어 / 日本語** 로 현지화되어 있습니다(시스템 언어에서 자동 감지, 기본값 영어).

## 요구 사항

- 애플 실리콘, macOS 26 (Tahoe) 이상
- **화면 기록** + **손쉬운 사용(접근성)** 권한 (첫 실행 시 요청 — 접근성은 창 추적/리사이즈용)

## 사용법

1. 패널에서 옵션을 설정합니다(설정은 저장됩니다).
2. 원하는 창을 **포커스**하고 **캡처 단축키**를 누릅니다(기본 ⌃⌥⌘U, 변경 가능). 다시 누르면 정지.
3. 배치는 자동입니다: **업스케일 끔** → 소스 위 1:1 오버레이(보간만); **업스케일 켬** → 소스가 있는 화면의 전체화면 뷰어.

### 설정

- **엔진** — *Metal Flow* (기본): 자체 GPU 파이프라인(피라미드 옵티컬 플로우 + 풀해상도 워프), 배율 자유, 원본 선명도 유지, 가장 가벼움. *Neural*: 학습된 옵티컬 플로우(RIFE v4.25, CoreML) + 풀해상도 Metal 워프 — 빠른 모션·물체 경계가 가장 깨끗(애니/영화에 강함), GPU를 더 쓰고 지연이 약간 늘어남. *Apple FI*: 애플 ANE 모델, 720p·2× 고정; 디스플레이를 fps × 2로 설정(60→120, 24→144).
- **배율** — Auto, 또는 ×2–×5 (디스플레이 주사율까지).
- **모션** / **경계** 슬라이더 (Metal Flow) — 화질이 아니라 취향. *모션* 예리↔부드러움(플로우 디테일 대 부드러움). *경계* 선명↔부드러움(물체 경계의 고스팅↔저더 트레이드오프; 게임엔 선명, 영화엔 부드럽게).
- **업스케일** — 끔 / ANE(신경망 2×, 소스 ≤960px) / MetalFX / ANE+FX. **샤픈(CAS)** 은 늘어난 영상의 선명도를 복원합니다.
- **소스** — 캡처 시 소스를 네이티브 해상도(짧은 변 360–1080p)로 리사이즈해 깔끔한 1:1 그랩을 얻습니다. 브라우저 PiP와 IINA(둘 다 크롬 없는 16:9)에 이상적.

## 설치

[Releases](https://github.com/NR2BJ/MacFG/releases)에서 `.dmg`를 받으세요. 자체 서명이라 첫 실행 시 우클릭 → **열기**를 한 번 하면 됩니다(또는 `xattr -dr com.apple.quarantine MacFG.app`).

## 빌드

```sh
swift build                    # 디버그
scripts/make_app.sh 1.0.6      # 릴리스 .app + .dmg (dist/에 생성)
```

`make_app.sh`는 로컬 **"MacFG Dev"** 인증서가 있으면 그것으로 서명하고(화면 기록/접근성 권한이 재빌드에도 유지됨), 없으면 ad-hoc 서명합니다.

## 아키텍처

- `Sources/CaptureKit` — ScreenCaptureKit 캡처(프레임 큐, 지문 기반 중복 제거, 끊김 없는 리사이즈)
- `Sources/Interpolation` — 엔진(`MetalFlowEngine`, `RIFEEngine` — CoreML flow + MTLSharedEvent 비동기 Metal 워프, `AppleFIEngine`, `PairEngine` 프로토콜) + `InterpBench`(PSNR/타이밍 회귀 + 실영상 삼중항 A/B)
- `Models/` — RIFE v4.25 CoreML flow 네트워크(단변 288/360/432p, MIT — [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)), 앱에 컴파일·번들됨
- `Sources/Overlay` — 오버레이/뷰어 창, `RenderSurface`(스레드 무관 인코드), 창 추적, 동일 디스플레이 색 패스스루, 셰이더 측 라운드 코너
- `Sources/MacFGApp` — `RenderDriver`(**전용 렌더 스레드 + CAMetalDisplayLink** — 진짜 120 Hz), 타임스탬프 출력 스케줄러(케이던스 스냅, vsync 그리드 위상, 적응형 지연), SwiftUI 패널, in-code 현지화(`Localization.swift`)
- `Sources/TestPattern` — 자체 검증 소스(`--fps N --jitter MS --complex`)

## 참고 & 한계

- 보간된 120 fps는 네이티브 120보다 움직임이 본질적으로 더 부드럽지 않게(soft) 보입니다 — 모든 실시간 보간이 공유하는 한계입니다. 모션/경계 슬라이더는 열화의 *방식*을 조절할 뿐 천장을 올리지는 않습니다; 손튜닝 플로우를 넘어서는 오클루전 품질은 학습 모델이 필요합니다(예정).
- **DRM** 콘텐츠(넷플릭스 등)는 설계상 검은 화면으로 캡처됩니다(macOS 보호 프레임 경로) — 범위 밖.
- Apple FI는 macOS 26 전용이며 720p / 2× 고정입니다(애플 측 세션 제한, 실측).
- HDR 캡처/표시는 아직 미구현입니다(SDR 파이프라인).
