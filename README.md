# MacFG

Real-time frame **interpolation + upscaling** overlay for macOS — a personal take on Lossless Scaling for the Mac, mainly for watching video (streams, PiP, local players) on Apple Silicon.

Set your options once, focus any window, and press the shortcut. MacFG captures that window, interpolates it to a rock-solid 120 Hz, optionally upscales a small source to fullscreen, and shows it either 1:1 over the source or in a fullscreen viewer. Verified on a base M4 at 4K 60→120 with every presented frame exactly one vsync apart (glass-time σ = 0.00).

## Requirements

- Apple Silicon, macOS 26 (Tahoe) or later
- **Screen Recording** + **Accessibility** permission (prompted on first run — Accessibility is for window tracking/resize)

## Usage

1. Set your options in the panel (they persist).
2. **Focus** the window you want, and **press the capture shortcut** (default ⌃⌥⌘U, customizable). Press again to stop.
3. Placement is automatic: **Upscale off** → a 1:1 overlay on the source (interpolation only); **Upscale on** → a fullscreen viewer on the source's screen.

### Settings

- **Engine** — *Metal Flow* (default): our GPU pipeline (pyramid optical flow + full-res warp), any multiplier, keeps native sharpness. *Apple FI*: Apple's ANE model, fixed 2× at 720p; set the display to fps × 2 (60→120, 24→144).
- **Multiplier** — Auto, or ×2–×5 (capped at your display's refresh rate).
- **Motion** / **Edges** sliders (Metal Flow) — taste, not quality. *Motion* sharp↔smooth (flow detail vs gentleness). *Edges* crisp↔soft (the ghosting-vs-judder trade at object boundaries; crisp for games, soft for film).
- **Upscale** — Off / ANE (neural 2×, ≤960px source) / MetalFX / ANE+FX. **Sharpen (CAS)** restores crispness on stretched video.
- **Source** — resize the source to a native resolution on capture (360–1080p short side) for a clean 1:1 grab. Ideal for browser Picture-in-Picture and IINA (both are chrome-free 16:9).

## Install

Download the `.dmg` from [Releases](https://github.com/NR2BJ/MacFG/releases). It's self-signed; on first launch right-click → **Open** once (or `xattr -dr com.apple.quarantine MacFG.app`).

## Build

```sh
swift build                    # debug
scripts/make_app.sh 1.0.5      # release .app + .dmg (in dist/)
```

`make_app.sh` signs with a local **"MacFG Dev"** identity if present (so Screen Recording / Accessibility grants survive rebuilds), else ad-hoc.

## Architecture

- `Sources/CaptureKit` — ScreenCaptureKit capture (frame queue, fingerprint dedup, seamless resize)
- `Sources/Interpolation` — engines (`MetalFlowEngine`, `AppleFIEngine`, `PairEngine` protocol) + `InterpBench` (deterministic PSNR/timing regression tool)
- `Sources/Overlay` — overlay/viewer windows, `RenderSurface` (thread-agnostic encode), window tracking, same-display color passthrough, shader-side rounded corners
- `Sources/MacFGApp` — `RenderDriver` (**dedicated render thread + CAMetalDisplayLink** — true 120 Hz), timestamp output scheduler (cadence snap, vsync-grid phases, adaptive latency), SwiftUI panel
- `Sources/TestPattern` — self-verification source (`--fps N --jitter MS --complex`)

## Notes & limitations

- Interpolated 120 fps is inherently softer in motion than native 120 — a limit shared by all real-time interpolation. The Motion/Edges sliders tune *how* it degrades, not the ceiling; occlusion quality beyond hand-tuned flow needs a learned model (planned).
- **DRM** content (Netflix etc.) captures black by design (macOS protected-frame path) — out of scope.
- Apple FI is macOS 26-only and fixed at 720p / 2× (Apple-side session limits, measured).
- HDR capture/display is not implemented yet (SDR pipeline).
