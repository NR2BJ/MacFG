# MacFG

Real-time frame interpolation overlay for macOS — a personal take on Lossless Scaling for the Mac.

Pick a window, and MacFG captures it, interpolates frames, and outputs at 120Hz+ either on top of the original window or in a separate viewer. Verified on a base M4 at 4K 60→120 (glass-time σ<1ms, source frames byte-identical in color).

## Requirements

- macOS 26 (Tahoe) or later, Apple Silicon
- Screen Recording permission (prompted on first run), Accessibility permission (window tracking)

## Usage

1. Launch MacFG → pick a target window → **Capture**
2. Output: **Cover Source** (overlays the original window — clicks pass through) or **Separate Window** (free-floating viewer)
3. Engine:
   - **Metal Flow** (recommended) — LSFG-style pure-GPU pipeline. Low-resolution motion field + full-resolution warp, so output keeps native sharpness. Arbitrary-phase interpolation aligned to the display's vsync grid: fills frame-drop gaps and adapts to any refresh rate (60fps→144Hz works). Runs on any Apple Silicon.
   - **Apple FI** — VideoToolbox low-latency frame interpolation (ANE, fixed 720p, fixed 2×). Its neural flow may handle complex motion better. Set your refresh rate to an integer multiple of source fps × 2 (60fps→120Hz, 24fps→144Hz).
4. Diagnostics: `/tmp/MacFG_diag.log` — `[SCHED]` lines; `glass σ` is the smoothness ground truth, `skip[...]` explains any interpolation skips.

## Build

```sh
swift build                    # debug
scripts/make_app.sh 1.0.0      # release .app + .dmg (in dist/)
```

Builds are ad-hoc signed. On another Mac, right-click → Open once (or `xattr -dr com.apple.quarantine MacFG.app`).

## Architecture

- `Sources/CaptureKit` — ScreenCaptureKit capture (frame queue, scattered-sample fingerprint dedup)
- `Sources/FramePacing` — display link (bound to the output screen), frame slots
- `Sources/Interpolation` — engines (`MetalFlowEngine`, `AppleFIEngine`, `PairEngine` protocol)
- `Sources/Overlay` — overlay/viewer windows, window tracking, color policy (same-display passthrough for byte-exact color), shader-side rounded-corner masking
- `Sources/MacFGApp` — timestamp output scheduler (cadence PLL, vsync-grid-aligned interpolation phases), UI
- `Sources/TestPattern` — self-verification source (`--size W H --pos X Y --fps N`)
- `research/rife` — RIFE CoreML spike (shelved alternative)

## Known limitations

- Scene-cut detection is luma-histogram based — flash/fade cuts may slip one 8ms morph frame through
- DRM-protected content (Netflix, etc.) captures black (macOS restriction)
- Apple FI is macOS 26-only and fixed at 720p (Apple-side session limit, measured)
- Translucent title bars look flat in capture (SCK renders the window in isolation — inherent to window capture)
