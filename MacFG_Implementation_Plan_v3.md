# MacFG Implementation Plan v3

Date: 2026-04-24 JST
Author: Codex

## 목적

여러 frame generation 접근을 더 붙이기 전에, 현재 실패 원인을 재현 가능하게 좁히고 4K 60fps source를 120Hz display에 안정적으로 올리는 최소 구조를 다시 세운다.

핵심 판단은 다음과 같다.

- M4 기본형의 순수 GPU 성능은 4K 60->120을 전혀 못할 수준이 아니다. 이전 `InterpBench` 기준 `Blend`와 `FastMotion` 모두 8ms budget 안에 들어왔다.
- 지금 문제는 engine 성능보다 capture lifetime, 색/샘플링 일관성, frame cadence, static UI 보존 실패가 섞인 파이프라인 문제일 가능성이 높다.
- 따라서 다음 구현은 "더 좋은 optical flow"가 아니라 "같은 입력이면 완전히 같은 출력, 움직이면 정해진 시간축으로 다른 출력"을 증명하는 것부터 시작한다.

## 멈출 것

당분간 아래 작업은 보류한다.

- `FastMotionInterpolator` threshold만 계속 조정하기
- Vision optical flow / VT FRC / RIFE 같은 중량 engine을 메인 경로에 바로 붙이기
- `displaySyncEnabled`, `gpuReady`, `maximumDrawableCount`를 감으로 바꾸기
- 실제 앱에서만 눈으로 보고 성공/실패 판단하기

이 작업들은 baseline이 흔들리는 상태에서는 신호보다 잡음이 많다.

## 목표 상태

1. `SourceOnly`, `SourceCopy`, `Blend`, `Motion`, `DebugDiff`를 앱에서 즉시 전환할 수 있다.
2. `SourceOnly`와 `SourceCopy`의 색/밝기/텍스트 선명도가 육안으로 동일하다.
3. 정적 화면에서 모든 보간 engine의 출력은 현재 source frame과 byte-level 또는 near-byte-level로 동일하다.
4. 60fps source를 120Hz display에 보여줄 때 출력 패턴은 `I S I S ...` 또는 timeline 기반 등 명확히 설명 가능한 cadence를 유지한다.
5. 정적 UI 영역은 절대 warp하지 않고, 움직이는 video-like 영역만 blend/warp한다.

## Phase 0: 진단 모드부터 고정

우선 렌더링 결과를 사람이 추측하지 않게 만든다.

구현 파일:

- `Sources/MacFGApp/AppState.swift`
- `Sources/MacFGApp/WindowPickerView.swift`
- `Sources/Monitoring/FGDiagnostics.swift` 신규

작업:

- `RenderMode` enum 추가: `sourceOnly`, `sourceCopy`, `blend`, `fastMotion`, `diff`, `tint`.
- UI에 mode picker 추가. 처음에는 개발용이어도 된다.
- 120 tick마다 source/hit/dup 외에 `framePattern`, `captureDelta`, `presentDelta`, `drawableWait`, `engineGpuMs`, `engineCpuMs`, `staticMismatch`를 기록한다.
- `Diff` 모드는 `abs(sourceCopy - output)`을 증폭해서 보여준다. 정적 화면에서 diff가 보이면 engine이 아니라 pipeline 오류로 본다.

완료 조건:

- 정적 창에서 `sourceCopy`, `blend(A==B)`, `fastMotion(A==B)`가 흔들리지 않는다.
- 흔들리면 다음 phase로 가지 않는다.

## Phase 1: capture를 최신 프레임 하나가 아니라 timestamp queue로 바꾸기

현재 `latestFrame()` 방식은 capture cadence와 display cadence를 섞어버린다. Frame generation은 "두 source frame 사이의 시간"이 중요하므로 timestamped queue가 필요하다.

구현 파일:

- `Sources/CaptureKit/CapturedFrameQueue.swift` 신규
- `Sources/CaptureKit/SCKCapture.swift`
- `Sources/CaptureKit/IOSurfaceCapture.swift`
- `Sources/CaptureKit/FrameSource.swift`

작업:

- `FrameSource.latestFrame()`만 쓰는 구조에서 `drainFrames(since:)` 또는 `takeFrames()` 구조로 확장한다.
- SCK는 `SCFrameStatus.complete` frame만 source timeline에 넣고, idle은 render duplicate 판단에만 사용한다.
- `SCStreamConfiguration.queueDepth = 5`를 기본값으로 둔다. Apple 문서 기준 queue depth는 stream stall 방지에 도움을 줄 수 있지만 8을 넘기면 안 된다.
- `minimumFrameInterval`은 일단 `1/60`과 `1/120`을 런타임 옵션으로 둔다. 60fps source를 만들 때는 60 capture가 더 안정적인지 반드시 비교한다.
- IOSurface path는 fingerprint/timestamp가 신뢰 가능해질 때까지 기본 우선순위에서 내린다. 직접 surface를 쓰려면 GPU 또는 CPU fingerprint를 반드시 채운다.

완료 조건:

- 60fps 동영상 창에서 capture timestamp delta가 16.6ms 근처로 안정적으로 기록된다.
- SCK 60 capture와 120 capture 중 어느 쪽이 더 안정적인지 로그로 판단 가능하다.

## Phase 2: output timeline을 따로 만들기

소스가 들어온 tick에 source를 바로 그리고, 빈 tick에 interpolation을 끼우는 방식은 시간 역전과 cadence 흔들림이 쉽게 생긴다. 대신 output tick마다 "지금 보여줄 virtual source time"을 계산한다.

구현 파일:

- `Sources/FramePacing/OutputTimeline.swift` 신규
- `Sources/MacFGApp/AppState.swift`
- `Sources/FramePacing/DisplayLinkSync.swift`

작업:

- `OutputTimeline`은 display tick time, display refresh, measured source fps를 입력받아 `OutputDecision`을 반환한다.
- 기본 latency는 source frame 1개 또는 1.5개로 둔다. 60fps source라면 약 16.6-25ms 지연을 받아들이고 안정성을 얻는다.
- 두 source frame A/B가 있으면 `virtualTime`이 A와 B 사이에 있을 때 `t=(virtualTime-A.ts)/(B.ts-A.ts)`로 interpolation을 요청한다.
- B 이후 시간이면 source B를 표시하고, source pair가 부족하면 last frame을 repeat한다.
- display tick마다 render 여부를 명확히 한다. duplicate tick을 스킵할지 repeat present할지는 mode로 분리해 측정한다.

완료 조건:

- 로그의 최근 30 frame pattern이 사람 눈으로 설명 가능하다.
- 60->120에서 출력 FPS가 display tick과 거의 일치하거나, 스킵 이유가 명확히 기록된다.

## Phase 3: 색과 identity 경로를 먼저 통과시키기

색감 변화와 정적 UI 떨림이 남아 있으면 어떤 FG도 성공처럼 보이지 않는다.

구현 파일:

- `Sources/Overlay/OverlayWindow.swift`
- `Sources/Interpolation/IdentityCopyInterpolator.swift` 신규
- `Sources/Interpolation/BlendInterpolator.swift`
- `Sources/InterpBench/main.swift`

작업:

- `IdentityCopyInterpolator`: frameB를 output으로 정확히 복사하는 compute 또는 blit engine 추가.
- overlay fragment sampler를 상황별로 나눈다. 1:1 출력일 때는 point sampling/read-equivalent path를 사용하고, 스케일링이 필요할 때만 linear sampling을 쓴다.
- `CAMetalLayer.colorspace`와 texture pixelFormat 조합을 matrix로 테스트한다: `.bgra8Unorm`, `.bgra8Unorm_srgb`, colorspace nil, sRGB.
- `InterpBench`의 PSNR은 중앙 row 샘플이 아니라 full-frame 또는 tile sample로 확장한다.
- 정적 테스트는 `A==B`뿐 아니라 text/edge pattern을 추가한다.

완료 조건:

- `IdentityCopy`가 정적 UI에서 source와 구분되지 않는다.
- `Blend(A==B)`가 `IdentityCopy`와 구분되지 않는다.
- 이 조건이 깨지면 motion engine 구현을 진행하지 않는다.

## Phase 4: Motion engine은 screen-content aware로 재작성

현재 block motion은 video 전체에는 약간 먹힐 수 있지만, UI/text를 건드리는 순간 부들거림이 생긴다. 새 engine은 "어디를 움직일지"를 먼저 결정한다.

구현 파일:

- `Sources/Interpolation/ScreenAwareInterpolator.swift` 신규
- `Sources/Interpolation/TileClassifier.swift` 신규
- `Sources/Interpolation/FastMotionInterpolator.swift`는 실험 engine으로 유지

작업:

- 16x16 또는 32x32 tile 단위로 `static`, `uiEdge`, `videoMotion`, `uncertain`을 분류한다.
- `static`과 `uiEdge`는 무조건 frameB를 exact copy한다.
- `videoMotion`만 blend 또는 block-warp를 적용한다.
- `uncertain`은 blend로 fallback한다. warp fallback보다 blend fallback이 시각적으로 안전하다.
- confidence map을 1-2 tile feathering해서 경계가 튀지 않게 한다.
- 첫 구현은 motion vector보다 mask 품질을 우선한다. mask가 안정되면 global pan + local residual motion으로 확장한다.

완료 조건:

- 텍스트/브라우저/메뉴바 같은 high-frequency UI는 정지 상태에서 흔들리지 않는다.
- 동영상 영역은 `Blend`보다 덜 고스팅이거나, 최소한 같은 안정성으로 보인다.

## Phase 5: VT low-latency path는 별도 optional engine으로 검증

Apple의 `VTLowLatencyFrameInterpolationConfiguration`은 source frame과 previous frame을 받아 temporal interpolation을 수행할 수 있고, temporal+spatial 조합도 지원한다. 단, session start 때 ML model loading이 frame time보다 오래 걸릴 수 있으므로 메인 render loop에 직접 끼우지 않는다.

구현 파일:

- `Sources/Interpolation/VTLowLatencyInterpolator.swift` 신규
- `Sources/InterpBench/main.swift`

작업:

- app 시작 또는 engine 선택 시 session을 미리 warm up한다.
- `VTFrameProcessor.process(with:parameters:)` Metal command buffer API를 우선 검토한다.
- 4K direct, 1440p process + upscale, 1080p process + upscale를 따로 측정한다.
- 실패하면 engine unavailable로 표시하고 main path에는 영향이 없게 한다.

완료 조건:

- 같은 input pair에 대해 latency, output correctness, failure code가 `InterpBench`와 app log에 남는다.
- 4K direct가 느리면 1440p/1080p assisted mode로만 남긴다.

## Phase 6: 기본 제품 경로

사용자 기본값은 안정성 우선으로 둔다.

- 기본 capture: SCK, 60fps, queueDepth 5
- 기본 output: timeline latency 1 source frame
- 기본 engine: `Blend` 또는 `ScreenAwareInterpolator`
- M4 이상: 4K 60->120 지원 목표
- M1: MacBook Air 내장 해상도급 60->120 우선, 4K는 performance mode 표시

## Acceptance Checklist

- `swift build` 성공
- `swift run InterpBench --width 3840 --height 2160 --frames 30 --engines identity,blend,screenaware` 성공
- 정적 UI 60초 테스트에서 visible jitter 없음
- 60fps 영상 120Hz 출력에서 pattern이 안정적임
- 색감 변화가 source-only 대비 눈에 띄지 않음
- worklog에 실제 수정 의도, 파일, 검증 결과 기록

## 참고한 공식 문서

- ScreenCaptureKit `SCStreamConfiguration.queueDepth`: 기본 queue depth는 3이고, 더 큰 queue는 stream stall 방지에 도움을 줄 수 있으나 8을 넘기지 말아야 한다.
- VideoToolbox `VTFrameProcessor`: frame-by-frame processor이며 Metal command buffer 기반 processing API가 있다.
- VideoToolbox `VTLowLatencyFrameInterpolationConfiguration`: previous/source frame 기반 temporal interpolation을 제공하고, temporal+spatial 조합도 지원한다.
- MetalFX: temporal scaler는 color, depth, motion information이 있을 때 render pipeline에 통합하는 쪽의 API다. 임의 창 캡처 기반 MacFG의 기본 path로 보기는 어렵다.

