# MacFG — 프로젝트 로드맵 v2

## 프로젝트 개요

- **프로젝트명**: MacFG (가칭)
- **목표**: macOS에서 특정 창 또는 전체화면을 실시간 캡처하여 프레임 생성 및 업스케일링을 적용하는 오버레이 프로그램
- **참고**: Lossless Scaling (Windows 전용)
- **차별점**: macOS 네이티브, 오픈소스, 창 추적 오버레이, 다중 보간/업스케일 엔진 지원
- **배포**: 오픈소스 (App Store 미배포, private API 활용 가능)

## 개발 환경

| 항목 | 내용 |
|------|------|
| 최소 OS | macOS 26 Tahoe |
| 최소 하드웨어 | Apple Silicon (M1 이상) |
| 주 개발 기기 | M4 기본 |
| 주 언어 | Swift (최신 Concurrency 전면 활용) |
| 보조 언어 | Python (Core ML 변환 1회성 작업) |
| UI 프레임워크 | SwiftUI |
| GPU 파이프라인 | Metal |
| VSync | CADisplayLink + MTLSharedEvent |
| 빌드 도구 | Xcode |
| 아키텍처 | Swift Package Manager 모듈화 |

## 기술 스택

| 역할 | 기술 | 비고 |
|------|------|------|
| 화면 캡처 | IOSurface 직접 캡처 (기본) + ScreenCaptureKit (폴백) | 내부 자동 전환 |
| 프레임 생성 (경량) | Metal 네이티브 Optical Flow + Metal 워핑 셰이더 | 사용자 선택 |
| 프레임 생성 (고품질) | RIFE → Core ML | 사용자 선택 |
| 프레임 생성 (대동작) | FILM → Core ML | 사용자 선택 |
| 업스케일링 (초경량) | Lanczos Metal 셰이더 | 사용자 선택 |
| 업스케일링 (경량) | MetalFX Spatial | 사용자 선택 |
| 업스케일링 (중량) | MetalFX Temporal (optical flow → 모션 벡터 변환) | 사용자 선택 |
| 업스케일링 (고품질 실사) | Real-ESRGAN → Core ML | 사용자 선택 |
| 업스케일링 (고품질 2D) | Real-CUGAN → Core ML | 사용자 선택 |
| 후처리 | Metal 셰이더 (Bilateral, HDR EDR) | |
| 오버레이 출력 | NSWindow (borderless, sharingType=.none) + CAMetalLayer | |
| 창 추적 | Accessibility API (우선) + CGWindowListCopyWindowInfo (폴백) | 내부 자동 전환 |
| VSync | CADisplayLink + MTLSharedEvent GPU-CPU 동기화 | |
| 모니터링 | Metal Performance Counters + CADisplayLink 타임스탬프 | |
| UI | SwiftUI + 메뉴바 NSStatusItem | |
| 프로파일 저장 | UserDefaults / JSON | |

## 전체 아키텍처

```
[화면 캡처]
   IOSurface 직접 캡처 ─(실패)─→ ScreenCaptureKit 폴백
        │
        ▼
[프레임 버퍼 (트리플 버퍼링)]
   락프리 큐 기반 버퍼 관리
   프레임 타임스탬프 검증 (드롭 감지)
        │
        ▼
[보간 엔진 (사용자 선택)]
   ├── Metal 네이티브 Optical Flow + Metal 워핑
   ├── RIFE Core ML
   └── FILM Core ML
        │
        ▼
[후처리 파이프라인 (Metal)]
   ├── 업스케일링 (사용자 선택)
   │   ├── Lanczos
   │   ├── MetalFX Spatial
   │   ├── MetalFX Temporal
   │   ├── Real-ESRGAN Core ML
   │   └── Real-CUGAN Core ML
   ├── Bilateral 필터
   └── HDR EDR 패스스루
        │
        ▼
[프레임 페이싱]
   균등 간격 제출 + MTLSharedEvent GPU 완료 시그널
        │
        ▼
[오버레이 출력]
   NSWindow (borderless, topmost, sharingType=.none)
   CAMetalLayer → 대상 창 위에 정확히 겹침
        │
        ▼
[CADisplayLink → VSync 동기화]

[창 추적]
   Accessibility API ─(미응답)─→ CGWindowList 폴백
```

## SPM 모듈 구조

```
MacFG/
├── MacFGApp/                  # 메인 앱 타겟 (SwiftUI, 메뉴바)
├── Packages/
│   ├── CaptureKit/            # 화면 캡처 (IOSurface + SCK 폴백)
│   ├── Interpolation/         # 프레임 생성 엔진 프로토콜 + 구현체
│   │   ├── OpticalFlowEngine/ #   Metal 네이티브 Optical Flow
│   │   ├── RIFEEngine/        #   RIFE Core ML
│   │   └── FILMEngine/        #   FILM Core ML
│   ├── Upscaling/             # 업스케일링 엔진 프로토콜 + 구현체
│   │   ├── LanczosEngine/     #   Metal Lanczos
│   │   ├── MetalFXEngine/     #   MetalFX Spatial + Temporal
│   │   ├── RealESRGANEngine/  #   Real-ESRGAN Core ML
│   │   └── RealCUGANEngine/   #   Real-CUGAN Core ML
│   ├── PostProcess/           # 후처리 (Bilateral, HDR EDR)
│   ├── Overlay/               # 오버레이 출력 + 창 추적
│   ├── FramePacing/           # 프레임 페이싱 + VSync
│   └── Monitoring/            # 성능 모니터링
```

---

## Phase 1 — 캡처 + 오버레이 파이프라인

**목표**: 프레임 생성 없이 캡처 → 오버레이 출력 파이프라인 완성. 레이턴시와 성능 기준선 확보.

### 1-1. IOSurface 직접 캡처

- CGSGetWindowSurfaceID로 대상 창의 IOSurface ID 획득
- IOSurfaceLookup으로 IOSurface 객체 참조
- IOSurface → MTLTexture 제로카피 변환 (`makeTexture(descriptor:iosurface:plane:)`)
- CADisplayLink 콜백마다 IOSurface 내용이 자동 갱신되므로 별도 스트림 불필요
- 실패 조건 감지: IOSurface가 nil이거나 접근 불가 시 ScreenCaptureKit으로 자동 폴백

### 1-2. ScreenCaptureKit 폴백

- SCShareableContent로 창 목록 조회
- SCStreamConfiguration으로 해상도, 픽셀 포맷(BGRA), 프레임레이트 설정
- SCStream + SCStreamOutput 콜백에서 CMSampleBuffer 수신
- CMSampleBuffer → CVPixelBuffer → IOSurface 백킹 MTLTexture 제로카피 변환
- async/await 래핑하여 Swift Concurrency Actor에서 관리

### 1-3. 프레임 버퍼 (트리플 버퍼링)

- 캡처/처리/출력 3개 슬롯을 락프리 스왑체인으로 관리
- 각 슬롯은 MTLTexture + 타임스탬프 쌍으로 구성
- 프레임 타임스탬프 검증: 연속 프레임 간격이 예상 대비 1.5배 초과 시 드롭으로 판정
- 드롭 발생 시 플래그 세팅 → 이후 Phase에서 보간 스킵 또는 이전 프레임 재사용

### 1-4. CAMetalLayer 오버레이 출력

- NSWindow를 borderless, non-activating, topmost로 생성
- `window.sharingType = .none` 설정 (다른 캡처 프로그램에 오버레이가 잡히지 않도록)
- contentView에 CAMetalLayer 배치, Metal 텍스처를 drawable로 출력
- `MTLCommandBuffer.present(_:)` → `commit()`으로 프레임 제출

### 1-5. 창 추적 (이중 폴백)

- **1순위 — Accessibility API**: AXUIElement로 `kAXPositionAttribute`, `kAXSizeAttribute` 조회. AXObserver로 창 이동/크기 변경 노티피케이션 수신
- **2순위 — CGWindowList 폴백**: AXUIElement가 응답하지 않는 앱(일부 게임, Electron 앱) 감지 시 CGWindowListCopyWindowInfo로 주기적 폴링 전환
- 전환 로직: AXObserver 노티피케이션이 N초간 없고 CGWindowList에서 위치 변화가 감지되면 자동 전환
- CGWindowLevel로 오버레이가 항상 대상 창 바로 위에 위치

### 1-6. CADisplayLink + MTLSharedEvent VSync

- CADisplayLink으로 디스플레이 주사율 동기화 콜백 루프 구성
- MTLSharedEvent 기반 GPU-CPU 동기화:
  - GPU 렌더링 완료 시 MTLSharedEvent 시그널
  - CPU에서 MTLSharedEvent.notify()로 비동기 대기 (폴링 없음)
  - GPU 완료 확인 후 다음 프레임 캡처/처리 시작
- targetTimestamp 활용한 프레임 타이밍 관리

### 1-7. 기본 UI

- SwiftUI로 창 선택 화면: 캡처 가능한 창 목록을 썸네일과 함께 표시
- NSStatusItem으로 메뉴바 아이콘 상주
- 캡처 시작/중지 토글
- 캡처 방식 표시 (IOSurface / ScreenCaptureKit 중 어느 쪽이 활성인지)

### 1-8. 성능 측정

- 입력 FPS: 프레임 도착 간격으로 계산
- 캡처→출력 레이턴시: 프레임 타임스탬프와 drawable 제출 시점 차이
- 콘솔 로깅으로 기준선 데이터 수집

### 1-9. SPM 모듈 초기 구조

- CaptureKit, Overlay, FramePacing 모듈 분리
- 각 모듈에 프로토콜 정의 (FrameSource, OverlayRenderer, DisplaySync)
- 메인 앱에서 모듈 조립

### 완료 기준

대상 창 위에 오버레이가 정확히 겹쳐지고, 창 이동 시 따라다니며, 디스플레이 주사율에 맞춰 안정적으로 출력됨. IOSurface 캡처 실패 시 ScreenCaptureKit으로 자동 전환 동작 확인.

---

## Phase 2 — Optical Flow 프레임 생성

**목표**: Apple Vision Optical Flow 기반 경량 프레임 생성 구현. 30fps 입력 → 60fps 출력 검증. Phase 6에서 Metal 네이티브로 교체할 기반 구축.

### 2-1. Optical Flow 계산 (Apple Vision — 초기 구현)

- VNGenerateOpticalFlowRequest로 연속 두 프레임 간 옵티컬 플로우 계산
- computationAccuracy .medium 또는 .high (성능/품질 트레이드오프)
- 결과: VNPixelBufferObservation → 픽셀별 2D displacement map (float16 × 2채널)

### 2-2. Vision → Metal 전달

- VNPixelBufferObservation의 CVPixelBuffer가 IOSurface 백킹이면 제로카피 MTLTexture 변환
- 아닌 경우 MTLBuffer memcpy 후 MTLTexture 변환 (폴백)

### 2-3. Metal 워핑 셰이더

- 양방향 워핑: 프레임 A→B, B→A 양쪽에서 중간 시점(t)으로 워핑
- fragment 셰이더에서 displacement map을 t 비율로 스케일링하여 소스 텍스처 샘플링
- 양쪽 워핑 결과를 가중 블렌딩하여 최종 중간 프레임 생성
- 오클루전 처리: 앞/뒤 플로우 일관성 검사 → 오클루전 마스크 → 한쪽 프레임만 사용

### 2-4. 장면 전환 감지

- Metal compute 셰이더로 연속 프레임 간 히스토그램 차이 / MAD 계산 (GPU에서 직접)
- 임계값 초과 시 보간 스킵, 원본 프레임 출력
- UI 슬라이더로 임계값 조절

### 2-5. 프레임 페이싱

- CADisplayLink 콜백 내에서 원본/보간 프레임 균등 간격 교차 제출
- MTLSharedEvent로 GPU 보간 완료 확인 후 제출 (CPU 폴링 없음)
- 보간 지연 시 이전 프레임 유지 (프레임 반복)

### 2-6. 배수 모드

- 고정 배수: x2, x3, x4. x3 이상은 다중 중간 시점(t=0.33, 0.67 등) 워핑
- 목표 FPS: 입력 FPS 실시간 측정 → 목표 달성에 필요한 배수 자동 계산
  - 비정수 배수는 프레임별 교차 적용 (예: x2.5 → x2와 x3 교차)

### 2-7. 모션 벡터 스무딩

- 시간적 스무딩: 이전/현재 플로우 벡터 지수 이동 평균 블렌딩
- 공간적 스무딩: 큰 벡터 변화 영역에 가우시안 블러
- 빠른 패닝/회전 리플 아티팩트 억제

### 2-8. FrameInterpolator 프로토콜

- Interpolation 모듈에 공통 프로토콜 정의:
  ```
  protocol FrameInterpolator {
      func interpolate(frameA: MTLTexture, frameB: MTLTexture, t: Float) async -> MTLTexture
      var name: String { get }
      var isAvailable: Bool { get }
  }
  ```
- OpticalFlowInterpolator 구현 (Phase 3, 3-1에서 RIFE/FILM도 동일 프로토콜 구현)

### 완료 기준

30fps 영상 입력 시 60fps로 부드럽게 출력. 장면 전환에서 아티팩트 없음.

---

## Phase 3 — RIFE + FILM Core ML 통합

**목표**: RIFE와 FILM을 Core ML로 변환하여 추가 엔진으로 통합. 세 엔진 간 전환 가능.

### 3-1. RIFE 모델 변환

- Practical-RIFE 4.25 PyTorch 체크포인트 사용
- Python coremltools로 .mlpackage 변환
  - 입력: 두 프레임 (RGB float32) + timestep (float32)
  - 출력: 보간된 중간 프레임 (RGB float32)
- compute_units 벤치마크: `.cpuAndNeuralEngine` / `.cpuAndGPU` / `.all` 세 가지로 M4 추론 속도 비교
- 입력 해상도: EnumeratedShapes로 주요 해상도 등록 (720p, 1080p, 1440p, 4K)

### 3-2. FILM 모델 변환

- FILM (Google Research, Apache 2.0) TensorFlow 체크포인트 사용
- TF → SavedModel → coremltools 변환 경로
  - TF 기반이라 RIFE보다 변환이 까다로움, 연산자 호환성 검증 필요
- 입력: 두 프레임 (RGB float32)
- 출력: 보간된 중간 프레임 (RGB float32)
- FILM은 큰 움직임에서 강점이 있으므로 품질 우선 옵션으로 포지셔닝

### 3-3. Core ML 추론 파이프라인

- MLModel.prediction(from:)으로 추론
- 입력: CVPixelBuffer 2장 (IOSurface 백킹 GPU 메모리 공유)
- 출력: CVPixelBuffer → MTLTexture 제로카피 전달
- Swift Concurrency Actor에서 추론 큐 관리

### 3-4. RIFEInterpolator / FILMInterpolator 구현

- Phase 2에서 정의한 FrameInterpolator 프로토콜 구현
- 각 엔진 SPM 모듈로 분리 (RIFEEngine, FILMEngine)

### 3-5. 엔진 전환 UI

- SwiftUI에서 세 엔진 선택 (Optical Flow / RIFE / FILM)
- 전환 시 파이프라인 즉시 교체
- 각 엔진 상태 표시 (로드 완료, 추론 시간 등)

### 3-6. 성능 비교 측정

- 세 엔진 동일 입력 대비 출력 FPS, 프레임 생성 레이턴시, GPU 사용률 로깅
- 자동 벤치마크 모드: 30초간 교차 실행 후 결과 요약

### 완료 기준

UI에서 세 엔진 전환 즉시 적용, M4 기준 1080p 60fps 출력 안정적 (RIFE 기준).

---

## Phase 4 — 업스케일링 + 후처리

**목표**: 다섯 가지 업스케일링 엔진 및 후처리 파이프라인 구현.

### 4-1. Lanczos 업스케일링

- Metal compute 셰이더로 Lanczos-3 커널 구현
- 분리 가능(separable) 필터로 가로/세로 2패스 처리
- 정수 및 비정수 배율 지원

### 4-2. MetalFX Spatial 업스케일링

- MTLFXSpatialScalerDescriptor로 Spatial Scaler 생성
- 입력: 저해상도 Metal 텍스처, 출력: 고해상도 Metal 텍스처
- Apple Silicon 최적화 엣지 감지 업샘플링
- FSR 1.0과 동일 포지션 — 벤치마크 후 FSR이 열위면 제거, 아니면 양쪽 유지

### 4-3. FSR 1.0 업스케일링 (비교용)

- AMD FidelityFX Super Resolution 1.0 (MIT) HLSL → Metal 셰이더 포팅
- EASU (엣지 감지 업샘플링) + RCAS (선명도 보정) 2패스
- RCAS 강도 UI 슬라이더 조절
- MetalFX Spatial 대비 벤치마크 후 열위 시 제거 가능

### 4-4. MetalFX Temporal 업스케일링

- MTLFXTemporalScalerDescriptor로 Temporal Scaler 생성
- 입력: 저해상도 텍스처 + 모션 벡터 텍스처 + depth (선택)
- **모션 벡터 공급**: Phase 2/6의 Optical Flow 결과를 MetalFX가 요구하는 모션 벡터 포맷으로 변환
  - Optical Flow displacement map (픽셀 단위) → NDC 모션 벡터 변환 Metal 셰이더
- **depth 없는 환경 대응**: depth 텍스처에 상수값(1.0) 또는 optical flow magnitude 기반 의사 depth 제공
- 이전 프레임 + 현재 프레임 + 모션 벡터로 시간적 안정성 확보
- 캡처 기반에서 Temporal이 실제로 Spatial 대비 이점이 있는지 실험 필요

### 4-5. Real-ESRGAN Core ML

- Real-ESRGAN x4plus 모델 coremltools 변환
- 타일링: 고해상도 입력을 타일 분할 → 추론 → 재조합 (VRAM 제한 대응)
- 타일 경계 오버랩 페더링으로 이음새 제거
- Core ML Neural Engine / GPU 추론, 결과 MTLTexture 제로카피 전달

### 4-6. Real-CUGAN Core ML

- Real-CUGAN 모델 coremltools 변환
- Real-ESRGAN과 동일한 타일링/페더링 파이프라인 공유
- 2D/애니메이션 콘텐츠 특화 — 앱별 프로파일에서 콘텐츠 유형별 자동 선택 가능

### 4-7. Upscaler 프로토콜 및 모듈화

- 공통 프로토콜 정의:
  ```
  protocol Upscaler {
      func upscale(input: MTLTexture, scaleFactor: Float) async -> MTLTexture
      var name: String { get }
      var isAvailable: Bool { get }
      var requiresMotionVector: Bool { get }  // MetalFX Temporal용
  }
  ```
- 각 엔진 SPM 모듈 분리

### 4-8. Bilateral 필터

- Metal compute 셰이더로 bilateral filter 구현
- 공간 가우시안 + 밝기 차이 가중치 결합
- 커널 크기, 시그마 UI 조절

### 4-9. HDR EDR 패스스루

- CAMetalLayer.wantsExtendedDynamicRangeContent = true
- 픽셀 포맷 .rgba16Float로 HDR 색역 보존
- Metal 셰이더에서 MaxEDR 참조, 톤매핑 없이 패스스루
- SDR 콘텐츠 자동 감지 시 EDR 비활성화

### 4-10. 후처리 파이프라인 순서

- 고정: 프레임 생성 → 업스케일 → Bilateral → HDR 패스스루
- Metal command buffer 내 개별 pass로 구성
- 중간 텍스처 풀링 재사용

### 4-11. 업스케일링 프리셋

- 사용자 편의를 위한 프리셋 제공:
  - **경량**: Lanczos
  - **균형**: MetalFX Spatial (또는 FSR 1.0)
  - **고품질**: MetalFX Temporal
  - **AI 실사**: Real-ESRGAN
  - **AI 2D**: Real-CUGAN
- 고급 설정에서 개별 엔진 직접 선택 가능

### 완료 기준

다섯 가지 업스케일링 엔진 전환 가능, Bilateral 필터 강도 조절 가능, HDR 콘텐츠 색상 깨짐 없음.

---

## Phase 5 — 모니터링 + 편의성

**목표**: 사용자 편의 기능 및 모니터링 UI 완성.

### 5-1. FPS 오버레이

- 오버레이 위에 별도 CATextLayer 또는 SwiftUI 뷰로 입력/출력 FPS 표시
- 0.5초 간격 갱신, 위치 드래그 가능

### 5-2. GPU 사용률 표시

- Metal Performance Counters(MTLCounterSampleBuffer) 또는 IOKit AGX 통계로 수집
- FPS 오버레이와 함께 표시

### 5-3. 프레임 타임 그래프

- 최근 120프레임 링버퍼 저장
- SwiftUI Canvas 또는 Metal 직접 렌더링으로 실시간 그래프
- 16.6ms (60fps) / 8.3ms (120fps) 기준선 표시

### 5-4. A/B 비교 토글

- 단축키로 보간/업스케일 즉시 켜기/끄기 (패스스루 전환)
- 화면 분할 비교: 좌측 원본 / 우측 처리 결과 (분할선 드래그)

### 5-5. 전역 단축키

- CGEventTap 기반 전역 키 등록
- 기본: 켜기/끄기, 엔진 전환, A/B 토글, FPS 오버레이
- 사용자 커스텀 키 바인딩

### 5-6. 앱별 프로파일

- Bundle Identifier 기반 설정 자동 로드/저장
- 저장 항목: 보간 엔진, 배수, 업스케일 엔진, 필터 강도, 해상도
- ~/Library/Application Support/MacFG/profiles/ (JSON)
- 대상 창 전환 시 자동 프로파일 로드

### 5-7. 시스템 시작 시 자동 실행

- SMAppService.mainApp로 로그인 시 자동 실행 등록/해제
- 설정 UI 토글

### 5-8. 메뉴바 완성

- NSStatusItem + NSMenu로 전체 기능 접근
- 항목: 대상 창 선택, 보간/업스케일 엔진 전환, 프로파일 관리, 모니터링, 설정, 종료
- Dock 아이콘 숨김 (activationPolicy = .accessory)

### 5-9. 다중 모니터

- NSScreen.screens로 모니터 탐지
- 모니터별 독립 CADisplayLink (주사율 상이 대응)
- 오버레이 NSWindow가 대상 창 위치한 모니터에 정확히 매핑

### 5-10. 권한 처리

- 화면 녹화: SCShareableContent 접근 시 시스템 프롬프트 자동 (ScreenCaptureKit 폴백 시)
- 접근성: AXIsProcessTrusted() 확인 → 미승인 시 안내 UI → 시스템 환경설정 열기
- 권한 상태 변경 감시, UI 실시간 반영

### 5-11. Graceful Degradation

- 프레임 타임이 목표 대비 연속 N프레임 초과 시 자동 품질 하향
- 하향 순서: 배수 감소 → 업스케일링 경량 전환 → Bilateral 비활성화
- GPU 부하 안정 시 자동 복원
- UI에서 자동/수동 모드 선택

### 완료 기준

메뉴바만으로 모든 기능 제어 가능, 앱 전환 시 프로파일 자동 적용, 다중 모니터 독립 동작.

---

## Phase 6 — Metal 네이티브 Optical Flow + 최적화 + 안정화

**목표**: Optical Flow를 Metal 네이티브로 교체하여 Apple Vision 의존 제거. 전체 파이프라인 최적화 및 안정화.

### 6-1. Metal 네이티브 Optical Flow 구현

- Metal compute 셰이더로 Farneback optical flow 구현
  - 다해상도 피라미드: 원본 → 1/2 → 1/4 → 1/8 단계별 축소
  - 각 레벨에서 다항식 확장 기반 displacement 계산
  - 상위 레벨 결과를 하위 레벨로 전파하여 정밀도 향상
- 전체 파이프라인이 GPU 메모리 내에서 완결 (제로카피 문제 원천 해결)
- Apple Vision OpticalFlowInterpolator를 교체하여 동일 프로토콜로 투입

### 6-2. Metal 파이프라인 최적화

- MTLTexture 풀링: 미리 할당된 텍스처 풀 재사용
- MTLHeap으로 텍스처 메모리 단편화 방지
- MTLCommandBuffer 최소화, render/compute pass 간 리소스 배리어 최적화

### 6-3. 메모리 관리

- CVPixelBufferPool 사용 (ScreenCaptureKit 폴백 경로)
- Instruments Leaks / Allocations로 누수 점검

### 6-4. 레이턴시 측정 및 최소화

- 전체 파이프라인 단계별 시간 측정: 캡처 → 버퍼 → 보간 → 후처리 → 출력
- Metal GPU Profiler로 셰이더 병목 식별
- 목표: 캡처→출력 전체 레이턴시 1프레임 이내

### 6-5. 엣지 케이스 처리

- 창 최소화: 캡처 일시정지, 복원 시 재개
- 전체화면 전환 / Space 변경: 오버레이 재배치
- Mission Control / Stage Manager: 오버레이 숨김
- 대상 앱 종료: 캡처 자동 중지, UI 알림

### 6-6. 권한 오류 복구

- 화면 녹화 권한 거부: 안내 → 시스템 환경설정 열기
- 접근성 권한 거부: 창 추적을 CGWindowList 전용으로 전환, 수동 위치 지정 모드
- 권한 재요청 플로우

### 6-7. 크래시 리포트 및 로깅

- os_log / Logger 기반 구조화 로깅 (카테고리별 분류)
- 비정상 종료 시 마지막 상태 저장
- 설정에서 로그 내보내기

### 6-8. 전체 테스트

- 대상: YouTube (Safari/Chrome), VLC, IINA, Steam 게임, Xcode 시뮬레이터
- 해상도: 720p, 1080p, 1440p, 4K
- 시나리오: 장시간(1시간+), 창 크기 반복 변경, 모니터 탈착, 잠자기/깨우기
- 캡처 방식 전환 테스트: IOSurface ↔ ScreenCaptureKit 폴백 동작 확인

### 완료 기준

1시간 이상 연속 크래시 없음. M4 기준 1080p CPU 10% 이하, GPU 30% 이하. Metal 네이티브 Optical Flow가 Apple Vision 대비 동등 이상 품질 + 낮은 레이턴시.

---

## 후속 개선 백로그 (2026-07-05 기록)

이번 세션에서 전체화면 뷰어 마우스 조작(상대커서: 호버·클릭·스크롤·드래그) + 저지연 인터랙티브 모드를 완성했다. 사용 중 관찰된 다음 항목을 후속으로 남긴다.

### B-1. [완료] 보간 페이싱 wobble 감소 → N3로 승계

- **증상**: present 타이밍은 완벽(glass σ≈0, 정확한 120Hz)인데 **콘텐츠 진행이 불균일**(content 8.3±3.7ms). 초당 프레임 수는 맞지만 모션이 살짝 흔들려 보임 — MetalFlow·AppleFI 공통.
- **원인**: 보간 프레임의 시간축(t-value)이 vsync 그리드에 완전히 정렬되지 않아 콘텐츠 진행 간격이 출렁임. staleDrop 8~12/2s(보간 프레임 일부가 표시 시한을 놓쳐 드롭)도 기여.
- **방향**:
  1. 보간 t-value를 present 슬롯 시각에 정확히 배치해 콘텐츠 균등 진행(그리드 정렬 정밀화).
  2. latencyOffset / 케이던스 락 재검토로 staleDrop(늦은 보간 프레임) 축소.
  3. 신경망 엔진(RIFE/AppleFI) 자체 품질·페이싱 튜닝을 함께.
- **참고**: 이번 세션 변경(격자 지문·추적율·마우스 탭)과 **무관** 확인 — 추적율 15↔6Hz 동일. 기존 보간 페이싱 고유 특성이다.

### B-2. 관찰 중 — teardown 크래시

- 스택: OverlayWindow dealloc → RenderSurface deinit → MetalFXUpscaler over-release.
- 유력 수정: RelativePointer 전용 탭 스레드를 **동기 정지**(disable()이 threadStopped 세마포어 대기)로 refcon dangling 차단.
- 드물어 무인 재현이 어려움 → 실사용 재발 여부 관찰 필요.

### B-3. 정리 — RIFE 모델 미추적

- `Models/rife540.mlpackage`, `rife720.mlpackage` 미추적 상태.
- Neural 모드를 배포에 포함할지(커밋) vs MetalFlow만 배포(gitignore) 결정 필요.


### N3. Neural 전콘텐츠 최상위 (2026-07-06 확정 계획)

목표(사용자): Neural이 게임·애니 등 모든 콘텐츠에서 최고. 2026-07-06 밤 격리 결과, 게임(오버워치급 35~50px/frame 모션)에서의 한계는 **flow 모델 천장**(RIFE 4.25 @360p flow)으로 확정 — 페이싱·GPU예산·워프셰이더 전부 실측 배제.

1. **실프레임 삼중항 회귀 세트** (최우선 인프라): 실콘텐츠(치지직 게임/애니)에서 (A, GT, B) 삼중항 캡처 → 오프라인 PSNR/육안 A/B. 합성-실전 갭으로 인한 사용자 핑퐁 종료.
2. **flow 해상도 한계 실험**: 게임 삼중항에 432/540/720 flow 오프라인 적용. 해상도로 풀리면 → 3, 안 풀리면 → 4.
3. **ANE 효율 재수출**: ANE가 GPU 대비 2배 느린 원인(resample 비효율)을 export 단계에서 제거 → 360/432 flow를 ANE ~14ms에 (GPU 무경합 고품질 flow).
4. **게임 도메인 모델** (장기): 게임 영상 파인튜닝/증류 or 타 아키텍처. research/rife 파이프라인 재활용.

확보 자산: 적응 사다리(288/ANE↔360/GPU), 앵커 멀티-t(arbitrary-t), 워프 일관성 폴백+모션 관용, 밝기정렬 컷 판정, [RIFE-CONF] 실계측.

### B-4. [조사완료·한계인정] 정적 UI(인게임 채팅) 흐림 — flow 화질 한계로 확정 (2026-07-06)

사용자 증상: 오버워치 좌측 인게임 채팅(저대비 반투명)이 흐리거나 흔들림. **실프레임 삼중항으로 완전 규명**:
- 증폭 diff(×6): 채팅 글자는 GT와 매칭(제자리·안 뭉개짐). 오차는 전부 텍스트 주변의 **밝은 고속 효과(레이저/이펙트)** — 이 장면 17.6dB.
- 즉 "텍스트 흔들림" = 저대비 글자 주위의 고속 효과가 부정확 보간돼 생기는 시각적 불안정. 텍스트 워프가 아님.
- 시도·기각(둘 다 실프레임 A/B 무효): ① 풀해상도 |A-B| 정적판정(레이저 실이동으로 무효) ② 고주파 구조 정적판정(저대비라 미발동, 공격적 임계는 이동콘텐츠 오보호로 하락).
- 결론: 정적판정 계열로 개선 불가. **고속 게임 효과 flow 화질 한계**(N3 Neural 천장과 동일 계열). 사용자 판단으로 한계 인정·보류.
- 미채택 대안(원하면 재개): 사용자 지정 보호영역(HUD 사각형 항상 소스=보간 안 함, 그 안 이동콘텐츠는 60fps judder 트레이드).

**상시 인프라 확보**: 실프레임 캡처(앱 ⌃⌥⌘D → bench_frames/frame_NNN.png) + InterpBench --triplets(A/GT/B 오프라인 PSNR) + dumpDiffPNG(증폭 diff). 합성-실전 갭 종식 — 앞으로 화질 논의는 실프레임으로.
