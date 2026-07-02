# MacFG

macOS용 실시간 프레임 보간 오버레이 — Lossless Scaling의 맥 대응을 목표로 하는 개인 프로젝트.

창 하나를 골라 캡처하고, 프레임을 보간해 원본 위(또는 별도 창)에 120Hz+로 출력한다.
4K 60→120을 M4 기본형에서 실측 검증했다 (glass 간격 σ<1ms, 소스 프레임 색 바이트 일치).

## 요구사항

- macOS 26 (Tahoe) 이상, Apple Silicon
- 화면 기록 권한 (첫 실행 시 요청), 접근성 권한 (창 추적용)

## 사용

1. MacFG 실행 → 목록에서 대상 창 선택 → **Capture**
2. 출력: **Cover Source** (원본 창 위에 덮음 — 클릭은 원본으로 통과) 또는 **Separate Window** (자유 이동 뷰어 창)
3. 엔진:
   - **Metal Flow** (권장) — LSFG 방식 순수 GPU. 저해상도 모션 지도 + 풀해상도 워프라 출력이 원본 선명도. 임의 위상 보간으로 프레임 드랍 갭 채움, 어떤 주사율에도 대응. M1부터 동작.
   - **Apple FI** — VideoToolbox 저지연 보간 (ANE, 720p 고정·2배 고정). 복잡한 모션에서 NN 품질이 유리할 수 있음. 주사율을 원본 fps×2의 정수배로 설정 권장 (60fps→120Hz, 24fps→144Hz).
4. 진단 로그: `/tmp/MacFG_diag.log` ([SCHED] 라인 — glass σ가 부드러움 지표)

## 빌드

```sh
swift build                    # 디버그
scripts/make_app.sh 1.0.0      # release .app + .dmg (dist/)
```

배포 서명이 아닌 adhoc 사인이므로, 다른 맥에서 받으면 최초 1회 우클릭→열기 (또는 `xattr -dr com.apple.quarantine MacFG.app`).

## 구조

- `Sources/CaptureKit` — ScreenCaptureKit 캡처 (프레임 큐, 산포 fingerprint 중복 제거)
- `Sources/FramePacing` — DisplayLink, 프레임 슬롯
- `Sources/Interpolation` — 보간 엔진들 (`MetalFlowEngine`, `AppleFIEngine`, `PairEngine` 프로토콜)
- `Sources/Overlay` — 오버레이/뷰어 창, 창 추적, 색 정책 (같은 디스플레이 passthrough)
- `Sources/MacFGApp` — 타임스탬프 출력 스케줄러 (케이던스 PLL, 갭 적응 다중 t), UI
- `Sources/TestPattern` — 자체 검증용 60fps 테스트 소스 (`--size W H --pos X Y --fps N`)
- `research/rife` — RIFE CoreML 스파이크 (보류된 대안 경로)

## 알려진 한계

- 장면 전환 감지는 휘도 히스토그램 기반 — 플래시/페이드 컷은 놓칠 수 있음 (8ms 모핑 프레임 1장 노출)
- DRM 보호 콘텐츠(Netflix 등)는 캡처 자체가 검게 나옴 (macOS 제약)
- Apple FI는 macOS 26 전용, 모델 해상도 720p 고정 (Apple 측 제약 — 세션 레벨 하드코딩 실측)
