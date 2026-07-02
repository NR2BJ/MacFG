# RIFE 자체 보간 모델 R&D (스파이크)

## 환경 / 실행
pixi 환경은 `~/Documents/pixi`의 `rife` feature에 통합됨 (python 3.12 + torch + torchvision + coremltools + numpy).
스크립트는 여기(`MacFG/research/rife/`)에 둔다. 실행:
```
pixi run --manifest-path ~/Documents/pixi/pixi.toml -e rife python ~/Documents/MacFG/research/rife/bench_ane.py ane
```
(스크립트는 자기 폴더로 chdir하므로 flownet.pkl 상대경로는 어디서 실행해도 정상.)
flownet.pkl = RIFE 4.x 구버전(t=0.5 전용) 미러. 최신 arbitrary-t는 4.25 필요(아래 메모).

## 2026-07-02 스파이크 결과 (M4)
- ifnet.py: flownet.pkl에서 역설계한 IFNet_HDv3 (c=90, 3블록, PReLU). strict 로드 성공, t=0.5 보간 정확.
- **이 미러 가중치는 t=0.5 전용** (t=0.25/0.75 어긋남, flow가 t 무관) → 144Hz용 arbitrary-t는 최신 4.25 필요.
- 성능 (CoreML, warp=grid_sample 제외한 conv-only, predict 고정비 뺀 실커널 추정):
  - 540p flow ~14ms / 720p ~27ms / 1080p ~23ms / 4K 불가
  - ANE가 GPU보다 3배 빠름 (720p 30 vs 100ms, predict 포함)
  - grid_sample(warp)은 ANE 미컴파일 → 반드시 Metal로 분리
- predict 하네스 고정비: 720p 3ms, 1080p 7ms (Swift 네이티브 IOSurface면 제거 가능)

## 결론 / 다음
풀 RIFE는 M4 실시간(16.7ms) 미달. 실시간 경로:
1. flow는 저해상도(≤540p)만 추정(~14ms) + 4K warp는 Metal 셰이더 (LS 방식)
2. Swift 네이티브 추론(predict 오버헤드 제거) + IOSurface 제로카피
3. palettization/int8 양자화로 ANE 추가 가속
4. arbitrary-t 최신 4.25 가중치 확보 (Google Drive 1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg) — 144Hz 임의 위상용

## 2026-07-02 v4.25 스파이크 2차 결과
- v425/ = 공식 패키지 (IFNet_HDv3.py + flownet.pkl 24MB, 5블록 c=192/128/96/64/32 + Head 인코더)
- **arbitrary-t 검증 통과**: 24px 평행이동, t=0.25/0.5/0.75 → 6/12/18px 정확 (sanity425.py)
- CoreML 변환: IFNet_HDv3_coreml.py (2곳 패치 — timestep repeat→broadcast, 동적 split→정적). timestep은 입력 텐서 유지.
- **ANE conv-only 실측 (bench425.py)**: 360p 8.5ms / 540p 20.7ms / 720p 36.3ms (predict 고정비 2-3ms 포함)
- 결론: 60fps x2 예산(쌍당 16.7ms)에서 **360~432p flow가 안전선**, 540p는 Swift 네이티브+양자화로 도전
- 다음: .mlpackage 저장 → Swift 네이티브 하네스 → Metal 4K warp → RIFEEngine 통합
