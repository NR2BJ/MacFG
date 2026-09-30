# Models

CoreML exports of RIFE v4.25 (`IFNet_HDv3`) optical-flow network — full graph
including internal warps (`grid_sample` → MIL `resample`), fp16, fixed 16:9 sizes.
Outputs: final `flow` (1×4×H×W, px at model scale) + pre-sigmoid `mask` (1×1×H×W).
Full-resolution warp + sigmoid blend happen in the app's Metal kernel.

## 배포본 = v3 그래프 (2026-09-30)

가중치와 입출력 계약은 원본과 같고 **그래프만** 다르다. 재현: `research/rife/export_coreml_v3.py`.

| 그래프 | 내부 워프 | 비고 |
|---|---|---|
| v1 원본 | 블록마다 모델 풀해상도에서 워프 후 곧바로 1/scale로 축소 | `export_coreml.py` |
| v2 (2026-07-08) | 축소 후 워프 — 풀해상도 워프 16 → ~5.3회 | `export_coreml_v2.py` |
| v3 (2026-09-30) | 블록 출력을 자기 레벨 해상도에 두고 다음 레벨로 2배만 올림(풀해상도 왕복 제거), 이미지+특징 7ch 묶음 워프(resample 4 → 2회/블록) | `export_coreml_v3.py` — 첫 conv 입력 채널 순열로 state_dict 그대로 로드 |

predict (M4, coremltools, `CPU_AND_NE` = 앱 경로, 40회 평균):

| file | input (W×H) | v2 | v3 |
|---|---|---|---|
| rife180.mlpackage | 320×192 | 2.95 ms | 2.33 ms |
| rife216.mlpackage | 384×256 | 4.78 ms | 3.56 ms |
| rife240.mlpackage | 448×256 | 5.61 ms | 4.28 ms |
| rife288.mlpackage | 512×320 | 7.42 ms | 5.88 ms |
| rife360.mlpackage | 640×384 | 12.22 ms | 8.70 ms |
| rife432.mlpackage | 768×448 | 16.41 ms | 12.20 ms |
| rife540.mlpackage | 960×576 | 27.44 ms | 20.82 ms |

화질 관문 (v3 교체 근거):
- `research/rife/gate_v3.py` — 코퍼스+계단/초록 장면에서 움직임 큰 실삼중항 40개(원 해상도 flow p95 26~330px),
  앱과 같은 경로(모델 크기로 늘려 넣기 → flow 업샘플 → 가장자리 고정 워프 + sigmoid 블렌드)로 GT 대비 PSNR.
  CoreML v3 − v2 (둘 다 fp16, CPU_AND_NE):

  | 티어 | 중앙값 | 범위 | fp16 변환 손실 중앙값(최악) v2 / v3 |
  |---|---|---|---|
  | 180 | +0.001 dB | −0.021~+0.084 | −0.005(−0.025) / −0.006(−0.038) |
  | 360 | +0.003 dB | −0.011~+0.028 | −0.003(−0.046) / −0.003(−0.036) |
  | 540 | +0.001 dB | −0.050~+0.053 | −0.007(−0.112) / −0.009(−0.061) |

  원본(v1) 대비 flow 차이는 v2와 v3가 같은 크기다(360: mean 0.17 / p99 1.46 모델 px) — 즉 v3는
  v2가 이미 받아들인 근사 위에 더한 게 없다.
- `scripts/bench_sweep.py --engine rife --rife-ane` (28시퀀스, 360): v2 = v3 = 중앙값 +2.820 dB,
  빠른 모션 +4.000 dB, 최악값 16.02 dB. 시퀀스별 차이 −0.06~0.00 dB.

사다리 규칙은 그대로다 — 예산 규칙(`다음 티어 예상 비용 < 쌍 간격×0.75`)이 빨라진 만큼 티어를 스스로 다시 앉힌다.

## 로컬 전용 (미추적)

`rife720.mlpackage`, `v1_orig/`(v1 백업) — 720은 화질이 432 대비 사실상 같고 predict가 실시간 예산을
넘어 배포하지 않는다. 필요하면 해당 export 스크립트로 재생성.

Weights: [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) v4.25 (MIT).
