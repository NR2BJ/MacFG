# Models

CoreML exports of RIFE v4.25 (`IFNet_HDv3`) optical-flow network — full graph
including internal warps (`grid_sample` → MIL `resample`), fp16, fixed 16:9 sizes.
Outputs: final `flow` (1×4×H×W, px at model scale) + pre-sigmoid `mask` (1×1×H×W).
Full-resolution warp + sigmoid blend happen in the app's Metal kernel.

| file | input (W×H) | content short side | M4 ANE | M4 GPU |
|---|---|---|---|---|
| rife288.mlpackage | 512×320 | 288 | 13.9 ms | 8.4 ms |
| rife360.mlpackage | 640×384 | 360 | 20.8 ms | 12.1 ms |
| rife432.mlpackage | 768×448 | 432 | 30.2 ms | 16.7 ms |

Weights: [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) v4.25 (MIT).
Export/repro: `research/rife/export_coreml.py`.
