"""두 갈래 정밀 측정:
A) GPU-only 풀그래프 360/288 스케일링 (기존 mlpackage 재사용)
B) ANE 친화 warp 재작성 — grid를 상수 텐서로 접고 per-warp 연산을
   mul+add+permute+grid_sample 4개로 축소 → ANE 30.2ms가 어디까지 내려가나
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

def bench_ml(ml, x, t, N=30):
    inp = {"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
    for _ in range(8): ml.predict(inp)
    t0 = time.perf_counter()
    for _ in range(N): ml.predict(inp)
    return (time.perf_counter() - t0) / N * 1000

# ── A) GPU-only 스케일링
print("=== A) GPU-only(CPU_AND_GPU) 풀그래프 ===")
for short, (W, H) in [(432, (768, 448)), (360, (640, 384)), (288, (512, 320))]:
    p = f"../../Models/rife{short}.mlpackage"
    if not os.path.exists(p): print(f"  rife{short}: 없음"); continue
    ml = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    x = torch.rand(1, 6, H, W); t = torch.full((1, 1, 1, 1), 0.5)
    print(f"  rife{short} ({W}x{H}): {bench_ml(ml, x, t):.2f} ms")

# ── B) ANE 친화 warp: 상수 정규화 그리드 + 최소 연산
#    vx = 2*(gx+fx)/(W-1)-1 = [2/(W-1)]*fx + const_gx_norm  → flow에 채널별 scale 곱 + 상수 add
class ANEWarp:
    """크기별 상수(정규화 base grid, flow scale)를 미리 만들어 클로저로 제공"""
    def __init__(self, H, W):
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
        gxn = 2.0 * xx / max(W - 1, 1) - 1.0   # HxW
        gyn = 2.0 * yy / max(H - 1, 1) - 1.0
        self.base = torch.stack((gxn, gyn), 0).unsqueeze(0)          # 1x2xHxW 상수
        self.scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)]).view(1, 2, 1, 1)  # 상수
    def __call__(self, img, flow):
        g = (self.base + flow * self.scale).permute(0, 2, 3, 1)      # mul+add+permute
        return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

print("\n=== B) ANE 친화 warp 그래프 ===")
for short, (W, H) in [(432, (768, 448)), (360, (640, 384))]:
    aw = ANEWarp(H, W)
    pkg = types.ModuleType('model'); pkg.__path__ = []
    wl = types.ModuleType('model.warplayer'); wl.warp = aw
    sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
    for mod in ['IFNet_HDv3_coreml']:
        if mod in sys.modules: del sys.modules[mod]
    sys.path.insert(0, 'v425/train_log')
    from IFNet_HDv3_coreml import IFNet
    net = IFNet()
    sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    net.eval()

    class FlowHead(torch.nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x, t):
            fl, m, _ = s.n(x, timestep=t, scale_list=[16, 8, 4, 2, 1]); return fl[4], m

    x = torch.rand(1, 6, H, W); t = torch.full((1, 1, 1, 1), 0.5)
    with torch.no_grad():
        traced = torch.jit.trace(FlowHead(net).eval(), (x, t))
    try:
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                    ct.TensorType(name="t", shape=t.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="flow", dtype=np.float16), ct.TensorType(name="mask", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16)
        ms_ane = bench_ml(ml, x, t)
        ml.save(f"../../Models/rife{short}_anewarp.mlpackage")
        mlg = ct.models.MLModel(f"../../Models/rife{short}_anewarp.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
        ms_gpu = bench_ml(mlg, x, t)
        print(f"  rife{short}-anewarp: ANE {ms_ane:.2f} ms | GPU {ms_gpu:.2f} ms")
    except Exception as e:
        print(f"  rife{short}-anewarp 실패: {str(e)[:150]}")
