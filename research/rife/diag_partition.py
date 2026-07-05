"""rife432.mlpackage 파티션 진단 — 어떤 op가 어느 유닛(ANE/GPU/CPU)에 배치되는지 +
macOS26 타깃 재변환/GPU-only 속도. 풀그래프 +20ms의 정체 규명.
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, time, types
import numpy as np
import torch
import coremltools as ct

MODEL = "../../Models/rife432.mlpackage"

# ── 1) compute plan: op별 디바이스 배치
try:
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice
    import coremltools.models.utils as cu
    compiled = cu.compile_model(MODEL)
    plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.ALL)
    prog = plan.model_structure.program
    fn = prog.functions["main"]
    counts = {}
    slow_ops = []
    for op in fn.block.operations:
        if not op.outputs: continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        est = plan.get_estimated_cost_for_mlprogram_operation(op)
        if usage is None: continue
        dev = type(usage.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")
        counts[(op.operator_name, dev)] = counts.get((op.operator_name, dev), 0) + 1
        if est is not None and est.weight is not None and est.weight > 0.005:
            slow_ops.append((est.weight, op.operator_name, dev))
    print("=== op종류×디바이스 배치 (상위) ===")
    for (name, dev), n in sorted(counts.items(), key=lambda kv: -kv[1])[:18]:
        print(f"  {name:28} {dev:14} ×{n}")
    print("\n=== 비용 상위 op ===")
    for w, name, dev in sorted(slow_ops, reverse=True)[:15]:
        print(f"  {w*100:5.1f}%  {name:28} {dev}")
    # 디바이스 전환 시퀀스 요약
    seq = []
    for op in fn.block.operations:
        if not op.outputs: continue
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None: continue
        d = type(u.preferred_compute_device).__name__.replace("ML","").replace("ComputeDevice","")
        if not seq or seq[-1] != d: seq.append(d)
    print(f"\n=== 디바이스 전환 시퀀스: {len(seq)-1}회 전환 ===")
    print("  " + " → ".join(seq[:40]) + (" …" if len(seq) > 40 else ""))
except Exception as e:
    print(f"compute_plan 실패: {e}")

# ── 2) macOS26 타깃 재변환 (신형 resample ANE 지원 여부) + GPU-only
def warp(img, flow):
    import torch.nn.functional as F
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vgrid = grid + flow
    vx = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
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

H, W = 448, 768
x = torch.rand(1, 6, H, W); t = torch.full((1, 1, 1, 1), 0.5)
with torch.no_grad():
    traced = torch.jit.trace(FlowHead(net).eval(), (x, t))

def bench(ml, label):
    inp = {"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
    for _ in range(8): ml.predict(inp)
    N = 30; t0 = time.perf_counter()
    for _ in range(N): ml.predict(inp)
    print(f"  {label}: {(time.perf_counter()-t0)/N*1000:.2f} ms")

for tgt_name, tgt in [("macOS26", getattr(ct.target, "macOS26", None))]:
    if tgt is None:
        print(f"{tgt_name}: coremltools에 타깃 없음"); continue
    try:
        ml26 = ct.convert(traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                    ct.TensorType(name="t", shape=t.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="flow", dtype=np.float16), ct.TensorType(name="mask", dtype=np.float16)],
            compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=tgt,
            compute_precision=ct.precision.FLOAT16)
        bench(ml26, f"432 풀그래프 [{tgt_name}, ALL]")
    except Exception as e:
        print(f"{tgt_name} 변환 실패: {str(e)[:150]}")

mlgpu = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.CPU_AND_GPU)
bench(mlgpu, "432 풀그래프 [macOS15, CPU_AND_GPU]")
