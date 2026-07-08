"""RIFE v4.25 CoreML export v2 — "다운스케일 후 워프" 그래프 (ANE resample 비용 ⅓ 목표).

게이트 3종을 내장 실측:
  ① 재구성 델타: torch v1(원본 그래프) vs torch v2 — flow px err / sigmoid(mask) err
  ② 변환 패리티: coreml v2 fp16 vs torch v2
  ③ 속도: ALL / CPU_AND_NE (원본 실측 대비 — 288=13.7 / 360=21.0 / 432=30.8ms)
사용: pixi run ... python export_coreml_v2.py [--sizes 360] [--out <dir>]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time, argparse, os
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

# ── ANE 친화 warp — 채널 슬라이스 제거 + 정규화 상수 융합 (export_coreml.py와 동일)
_WARP_CONSTS = {}
def warp(img, flow):
    _, _, H, W = flow.shape
    key = (H, W)
    if key not in _WARP_CONSTS:
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
        ng = torch.stack((2.0 * xx / max(W - 1, 1) - 1.0,
                          2.0 * yy / max(H - 1, 1) - 1.0), 0).unsqueeze(0)
        sc = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
                          dtype=torch.float32).view(1, 2, 1, 1)
        _WARP_CONSTS[key] = (ng, sc)
    ng, sc = _WARP_CONSTS[key]
    g = (flow * sc + ng).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, 'v425/train_log')
from IFNet_HDv3_coreml import IFNet as IFNetV1
from IFNet_HDv3_coreml_v2 import IFNet as IFNetV2

SCALES = [16, 8, 4, 2, 1]

class FlowHead(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x, t):
        flow_list, mask, _ = self.net(x, timestep=t, scale_list=SCALES)
        return flow_list[4], mask

def pad64(n): return ((n + 63) // 64) * 64

def load(cls):
    net = cls()
    sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return net.eval()

def export_size(net_v1, net_v2, short, outdir):
    H, W = pad64(short), pad64(short * 16 // 9)
    head_v1 = FlowHead(net_v1).eval()
    head_v2 = FlowHead(net_v2).eval()
    x = torch.rand(1, 6, H, W); t = torch.full((1, 1, 1, 1), 0.5)

    # ── ① 재구성 델타 (torch fp32 v1 vs v2)
    with torch.no_grad():
        f1, m1 = head_v1(x, t)
        f2, m2 = head_v2(x, t)
    ferr = (f2 - f1).abs()
    merr = (torch.sigmoid(m2) - torch.sigmoid(m1)).abs()
    print(f"  ①재구성 델타: flow |err| mean={ferr.mean():.4f}px max={ferr.max():.2f}px "
          f"(flow 범위 ±{f1.abs().max():.1f}px), sigmoid(mask) mean={merr.mean():.5f}")

    with torch.no_grad():
        traced = torch.jit.trace(head_v2, (x, t))
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                ct.TensorType(name="t", shape=t.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="flow", dtype=np.float16),
                 ct.TensorType(name="mask", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    path = os.path.join(outdir, f"rife{short}.mlpackage")
    ml.save(path)

    # ── ② 변환 패리티 (coreml fp16 vs torch v2 fp32)
    pred = ml.predict({"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)})
    cf = torch.from_numpy(pred["flow"].astype(np.float32))
    cm = torch.from_numpy(pred["mask"].astype(np.float32))
    ferr2 = (cf - f2).abs()
    merr2 = (torch.sigmoid(cm) - torch.sigmoid(m2)).abs()
    print(f"  ②변환 패리티: flow |err| mean={ferr2.mean():.4f}px max={ferr2.max():.2f}px, "
          f"sigmoid(mask) mean={merr2.mean():.5f}")

    # ── ③ 속도
    for label, cu in [("ALL", ct.ComputeUnit.ALL), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(path, compute_units=cu)
        inp = {"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
        for _ in range(8): m.predict(inp)
        N = 40
        t0 = time.perf_counter()
        for _ in range(N): m.predict(inp)
        ms = (time.perf_counter() - t0) / N * 1000
        print(f"  ③속도[{label}]: {ms:.2f} ms/추론  ({1000 / ms:.0f} fps)")
    return path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="360")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = os.path.abspath(a.out); os.makedirs(outdir, exist_ok=True)
    v1 = load(IFNetV1)
    v2 = load(IFNetV2)
    for s in [int(v) for v in a.sizes.split(",")]:
        H, W = pad64(s), pad64(s * 16 // 9)
        print(f"── rife{s} v2: 입력 {W}x{H}")
        p = export_size(v1, v2, s, outdir)
        print(f"  저장: {p}")
