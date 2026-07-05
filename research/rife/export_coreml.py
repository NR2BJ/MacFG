"""RIFE v4.25 풀그래프(내부 grid_sample 포함) CoreML export + 패리티/속도 실측.

conv-only 벤치와 달리 warp를 identity로 치환하지 않는다 — 진짜 IFNet 그래프를
내보내고(grid_sample→MIL resample, GPU 파티션), 출력은 최종 flow(1x4xHxW) +
pre-sigmoid mask(1x1xHxW). 풀해상도 워프/시그모이드는 앱의 Metal 커널 몫.

사이즈는 16:9 고정 3종(콘텐츠 단변 432/360/288 → pad64):
  rife432 = 768x448, rife360 = 640x384, rife288 = 512x320
사용: pixi run ... python export_coreml.py [--sizes 432,360,288] [--out ../../Models]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

# ── 추적 가능한 실제 warp (grid_sample; trace 시 고정 크기라 meshgrid는 상수화됨)
def warp(img, flow):
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

SCALES = [16, 8, 4, 2, 1]

class FlowHead(torch.nn.Module):
    """x(1x6xHxW 0-1 RGB쌍) + t(1x1x1x1) → 최종 flow(1x4xHxW), pre-sigmoid mask(1x1xHxW)"""
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x, t):
        flow_list, mask, _ = self.net(x, timestep=t, scale_list=SCALES)
        return flow_list[4], mask

def pad64(n): return ((n + 63) // 64) * 64

def load_net():
    net = IFNet()
    sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return net.eval()

def export_size(net, short, outdir):
    H, W = pad64(short), pad64(short * 16 // 9)
    wrap_mod = FlowHead(net).eval()
    x = torch.rand(1, 6, H, W); t = torch.full((1, 1, 1, 1), 0.5)
    with torch.no_grad():
        traced = torch.jit.trace(wrap_mod, (x, t))
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

    # ── 패리티: 동일 입력 PyTorch fp32 vs CoreML fp16
    with torch.no_grad():
        ref_flow, ref_mask = wrap_mod(x, t)
    pred = ml.predict({"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)})
    cf, cm = torch.from_numpy(pred["flow"].astype(np.float32)), torch.from_numpy(pred["mask"].astype(np.float32))
    ferr = (cf - ref_flow).abs()
    merr = (torch.sigmoid(cm) - torch.sigmoid(ref_mask)).abs()
    print(f"  패리티: flow |err| mean={ferr.mean():.4f}px max={ferr.max():.2f}px (범위 ±{ref_flow.abs().max():.1f}px), "
          f"sigmoid(mask) |err| mean={merr.mean():.5f}")

    # ── 속도: ALL vs CPU_AND_NE
    for label, cu in [("ALL", ct.ComputeUnit.ALL), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        m2 = ct.models.MLModel(path, compute_units=cu)
        inp = {"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
        for _ in range(8): m2.predict(inp)
        N = 40
        t0 = time.perf_counter()
        for _ in range(N): m2.predict(inp)
        ms = (time.perf_counter() - t0) / N * 1000
        print(f"  속도[{label}]: {ms:.2f} ms/추론  ({1000/ms:.0f} fps)")
    return path

if __name__ == '__main__':
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="432,360,288")
    ap.add_argument("--out", default="../../Models")
    a = ap.parse_args()
    outdir = os.path.abspath(a.out); os.makedirs(outdir, exist_ok=True)
    net = load_net()
    for s in [int(v) for v in a.sizes.split(",")]:
        H, W = pad64(s), pad64(s * 16 // 9)
        print(f"── rife{s}: 입력 {W}x{H} (콘텐츠 단변 {s})")
        p = export_size(net, s, outdir)
        print(f"  저장: {p}")
    print("\n예산: 60→120=16.7ms/쌍, 30→60=33ms/쌍 (+4K워프 ~1.5ms, 팩 ~0.5ms 별도)")
