"""A1 ① — 멀티프레임(과거 프레임) IFNet의 ANE 수출 관문. 학습 전에 "배포 가능한 비용인가"부터 잰다.

  ① 등가성: v3 가중치를 MF로 이식(zero-init extension)하면 flow[:, :4]·mask[:, :1]이 v3와 같아야 한다
     (= 파인튜닝 출발점이 정확히 배포 모델이다). torch fp32, 실프레임.
  ② 변환: MF를 CoreML fp16으로 export, torch MF 대비 패리티.
  ③ 비용: 배포 v3(Models/) vs MF, CPU_AND_NE predict — 번갈아 재서 드리프트 상쇄.
사용: pixi run ... python ane_gate.py --out <dir> [--sizes 288,360,432]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time, argparse, os, glob
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from PIL import Image

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
sys.path.insert(0, '../v425/train_log')
from IFNet_HDv3_coreml_v3 import IFNet as IFNetV3
from ifnet_mf import from_v3

ROOT = os.path.abspath("../../..")
SCALES = [16, 8, 4, 2, 1]
def pad64(n): return ((n + 63) // 64) * 64

class HeadMF(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x, t): return self.net(x, t, SCALES)

def real_frames(H, W):
    """움직임이 있는 실프레임 3장 (P, A, B) — 계단 장면 우선, 없으면 코퍼스."""
    cands = sorted(glob.glob(os.path.join(ROOT, "bench_frames", "2026*", "frame_000.png")))
    d = os.path.dirname(cands[len(cands) // 2])
    def ld(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1)
    fs = sorted(glob.glob(os.path.join(d, "frame_*.png")))
    return torch.cat([ld(fs[0]), ld(fs[2]), ld(fs[4])], 0).unsqueeze(0), d

def bench(m, inp, n=40):
    for _ in range(8): m.predict(inp)
    t0 = time.perf_counter()
    for _ in range(n): m.predict(inp)
    return (time.perf_counter() - t0) / n * 1000

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="288,360,432")
    a = ap.parse_args()
    outdir = os.path.abspath(a.out); os.makedirs(outdir, exist_ok=True)

    v3 = IFNetV3()
    sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    v3.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    v3 = v3.eval().permute_input_channels()
    mf = from_v3(v3, warp).eval()
    nparam = lambda n: sum(p.numel() for p in n.parameters())
    print(f"파라미터 v3 {nparam(v3) / 1e6:.3f}M → MF {nparam(mf) / 1e6:.3f}M")

    for short in [int(s) for s in a.sizes.split(",")]:
        H, W = pad64(short), pad64(short * 16 // 9)
        print(f"── {short} ({W}x{H})")
        x9, d = real_frames(H, W)
        t = torch.full((1, 1, 1, 1), 0.5)
        # ① 등가성
        with torch.no_grad():
            fl3, _, _ = v3(x9[:, 3:9], timestep=0.5, scale_list=SCALES)
            _, m3, _ = v3(x9[:, 3:9], timestep=0.5, scale_list=SCALES)
            flm, mkm = mf(x9, t, SCALES)
        f3 = fl3[4]
        print(f"  ①등가성: flow |MF−v3| max={float((flm[:, :4] - f3).abs().max()):.2e}px  "
              f"mask max={float((mkm[:, :1] - m3).abs().max()):.2e}  sigmoid(mP) max={float(torch.sigmoid(mkm[:, 1:2]).max()):.1e}  "
              f"(flow 범위 ±{float(f3.abs().max()):.1f}px, {os.path.basename(d)})")
        # ② 변환
        head = HeadMF(mf).eval()
        with torch.no_grad():
            traced = torch.jit.trace(head, (x9, t))
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="x", shape=x9.shape, dtype=np.float16),
                                ct.TensorType(name="t", shape=t.shape, dtype=np.float16)],
                        outputs=[ct.TensorType(name="flow", dtype=np.float16),
                                 ct.TensorType(name="mask", dtype=np.float16)],
                        compute_units=ct.ComputeUnit.ALL,
                        minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16)
        path = os.path.join(outdir, f"rifemf{short}.mlpackage")
        ml.save(path)
        pred = ml.predict({"x": x9.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)})
        cf = torch.from_numpy(pred["flow"].astype(np.float32))
        print(f"  ②변환 패리티: flow |coreml−torch| mean={float((cf - flm).abs().mean()):.4f}px max={float((cf - flm).abs().max()):.2f}px")
        # ③ 비용 — 번갈아 두 번씩
        m_v3 = ct.models.MLModel(os.path.join(ROOT, "Models", f"rife{short}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        m_mf = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        in3 = {"x": x9[:, 3:9].numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
        inm = {"x": x9.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}
        r3, rm = [], []
        for _ in range(2):
            r3.append(bench(m_v3, in3)); rm.append(bench(m_mf, inm))
        a3, am = sum(r3) / len(r3), sum(rm) / len(rm)
        print(f"  ③비용 CPU_AND_NE: v3 {a3:.2f}ms ({' / '.join(f'{v:.2f}' for v in r3)})  MF {am:.2f}ms "
              f"({' / '.join(f'{v:.2f}' for v in rm)})  → {100 * (am / a3 - 1):+.1f}%")
