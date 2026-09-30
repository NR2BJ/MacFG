"""RIFE v4.25 CoreML export v3 — "피라미드 체이닝" 그래프 (design.md O3-1: 블록 간 풀해상도 왕복 제거).

게이트 3종을 내장 실측:
  ① 재구성 델타: torch v1(원본 그래프) vs torch v3 (참고로 v2도) — flow px err / sigmoid(mask) err
  ② 변환 패리티: coreml v3 fp16 vs torch v3
  ③ 속도: ALL / CPU_AND_NE (배포 v2 실측 2026-09-30 — 288=7.3 / 360=11.7 / 432=16.3ms)
사용: pixi run ... python export_coreml_v3.py [--sizes 360] [--out <dir>]
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
from IFNet_HDv3_coreml_v3 import IFNet as IFNetV3

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

def real_pair(H, W):
    """실프레임 쌍(무작위 잡음은 flow가 무의미해 게이트가 약하다) — bench_frames의 첫 시퀀스에서 A·B를 모델 크기로."""
    import glob
    from PIL import Image
    seqs = sorted(glob.glob(os.path.expanduser("~/Documents/MacFG/bench_frames/2026*/frame_000.png")))
    if not seqs: return None
    d = os.path.dirname(seqs[len(seqs) // 2])
    def ld(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return torch.cat((ld(d + "/frame_000.png"), ld(d + "/frame_002.png")), 0).unsqueeze(0), d

def export_size(net_v1, net_v2, net_v3, short, outdir):
    H, W = pad64(short), pad64(short * 16 // 9)
    head_v1 = FlowHead(net_v1).eval()
    head_v2 = FlowHead(net_v2).eval()
    head_v3 = FlowHead(net_v3).eval()
    t = torch.full((1, 1, 1, 1), 0.5)
    rp = None
    try: rp = real_pair(H, W)
    except Exception as e: print("  (실프레임 로드 실패:", e, ")")
    x = rp[0] if rp else torch.rand(1, 6, H, W)
    if rp: print(f"  입력: 실프레임 쌍 {rp[1]}")

    # ── ① 재구성 델타 (torch fp32 v1 vs v2 / v3)
    with torch.no_grad():
        f1, m1 = head_v1(x, t)
        f2, m2 = head_v2(x, t)
        f3, m3 = head_v3(x, t)
    for name, fv, mv in [("v2", f2, m2), ("v3", f3, m3)]:
        ferr = (fv - f1).abs()
        merr = (torch.sigmoid(mv) - torch.sigmoid(m1)).abs()
        print(f"  ①재구성 델타 {name}: flow |err| mean={ferr.mean():.4f}px p99={ferr.flatten().quantile(0.99):.3f}px max={ferr.max():.2f}px "
              f"(flow 범위 ±{f1.abs().max():.1f}px), sigmoid(mask) mean={merr.mean():.5f}")

    with torch.no_grad():
        traced = torch.jit.trace(head_v3, (x, t))
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

    # ── ② 변환 패리티 (coreml fp16 vs torch v3 fp32)
    pred = ml.predict({"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)})
    cf = torch.from_numpy(pred["flow"].astype(np.float32))
    cm = torch.from_numpy(pred["mask"].astype(np.float32))
    ferr2 = (cf - f3).abs()
    merr2 = (torch.sigmoid(cm) - torch.sigmoid(m3)).abs()
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
    ap.add_argument("--weights", default=None,
                    help="파인튜닝한 v3 state_dict(.pt, 이미 순열된 배치 — mf/train_mf.py --arch v3). "
                         "주면 permute를 건너뛰고, ①은 원본 대비 '학습으로 달라진 양'이 된다(정보용)")
    a = ap.parse_args()
    outdir = os.path.abspath(a.out); os.makedirs(outdir, exist_ok=True)
    v1 = load(IFNetV1)
    v2 = load(IFNetV2)
    if a.weights:
        v3 = IFNetV3()
        sd = torch.load(a.weights, map_location='cpu')
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}   # V3AsMF 래퍼 접두사
        v3.load_state_dict(sd)
        v3 = v3.eval()
        print(f"파인튜닝 가중치: {a.weights}")
    else:
        v3 = load(IFNetV3).permute_input_channels()
    for s in [int(v) for v in a.sizes.split(",")]:
        H, W = pad64(s), pad64(s * 16 // 9)
        print(f"── rife{s} v3: 입력 {W}x{H}")
        p = export_size(v1, v2, v3, s, outdir)
        print(f"  저장: {p}")
