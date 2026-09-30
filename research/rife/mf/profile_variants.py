"""A1 ① 비용 분해 — MF 변형별 ANE predict (무작위 가중치, 지연만). ane_gate.py의 +40%가 어디서 오는지.
변형: P를 받는 블록 수(p_blocks) × P 특징 사용(p_feat), 그리고 인코더(Head) 단독 비용.
사용: pixi run ... python profile_variants.py --out <dir> [--size 288]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time, argparse, os
import numpy as np
import torch
import coremltools as ct
from ane_gate import warp, IFNetV3, SCALES, pad64, bench   # warp 주입·v3 클래스 재사용
from ifnet_mf import IFNetMF, Head

class V3Head(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x, t):
        fl, m, _ = self.net(x, timestep=t, scale_list=SCALES)
        return fl[4], m

class MFHead(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x, t): return self.net(x, t, SCALES)

class EncOnly(torch.nn.Module):
    def __init__(self): super().__init__(); self.enc = Head()
    def forward(self, x, t): return self.enc(x) + t

def export(mod, nin, H, W, path):
    x = torch.rand(1, nin, H, W); t = torch.full((1, 1, 1, 1), 0.5)
    with torch.no_grad():
        tr = torch.jit.trace(mod.eval(), (x, t))
    ml = ct.convert(tr, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
                                ct.TensorType(name="t", shape=t.shape, dtype=np.float16)],
                    compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.macOS15,
                    compute_precision=ct.precision.FLOAT16)
    ml.save(path)
    return ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE), \
        {"x": x.numpy().astype(np.float16), "t": t.numpy().astype(np.float16)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=288)
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    H, W = pad64(a.size), pad64(a.size * 16 // 9)
    torch.manual_seed(0)
    variants = [("v3", V3Head(IFNetV3()), 6)]
    for pb, pf in [(5, True), (3, True), (2, True), (5, False), (3, False), (2, False)]:
        variants.append((f"MF p{pb}{'f' if pf else 'i'}", MFHead(IFNetMF(warp, p_blocks=pb, p_feat=pf)), 9))
    variants.append(("Head×1(3→4ch, 모델해상도)", EncOnly(), 3))
    models = []
    for name, mod, nin in variants:
        m, inp = export(mod, nin, H, W, os.path.join(out, name.split("(")[0].replace(" ", "_").replace("×", "x") + ".mlpackage"))
        models.append((name, m, inp))
        print(f"  export {name}", flush=True)
    res = {n: [] for n, _, _ in models}
    for _ in range(2):
        for name, m, inp in models:
            res[name].append(bench(m, inp))
    base = sum(res["v3"]) / 2
    print(f"── {a.size} ({W}x{H}) CPU_AND_NE")
    for name, _, _ in models:
        v = sum(res[name]) / 2
        print(f"  {name:28s} {v:6.2f} ms  ({' / '.join(f'{x:.2f}' for x in res[name])})  {100 * (v / base - 1):+6.1f}% vs v3")
