"""A1 판정 잣대 — 같은 평가 창에서 v3의 티어 한 칸(288→360→432)이 주는 배포 경로 PSNR 이득.
멀티프레임의 이득이 "티어 한 칸보다 큰가"를 같은 하네스·같은 창으로 비교하기 위해서다.
train_mf.py의 평가와 같은 창 순서(random.Random(1) 셔플)·같은 케이던스 두 가지를 쓴다.
사용: pixi run ... python tier_yardstick.py --data ~/Documents/MacFG/datasets/mf [--n 60] [--shorts 288,360,432]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, os, random, math
import numpy as np
import torch
import torch.nn.functional as F
import train_mf as T   # 수동 워프 주입·v3 클래스·load_jpg·synth 재사용

def pad64(n): return ((n + 63) // 64) * 64

def v3_deploy(net, A, G, B, t, mW, mH):
    with torch.no_grad():
        x = torch.cat((F.interpolate(A, size=(mH, mW), mode='bilinear', align_corners=False),
                       F.interpolate(B, size=(mH, mW), mode='bilinear', align_corners=False)), 1)
        fl, mk, _ = net(x, timestep=t, scale_list=[16, 8, 4, 2, 1])
        H, W = A.shape[-2:]
        f = F.interpolate(fl[4], size=(H, W), mode='bilinear', align_corners=False)
        f = f * torch.tensor([W / mW, H / mH, W / mW, H / mH], device=A.device).view(1, 4, 1, 1)
        m = torch.sigmoid(F.interpolate(mk, size=(H, W), mode='bilinear', align_corners=False))
        out = T.warp_manual(A, f[:, :2]) * m + T.warp_manual(B, f[:, 2:4]) * (1 - m)
        return -10 * math.log10(max(float(((out.clamp(0, 1) - G) ** 2).mean()), 1e-10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--shorts", default="288,360,432")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    ev = [w["id"] for w in man["eval"]]
    random.Random(1).shuffle(ev)
    ev = ev[:a.n]
    net = T.IFNetV3()
    sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    net = net.eval().permute_input_channels().to(T.DEV)
    res = {}
    for short in [int(s) for s in a.shorts.split(",")]:
        mH, mW = pad64(short), pad64(short * 16 // 9)
        rows = []
        for wid in ev:
            for ks, t in (((1, 3, 4, 5), 0.5), ((0, 3, 4, 6), 1 / 3)):
                _, A, G, B = [T.load_jpg(os.path.join(a.data, "eval", f"{wid}_{k}.jpg")).unsqueeze(0).to(T.DEV) for k in ks]
                rows.append(v3_deploy(net, A, G, B, t, mW, mH))
        res[short] = np.array(rows)
        print(f"v3 {short} ({mW}x{mH}): PSNR 중앙 {np.median(rows):.3f} 평균 {np.mean(rows):.3f}", flush=True)
    ks = sorted(res)
    for lo, hi in zip(ks, ks[1:]):
        d = res[hi] - res[lo]
        worst = np.argsort(res[360] if 360 in res else res[lo])[:max(1, len(d) // 4)]
        print(f"티어 {lo}→{hi}: Δ중앙 {np.median(d):+.3f}  Δ평균 {d.mean():+.3f}  "
              f"최악25%(360 기준) Δ중앙 {np.median(d[worst]):+.3f} Δ평균 {d[worst].mean():+.3f}")
