"""삼중항별 각 엔진 예측 vs GT의 PSNR/SSIM을 한 곳에서 계산 (공정 비교).
엔진: blend(베이스라인), metalflow, applefi, rife_low(shippable), rife_full(천장).
모션 크기(mean|a-b|)로 고/저모션 분리 집계 — 저모션은 거의 정답이라 변별력 낮음.
사용: python metrics.py <triplets_dir>
"""
import os, sys
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

ENGINES = ["metalflow", "applefi", "rife_low288", "rife_low360", "rife_low432", "rife_full"]

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

def psnr(x, y):
    mse = torch.mean((x - y) ** 2).item()
    return 99.0 if mse < 1e-8 else 10 * np.log10(1.0 / mse)

def _gauss(ks=11, sig=1.5):
    c = torch.arange(ks) - ks // 2
    g = torch.exp(-(c ** 2) / (2 * sig ** 2)); g /= g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, ks, ks)

_W = _gauss()

def ssim(x, y):  # luminance SSIM, 11x11 gaussian
    lx = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    ly = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(lx, _W); my = F.conv2d(ly, _W)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    vx = F.conv2d(lx * lx, _W) - mx2
    vy = F.conv2d(ly * ly, _W) - my2
    vxy = F.conv2d(lx * ly, _W) - mxy
    s = ((2 * mxy + C1) * (2 * vxy + C2)) / ((mx2 + my2 + C1) * (vx + vy + C2))
    return s.mean().item()

tdir = sys.argv[1]
subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
rows = {e: [] for e in ENGINES}       # (psnr, ssim, motion)
print(f"{'triplet':8} {'motion':>7} | " + " | ".join(f"{e:>18}" for e in ENGINES))
for name in subs:
    d = os.path.join(tdir, name)
    gt = load(f"{d}/gt.png")
    a, b = load(f"{d}/a.png"), load(f"{d}/b.png")
    motion = torch.mean(torch.abs(a - b)).item()
    cells = []
    for e in ENGINES:
        p = f"{d}/pred_{e}.png"
        if not os.path.exists(p): cells.append(f"{'--':>18}"); continue
        pr = load(p)
        if pr.shape != gt.shape:
            pr = F.interpolate(pr, size=gt.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
        pv, sv = psnr(pr, gt), ssim(pr, gt)
        rows[e].append((pv, sv, motion))
        cells.append(f"{pv:6.2f}dB {sv:.3f}".rjust(18))
    print(f"{name:8} {motion*100:6.1f}% | " + " | ".join(cells))

def agg(key):
    print(f"\n=== {key} 평균 ===")
    print(f"{'engine':>12} | {'PSNR':>8} | {'SSIM':>7} | n")
    stats = []
    for e in ENGINES:
        sel = [r for r in rows[e] if (key == 'ALL' or (key == 'HIGH' and r[2] >= 0.05) or (key == 'LOW' and r[2] < 0.05))]
        if not sel: continue
        mp = np.mean([r[0] for r in sel]); ms = np.mean([r[1] for r in sel])
        stats.append((e, mp, ms, len(sel)))
    stats.sort(key=lambda s: -s[1])
    for e, mp, ms, n in stats:
        print(f"{e:>12} | {mp:6.2f}dB | {ms:.4f} | {n}")

agg('ALL'); agg('HIGH'); agg('LOW')
print("\n(HIGH=모션≥5%: 변별 핵심 · LOW=모션<5%: 거의 정답 · blend는 정합 검증 베이스라인)")
