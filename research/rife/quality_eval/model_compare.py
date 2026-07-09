"""모델 패밀리 A/B — 같은 24삼중항·같은 배포경로(저해상 flow+풀 warp)로 여러 RIFE 가중치 비교.
아키텍처가 동일한 버전(예: 4.25/4.26)만 유효 — state_dict strict=False 로드 후 형상 검증.
사용: python model_compare.py <triplets_dir> <short> <name1>=<pkl1> <name2>=<pkl2> ...
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, types
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

RIFE = "../v425/train_log"

def warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype, device=flow.device), torch.arange(W, dtype=img.dtype, device=flow.device), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vg = grid + flow
    vx = 2.0 * vg[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vg[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, RIFE)
from IFNet_HDv3_coreml import IFNet

SCALES = [16, 8, 4, 2, 1]

def load_net(pkl):
    net = IFNet()
    sd = torch.load(pkl, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    inet = {k for k in net.state_dict()}
    bad = [k for k in inet if k in sd and net.state_dict()[k].shape != sd[k].shape]
    if bad:
        raise RuntimeError(f"형상 불일치 {len(bad)} (아키텍처 다름): {bad[:3]}")
    return net.eval()

def pad64(n): return ((n + 63) // 64) * 64

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

def _gauss(ks=11, sig=1.5):
    c = torch.arange(ks) - ks // 2
    g = torch.exp(-(c ** 2) / (2 * sig ** 2)); g /= g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, ks, ks)
_W = _gauss()

def ssim(x, y):
    lx = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    ly = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    W = _W.to(x.device)
    mx = F.conv2d(lx, W, padding=5); my = F.conv2d(ly, W, padding=5)
    vx = F.conv2d(lx * lx, W, padding=5) - mx * mx
    vy = F.conv2d(ly * ly, W, padding=5) - my * my
    cxy = F.conv2d(lx * ly, W, padding=5) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))).mean().item()

def rife_low(net, a, b, short):
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        fl, mask, _ = net(torch.cat((alo, blo), 1), timestep=0.5, scale_list=SCALES)
    flow = F.interpolate(fl[4], size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return warp(a, flow[:, :2]) * m + warp(b, flow[:, 2:4]) * (1 - m)

if __name__ == '__main__':
    tdir = sys.argv[1]; short = int(sys.argv[2])
    specs = [s.split("=", 1) for s in sys.argv[3:]]
    subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
    triplets = [(n, load(f"{tdir}/{n}/a.png"), load(f"{tdir}/{n}/b.png"), load(f"{tdir}/{n}/gt.png")) for n in subs]
    motions = {n: torch.mean(torch.abs(a - b)).item() for n, a, b, _ in triplets}
    med = float(np.median(list(motions.values())))
    print(f"삼중항 {len(subs)}, short={short}, 모션중앙값 {med:.4f}\n")
    print(f"{'model':<10} {'SSIM':>7} {'고모션':>7} {'최악5':>7}")
    per = {}
    for name, pkl in specs:
        net = load_net(pkl)
        rows = []
        for n, a, b, gt in triplets:
            rows.append((n, ssim(rife_low(net, a, b, short), gt), motions[n]))
        per[name] = {n: s for n, s, _ in rows}
        alls = [r[1] for r in rows]
        hi = [r[1] for r in rows if r[2] >= med]
        w5 = np.mean([r[1] for r in sorted(rows, key=lambda r: r[1])[:5]])
        print(f"{name:<10} {np.mean(alls):7.4f} {np.mean(hi):7.4f} {w5:7.4f}")
    if len(specs) == 2:
        (n1, _), (n2, _) = specs
        print(f"\n삼중항별 {n2}-{n1} SSIM 델타 (하드케이스순):")
        deltas = sorted(((n, per[n2][n] - per[n1][n]) for n in subs), key=lambda x: per[n1][x[0]])
        for n, d in deltas[:8]:
            print(f"  {n}: {n1}={per[n1][n]:.4f} {n2}={per[n2][n]:.4f}  Δ={d:+.4f}")
