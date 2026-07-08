"""RIFE 4.25 도메인 파인튜닝 — 배포 경로(저해상 flow+풀 warp) end-to-end, MPS 네이티브.
목적: 반투명 인게임 채팅이 배경 flow에 끌리는 것을 학습으로 억제 가능한지.

핵심 설계:
- grid_sample MPS backward 미구현 → 수동 bilinear 워프(gather 기반, grid_sample과 max|Δ|1.9e-5).
- 손실은 배포가 실제 내는 픽셀(풀해상도 warp+blend 결과)에 걸어 배포 거동을 직접 타깃.
- L1 + Laplacian 피라미드(고주파 구조=텍스트에 민감).
사용: python train_finetune.py <triplets_dir> <short> <steps> <lr> [out.pkl] [--holdout t15,t03]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, types, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

DEV = torch.device("mps")
RIFE = "../v425/train_log"

def warp_manual(img, flow):
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing='ij')
    x = xx[None] + flow[:, 0]; y = yy[None] + flow[:, 1]
    x0 = torch.floor(x); y0 = torch.floor(y)
    wx = (x - x0).unsqueeze(1); wy = (y - y0).unsqueeze(1)
    x0i = x0.long().clamp(0, W - 1); x1i = (x0.long() + 1).clamp(0, W - 1)
    y0i = y0.long().clamp(0, H - 1); y1i = (y0.long() + 1).clamp(0, H - 1)
    flat = img.reshape(B, C, H * W)
    def g(yi, xi):
        idx = (yi * W + xi).reshape(B, 1, H * W).expand(B, C, H * W)
        return torch.gather(flat, 2, idx).reshape(B, C, H, W)
    Ia, Ib, Ic, Id = g(y0i, x0i), g(y0i, x1i), g(y1i, x0i), g(y1i, x1i)
    return Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy

# IFNet은 원본 warp(grid_sample) 대신 수동 워프를 쓰도록 주입
pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp_manual
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, RIFE)
from IFNet_HDv3_coreml import IFNet

SCALES = [16, 8, 4, 2, 1]

def load_net(pkl):
    net = IFNet()
    sd = torch.load(pkl, map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return net

def pad64(n): return ((n + 63) // 64) * 64

def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

def deploy_forward(net, a, b, short):
    # 배포 경로: 입력을 short로 축소 → IFNet flow/mask → 풀해상도 업스케일 → 풀 워프+블렌드
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    fl, mask, _ = net(torch.cat((alo, blo), 1), timestep=0.5, scale_list=SCALES)
    flow = F.interpolate(fl[4], size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy], device=a.device).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return warp_manual(a, flow[:, :2]) * m + warp_manual(b, flow[:, 2:4]) * (1 - m)

# ── Laplacian 피라미드 손실
def gauss_kernel(ch, dev):
    k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
    k = (k[:, None] * k[None, :]); k /= k.sum()
    return k.expand(ch, 1, 5, 5).contiguous()

def lap_loss(x, y, dev, levels=4):
    ker = gauss_kernel(3, dev)
    def down(t):
        t = F.pad(t, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(t, ker, stride=2, groups=3)
    def lap(t):
        pyr = []; cur = t
        for _ in range(levels):
            d = down(cur)
            up = F.interpolate(d, size=cur.shape[-2:], mode='bilinear', align_corners=False)
            pyr.append(cur - up); cur = d
        pyr.append(cur); return pyr
    lx, ly = lap(x), lap(y)
    return sum(F.l1_loss(a, b) * (2 ** i) for i, (a, b) in enumerate(zip(lx, ly)))

def ssim(x, y):
    def g(ks=11, sig=1.5):
        c = torch.arange(ks) - ks // 2; w = torch.exp(-(c ** 2) / (2 * sig ** 2)); w /= w.sum()
        return (w[:, None] * w[None, :]).view(1, 1, ks, ks).to(x.device)
    W = g()
    lx = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    ly = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(lx, W, padding=5); my = F.conv2d(ly, W, padding=5)
    vx = F.conv2d(lx * lx, W, padding=5) - mx * mx; vy = F.conv2d(ly * ly, W, padding=5) - my * my
    cxy = F.conv2d(lx * ly, W, padding=5) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))).mean().item()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("tdir"); ap.add_argument("short", type=int)
    ap.add_argument("steps", type=int); ap.add_argument("lr", type=float)
    ap.add_argument("out", nargs="?", default="ft.pkl")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--track", default="")   # 학습엔 포함하되 별도 추적(과적합/학습가능성 probe)
    a = ap.parse_args()
    hold = set(a.holdout.split(",")) if a.holdout else set()
    track = set(a.track.split(",")) if a.track else set()

    subs = sorted(d for d in os.listdir(a.tdir) if os.path.isdir(os.path.join(a.tdir, d)))
    data = {}
    for n in subs:
        d = os.path.join(a.tdir, n)
        data[n] = tuple(load_img(f"{d}/{k}.png").to(DEV) for k in ("a", "gt", "b"))
    train = [n for n in subs if n not in hold]
    print(f"train {len(train)} / holdout {sorted(hold)} @ short={a.short}")

    net = load_net(f"{RIFE}/flownet.pkl").to(DEV)

    # 기준선 SSIM (프리즈)
    net.eval()
    def eval_ssim(names):
        with torch.no_grad():
            return {n: ssim(deploy_forward(net, data[n][0], data[n][2], a.short), data[n][1]) for n in names}
    base = eval_ssim(subs)
    print("기준선 SSIM: " + " ".join(f"{n}={base[n]:.3f}" for n in (sorted(hold) or subs[:5])))

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    net.train()
    order = train.copy()
    t0 = time.time()
    for step in range(a.steps):
        n = order[step % len(order)]
        if step % len(order) == 0:
            import random; random.Random(step).shuffle(order)
        aimg, gt, bimg = data[n]
        pred = deploy_forward(net, aimg, bimg, a.short)
        loss = F.l1_loss(pred, gt) + 0.5 * lap_loss(pred, gt, DEV)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, a.steps // 12) == 0 or step == a.steps - 1:
            net.eval(); cur = eval_ssim(subs); net.train()
            tr = np.mean([cur[n] for n in train]); ho = np.mean([cur[n] for n in hold]) if hold else 0
            rep = sorted(hold | track)
            hd = " ".join(f"{n}={cur[n]:.3f}(Δ{cur[n]-base[n]:+.3f})" for n in rep)
            print(f"step {step+1:4d} loss={loss.item():.4f} train_ssim={tr:.4f} holdout={ho:.4f}  {hd}")
    torch.save(net.state_dict(), a.out)
    fin = eval_ssim(subs)
    print(f"\n완료 {time.time()-t0:.0f}s → {a.out}")
    print(f"전체 SSIM 기준선 {np.mean([base[n] for n in subs]):.4f} → 파인튜닝 {np.mean([fin[n] for n in subs]):.4f}")
    print("하드케이스 Δ: " + " ".join(f"{n}:{base[n]:.3f}→{fin[n]:.3f}" for n in sorted(hold, key=lambda x: base.get(x,1))))
