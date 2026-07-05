"""RIFE v4.25로 삼중항 중간프레임 예측 — 두 가지 경로.
  full : 원해상도 입력 (품질 천장, 실시간 예산 초과)
  low  : 입력을 short-side 432로 축소해 flow만 신경망, 그 flow를 풀해상도로
         업스케일해 원본을 warp+blend (= shippable 배포 경로: 저해상도 flow+full warp)
사용: python rife_predict.py <triplets_dir> [short=432]
각 tNN/에 pred_rife_full.png, pred_rife_low.png 생성.
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, types
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

RIFE = "../v425/train_log"

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
sys.path.insert(0, RIFE)
from IFNet_HDv3 import IFNet

def load_net():
    net = IFNet()
    sd = torch.load(f"{RIFE}/flownet.pkl", map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    return net.eval()

def pad64(n): return ((n + 63) // 64) * 64

def load(p):
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

def save(t, p):
    arr = (t.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(p)

SCALES = [16, 8, 4, 2, 1]

def rife_full(net, a, b):
    H, W = a.shape[-2:]
    pH, pW = pad64(H), pad64(W)
    ap = F.pad(a, (0, pW - W, 0, pH - H), mode='replicate')
    bp = F.pad(b, (0, pW - W, 0, pH - H), mode='replicate')
    with torch.no_grad():
        _, _, merged = net(torch.cat((ap, bp), 1), timestep=0.5, scale_list=SCALES)
    return merged[4][:, :, :H, :W]

def rife_low(net, a, b, short=432):
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        flow_list, mask, _ = net(torch.cat((alo, blo), 1), timestep=0.5, scale_list=SCALES)
    flow = flow_list[4]                       # 1x4xlHxlW  (저해상도 flow)
    # 풀해상도로 업스케일 + 벡터 크기 스케일
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    wa = warp(a, flow[:, :2])
    wb = warp(b, flow[:, 2:4])
    return wa * m + wb * (1 - m)

if __name__ == '__main__':
    tdir = sys.argv[1]
    short = int(sys.argv[2]) if len(sys.argv) > 2 else 432
    net = load_net()
    subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
    for name in subs:
        d = os.path.join(tdir, name)
        a, b = load(f"{d}/a.png"), load(f"{d}/b.png")
        if not os.path.exists(f"{d}/pred_rife_full.png"):
            save(rife_full(net, a, b), f"{d}/pred_rife_full.png")
        save(rife_low(net, a, b, short), f"{d}/pred_rife_low{short}.png")
        print(f"{name}: low({short}) 완료  ({a.shape[-1]}x{a.shape[-2]})")
    print(f"\n✅ RIFE 예측 {len(subs)}개 → {tdir}/*/pred_rife_[full|low].png")
