"""v2(다운스케일 후 워프) 넷의 배포경로 화질 — rife_predict.py의 rife_low와 동일 경로를
v2 IFNet으로 실행해 pred_rife_v2_{short}.png 생성 (ladder_metrics로 v1과 직접 비교).
사용: python eval_v2.py <triplets_dir> [short=360]
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
from IFNet_HDv3_coreml_v2 import IFNet

SCALES = [16, 8, 4, 2, 1]

def load_net():
    net = IFNet()
    sd = torch.load(f"{RIFE}/flownet.pkl", map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return net.eval()

def pad64(n): return ((n + 63) // 64) * 64

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

def save(t, p):
    arr = (t.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(p)

def rife_low_v2(net, a, b, short):
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        flow_list, mask, _ = net(torch.cat((alo, blo), 1), timestep=0.5, scale_list=SCALES)
    flow = F.interpolate(flow_list[4], size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    wa = warp(a, flow[:, :2])
    wb = warp(b, flow[:, 2:4])
    return wa * m + wb * (1 - m)

if __name__ == '__main__':
    tdir = sys.argv[1]
    short = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    net = load_net()
    subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
    for name in subs:
        d = os.path.join(tdir, name)
        a, b = load(f"{d}/a.png"), load(f"{d}/b.png")
        save(rife_low_v2(net, a, b, short), f"{d}/pred_rife_v2_{short}.png")
        print(f"{name}: v2({short}) 완료")
    print(f"✅ v2 예측 {len(subs)}개")
