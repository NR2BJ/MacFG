"""내부 warp 소거 실험 — IFNet의 블록간 grid_sample(ANE 미지원, 전환비용 주범)을
identity로 바꾸면 품질이 얼마나 깨지나. conv-only 그래프(빠름)가 쓸만하면 대승.
사용: python ablate_warp.py <triplets_dir>   → pred_rife_nowarp432.png 생성
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, types
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

# warp = identity 셔밍 (관건)
pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = lambda img, flow: img
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, '../v425/train_log')
from IFNet_HDv3 import IFNet

def real_warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vgrid = grid + flow
    vx = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

net = IFNet()
sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
net.eval()

def pad64(n): return ((n + 63) // 64) * 64
def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
def save(t, p):
    Image.fromarray((t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(p)

def predict_low(a, b, short=432):
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H*s))), pad64(int(round(W*s)))
    alo = F.interpolate(a, size=(lH,lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH,lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        flow_list, mask, _ = net(torch.cat((alo,blo),1), timestep=0.5, scale_list=[16,8,4,2,1])
    flow = F.interpolate(flow_list[4], size=(H,W), mode='bilinear', align_corners=False)
    sx, sy = W/lW, H/lH
    flow = flow * torch.tensor([sx,sy,sx,sy]).view(1,4,1,1)
    m = torch.sigmoid(F.interpolate(mask, size=(H,W), mode='bilinear', align_corners=False))
    # 최종 풀해상도 워프는 실제 워프 (Metal 몫과 동일)
    wa = real_warp(a, flow[:, :2]); wb = real_warp(b, flow[:, 2:4])
    return wa*m + wb*(1-m)

tdir = sys.argv[1]
for name in sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir,d))):
    d = os.path.join(tdir, name)
    a, b = load(f"{d}/a.png"), load(f"{d}/b.png")
    save(predict_low(a, b), f"{d}/pred_rife_nowarp432.png")
    print(f"{name} 완료")
print("✅ 내부워프 소거 예측 → pred_rife_nowarp432.png")
