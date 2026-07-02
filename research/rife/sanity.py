import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # flownet.pkl 상대경로 보정
import torch
from ifnet import load_ifnet, warp

net, missing, unexpected = load_ifnet('flownet.pkl')
net.eval()

torch.manual_seed(0)
H = W = 256
# 텍스처 있는 64x64 패치를 검은 배경에 놓고 이동 (특징 충분 → flow 추정 가능)
patch = torch.rand(1, 3, 64, 64)
def place(x0):
    img = torch.zeros(1, 3, H, W)
    img[:, :, 96:160, x0:x0+64] = patch
    return img

img0 = place(40)
img1 = place(140)  # 100px 이동

with torch.no_grad():
    for t in [0.25, 0.5, 0.75]:
        ts = torch.full((1, 1, H, W), t)
        merged, flow, sig = net(img0, img1, ts)
        # 패치 중심 x 위치 추정 (밝기 무게중심)
        col = merged[0].mean(0).mean(0)  # [W]
        xs = torch.arange(W, dtype=torch.float32)
        centroid = (col * xs).sum() / col.sum().clamp(min=1e-6)
        expected = 72 + 100 * t  # 패치 중심: 40+32=72 → 140+32=172
        print(f"t={t}: 패치중심 x={centroid:.0f} (기대 {expected:.0f})  flow[{flow[:, :2].min():.0f}~{flow[:, :2].max():.0f}]")
