import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 상대경로 보정
# RIFE v4.25 (공식 코드) 로드 + arbitrary-t 검증
import sys, types, torch
import torch.nn.functional as F

# IFNet_HDv3.py가 요구하는 model.warplayer 셔밍
def warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vgrid = grid + flow
    vx = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl

sys.path.insert(0, 'v425/train_log')
from IFNet_HDv3 import IFNet

def load425():
    net = IFNet()
    sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if not (m.startswith('teacher') or m.startswith('caltime'))]
    return net, real_missing, unexpected

if __name__ == '__main__':
    net, missing, unexpected = load425()
    print('missing(추론 필수):', missing)
    print('unexpected:', [u for u in unexpected][:6])
    net.eval()

    torch.manual_seed(0)
    H = W = 512
    SHIFT = 24
    # 자연스러운 텍스처: 저주파 노이즈 (블러로 부드럽게)
    base = torch.rand(1, 3, H, W + SHIFT)
    base = F.avg_pool2d(base, 9, stride=1, padding=4)
    img0 = base[:, :, :, :W].contiguous()          # 원본
    img1 = base[:, :, :, SHIFT:SHIFT + W].contiguous()  # 왼쪽으로 24px 이동한 뷰
    x = torch.cat((img0, img1), 1)

    with torch.no_grad():
        for t in [0.25, 0.5, 0.75]:
            flow_list, mask, merged = net(x, timestep=t, scale_list=[16, 8, 4, 2, 1])
            out = merged[4]
            # 보간 프레임이 base의 어느 x-오프셋과 가장 일치하는지 탐색 (기대: SHIFT*t)
            errs = []
            for off in range(SHIFT + 1):
                ref = base[:, :, :, off:off + W]
                errs.append(float((out - ref).abs().mean()))
            best = min(range(len(errs)), key=lambda i: errs[i])
            print(f"t={t}: 최적 정합 오프셋={best}px (기대 {SHIFT*t:.0f}px), 오차={errs[best]:.4f}")
