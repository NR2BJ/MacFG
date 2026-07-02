# RIFE IFNet_HDv3 (Practical-RIFE, MIT) — flownet.pkl 텐서 shape에서 정확히 역설계.
# 구조: 3 flow block (c=90) + block_tea(추론 미사용). 활성 PReLU. conv1=flow(4ch), conv2=mask(1ch).
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=90):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1))
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1))

    def forward(self, x, flow, scale=1.0):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * (1. / scale)
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False)
        return flow, mask

def warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    vgrid = torch.cat((vgrid_x, vgrid_y), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)  # 추론 미사용 (strict 로드용)

    def forward(self, img0, img1, timestep, scale_list=(4.0, 2.0, 1.0)):
        # img0/img1: [B,3,H,W] 0..1, timestep: [B,1,H,W]
        B, _, H, W = img0.shape
        flow = torch.zeros(B, 4, H, W, device=img0.device, dtype=img0.dtype)
        mask = torch.zeros(B, 1, H, W, device=img0.device, dtype=img0.dtype)
        blocks = [self.block0, self.block1, self.block2]
        for i, block in enumerate(blocks):
            warped0 = warp(img0, flow[:, :2]) if i > 0 else img0
            warped1 = warp(img1, flow[:, 2:4]) if i > 0 else img1
            x = torch.cat((warped0, warped1, timestep), 1)
            f, m = block(x, flow, scale=scale_list[i])
            flow = flow + f
            mask = mask + m if i > 0 else m
        sig = torch.sigmoid(mask)
        warped0 = warp(img0, flow[:, :2])
        warped1 = warp(img1, flow[:, 2:4])
        merged = warped0 * sig + warped1 * (1 - sig)
        return merged, flow, sig

def load_ifnet(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net = IFNet()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return net, missing, unexpected
