# RIFE v4.25 CoreML export용 v2 — "다운스케일 후 워프" 재구성 (ANE resample 비용 ⅓).
#
# 원본은 매 반복의 warp(이미지 2 + 특징 2)를 모델 풀해상도에서 수행한 뒤 블록이 입력을
# 곧장 1/scale로 다운스케일한다 — 블록이 보는 정보는 다운스케일본뿐인데 워프는 풀해상도
# 비용을 낸다 (ANE 병목 실측: conv-only 8.5ms vs full 20.8ms @360 — 차이가 거의 resample).
# v2는 순서를 교환: 블록 입력 해상도(1/scale)로 먼저 줄이고 그 해상도에서 워프.
#   풀해상도 등가 워프 수 16 → 4×(1/64+1/16+1/4) + 4(scale1) ≈ 5.3 (⅓).
# 주의: bilinear 다운스케일∘워프는 정확히 가환이 아님 — 패리티(px err)와 삼중항 SSIM
# 게이트를 export 스크립트에서 검증한다. 모듈 정의는 원본과 동일(state_dict 호환),
# forward 로직만 다르다.
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward_prescaled(self, x, flow_s, scale=1):
        # x: 이미 1/scale 해상도. flow_s: 1/scale 해상도 + 그 해상도 픽셀 단위 (또는 None).
        if flow_s is not None:
            x = torch.cat((x, flow_s), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale       # 모델 해상도 픽셀 단위로 복원 (원본과 동일)
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, x, timestep=0.5, scale_list=[16, 8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        timestep = (img0[:, :1] * 0 + 1) * timestep
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        flow = None
        mask = None
        feat = None
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            s = scale_list[i]
            inv = 1.0 / s
            if flow is None:
                x_in = torch.cat((img0, img1, f0, f1, timestep), 1)
                x_s = F.interpolate(x_in, scale_factor=inv, mode="bilinear", align_corners=False) if s != 1 else x_in
                flow, mask, feat = blocks[i].forward_prescaled(x_s, None, scale=s)
            else:
                if s != 1:
                    # 입력 해상도로 먼저 축소 → 그 해상도에서 워프 (핵심 교환)
                    flow_s = F.interpolate(flow, scale_factor=inv, mode="bilinear", align_corners=False) * inv
                    i0 = F.interpolate(img0, scale_factor=inv, mode="bilinear", align_corners=False)
                    i1 = F.interpolate(img1, scale_factor=inv, mode="bilinear", align_corners=False)
                    g0 = F.interpolate(f0, scale_factor=inv, mode="bilinear", align_corners=False)
                    g1 = F.interpolate(f1, scale_factor=inv, mode="bilinear", align_corners=False)
                    aux = F.interpolate(torch.cat((timestep, mask, feat), 1), scale_factor=inv,
                                        mode="bilinear", align_corners=False)
                else:
                    flow_s = flow
                    i0, i1, g0, g1 = img0, img1, f0, f1
                    aux = torch.cat((timestep, mask, feat), 1)
                w0 = warp(i0, flow_s[:, :2])
                w1 = warp(i1, flow_s[:, 2:4])
                wf0 = warp(g0, flow_s[:, :2])
                wf1 = warp(g1, flow_s[:, 2:4])
                # 채널 순서 = 원본과 동일: (warped0, warped1, wf0, wf1, timestep, mask, feat) + flow
                x_s = torch.cat((w0, w1, wf0, wf1, aux), 1)
                fd, m0, feat = blocks[i].forward_prescaled(x_s, flow_s, scale=s)
                mask = m0
                flow = flow + fd
        # FlowHead 호환 반환 (flow_list[4], mask_list[4]=pre-sigmoid, merged 미사용)
        return [flow, flow, flow, flow, flow], mask, None
