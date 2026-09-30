# RIFE v4.25 CoreML export용 v3 — "피라미드 체이닝" (design.md O3-1).
#
# v2는 블록마다 출력(flow·mask·feat 13ch)을 모델 풀해상도로 올렸다가, 다음 블록이 그걸(그리고 이미지·특징을)
# 다시 1/scale로 내렸다 — 풀해상도 왕복이 블록당 한 번씩, 그리고 그 뒤의 채널 slice도 풀해상도에서 일어났다
# (ANE 프로파일: slice 21% + add 15% + upsample 11% + convT 9.5% + resample 8.5% vs conv 8.4%).
# v3는:
#   1) 블록 출력은 자기 레벨 해상도(1/s)에 둔다. 다음 레벨(1/(s/2))로는 2배 업샘플만 한다.
#      flow는 레벨 픽셀 단위로 누적한다: flow_{s/2} = up2(flow_s)·2 + Δ.
#   2) 이미지+특징을 7채널로 묶어 워프한다(같은 flow를 쓰므로) — resample 4회 → 2회/블록.
#   3) 그래서 블록 입력 채널 순서가 원본과 달라진다 → 첫 conv(conv0[0][0])의 입력 채널을 load 후 순열로
#      재배열한다(permute_input_channels). 모듈 정의는 원본과 동일(state_dict 호환).
# 수치: bilinear "×s 올림 → ÷s' 내림"과 "×2 올림"은 정확히 같지 않다 — export_coreml_v3.py의 게이트(원본 대비
# flow px 오차, 변환 패리티, 속도)와 gate_v3.py(움직임 큰 실삼중항 40개, 앱 경로 PSNR)로 검증한다.
# 결과(2026-09-30): CoreML v3 − v2 PSNR 중앙값 +0.001~+0.003dB, predict −21~29% — Models/README.md.
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

    def forward_level(self, x):
        # x: 레벨 해상도 입력(채널 순서는 permute_input_channels로 맞춘 v3 순서). 출력도 레벨 해상도 13ch.
        feat = self.conv0(x)
        feat = self.convblock(feat)
        return self.lastconv(feat)

def up2(t):
    return F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def permute_input_channels(self):
        """원본 채널 순서 → v3 순서로 첫 conv 가중치를 재배열한다. load_state_dict 뒤에 한 번 호출.
        원본 block0 입력: [img0 0:3, img1 3:6, f0 6:10, f1 10:14, t 14]
        v3   block0 입력: [img0, f0, img1, f1, t]
        원본 block1~4:   [w0 0:3, w1 3:6, wf0 6:10, wf1 10:14, t 14, mask 15, feat 16:24, flow 24:28]
        v3   block1~4:   [w0, wf0, w1, wf1, t, mask, feat, flow]   (워프를 이미지+특징 7ch 묶음으로 하기 때문)"""
        head = [0, 1, 2, 6, 7, 8, 9, 3, 4, 5, 10, 11, 12, 13]
        perm0 = head + [14]
        permN = head + list(range(14, 28))
        with torch.no_grad():
            for b, perm in [(self.block0, perm0), (self.block1, permN), (self.block2, permN),
                            (self.block3, permN), (self.block4, permN)]:
                c = b.conv0[0][0]
                c.weight.copy_(c.weight[:, perm, :, :].clone())
        return self

    def forward(self, x, timestep=0.5, scale_list=[16, 8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        A = torch.cat((img0, f0), 1)     # 7ch: 같은 flow(0→t)로 워프
        B = torch.cat((img1, f1), 1)     # 7ch: 같은 flow(1→t)로 워프
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow = None       # 현재 레벨 해상도, 레벨 픽셀 단위
        mf = None         # mask(1)+feat(8), 현재 레벨 해상도
        for i in range(5):
            s = scale_list[i]
            if s != 1:
                A_s = F.interpolate(A, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
                B_s = F.interpolate(B, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
            else:
                A_s, B_s = A, B
            t_s = A_s[:, :1] * 0 + timestep
            if flow is None:
                tmp = blocks[i].forward_level(torch.cat((A_s, B_s, t_s), 1))
                flow = tmp[:, :4]
                mf = tmp[:, 4:]
            else:
                flow = up2(flow) * 2.0            # 이전 레벨 → 이번 레벨 (해상도 2배, 픽셀 단위 2배)
                mf = up2(mf)
                wA = warp(A_s, flow[:, :2])
                wB = warp(B_s, flow[:, 2:4])
                tmp = blocks[i].forward_level(torch.cat((wA, wB, t_s, mf, flow), 1))
                flow = flow + tmp[:, :4]
                mf = tmp[:, 4:]
        mask = mf[:, :1]
        # FlowHead 호환 반환 (flow_list[4], mask=pre-sigmoid, merged 미사용). 마지막 레벨(s=1)은 모델 풀해상도.
        return [flow, flow, flow, flow, flow], mask, None
