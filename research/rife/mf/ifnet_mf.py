"""A1(N5) 멀티프레임 IFNet — v3 그래프(피라미드 체이닝) + 과거 프레임 P = img_{-1}.

설계 원칙: **사전학습 v4.25 가중치에서 파인튜닝**할 수 있게, 새 입력·출력을 0으로 초기화하면
출력이 v3와 정확히 같아지도록 만든다(zero-init extension). 무에서 학습할 코퍼스가 없다(worklog A1).
  입력 x: [img_{-1}, img0, img1] 9ch, t: (1,1,1,1)
  레벨 묶음 A=[img0,f0], B=[img1,f1], P=[img_{-1},f_{-1}] 각 7ch — v3처럼 묶음 워프.
  flow 6ch = [t→0, t→1, P 잔차]. P의 실제 flow = (t→0)·(1+t)/t + 잔차 (등속 외삽 prior).
  mask 2ch = [m(A vs B 로짓, v3와 동일), mP(P 게이트 로짓)] — 합성:
      base = sigmoid(m)·wA + (1−sigmoid(m))·wB ;  out = base·(1−sigmoid(mP)) + wP·sigmoid(mP)
  mP의 lastconv 바이어스를 크게 음수로 두면 P 기여 ≈ 0 → v3와 같다.
블록 입력 채널: block0 = A,B,P,t = 22 / block1~4 = wA,wB,wP,t, mask2, feat8, flow6 = 38.
블록 출력(레벨 해상도): flow 6 + mask 2 + feat 8 = 16ch (v3는 13).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(i, o, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        x = self.relu(self.cnn0(x)); x = self.relu(self.cnn1(x)); x = self.relu(self.cnn2(x))
        return self.cnn3(x)

class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

OUT_CH = 16   # flow 6 + mask 2 + feat 8

class IFBlock(nn.Module):
    def __init__(self, in_planes, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * OUT_CH, 4, 2, 1), nn.PixelShuffle(2))
    def forward_level(self, x):
        return self.lastconv(self.convblock(self.conv0(x)))

def up2(t):
    return F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)

class IFNetMF(nn.Module):
    """p_blocks: P를 입력으로 받는 블록 수(앞에서부터, 기본 5=전부). 나머지 블록은 P 워프 없이
    [wA, wB, t, mask2, feat8, flow6] = 31ch — P의 유용성은 mf 특징으로만 전달된다.
    p_feat: P를 특징까지 7ch로 쓸지(기본), 이미지 3ch만 쓸지(인코더 1회 절약)."""
    def __init__(self, warp, p_blocks=5, p_feat=True):
        super().__init__()
        self.warp = warp
        self.p_blocks = p_blocks
        self.p_feat = p_feat
        pc = 7 if p_feat else 3
        cs = [192, 128, 96, 64, 32]
        blocks = []
        for i, c in enumerate(cs):
            withP = i < p_blocks
            if i == 0:
                n_in = 7 * 2 + (pc if withP else 0) + 1
            else:
                n_in = 7 * 2 + (pc if withP else 0) + 1 + 2 + 8 + 6
            blocks.append(IFBlock(n_in, c=c))
        self.block0, self.block1, self.block2, self.block3, self.block4 = blocks
        self.encode = Head()

    def forward(self, x, t, scale_list=(16, 8, 4, 2, 1)):
        imgP, img0, img1 = x[:, :3], x[:, 3:6], x[:, 6:9]
        A = torch.cat((img0, self.encode(img0)), 1)
        B = torch.cat((img1, self.encode(img1)), 1)
        P = torch.cat((imgP, self.encode(imgP)), 1) if self.p_feat else imgP
        kP = (1.0 + t) / t                               # 등속 외삽 배율 (t=0.5 → 3)
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow = None   # 6ch: t→0, t→1, P 잔차 (레벨 px)
        mf = None     # mask 2 + feat 8
        for i, s in enumerate(scale_list):
            if s != 1:
                A_s = F.interpolate(A, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
                B_s = F.interpolate(B, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
                P_s = F.interpolate(P, scale_factor=1.0 / s, mode="bilinear", align_corners=False) if i < self.p_blocks else None
            else:
                A_s, B_s, P_s = A, B, P
            t_s = A_s[:, :1] * 0 + t
            withP = i < self.p_blocks
            if flow is None:
                tmp = blocks[i].forward_level(torch.cat((A_s, B_s, P_s, t_s) if withP else (A_s, B_s, t_s), 1))
                flow = tmp[:, :6]
                mf = tmp[:, 6:]
            else:
                flow = up2(flow) * 2.0
                mf = up2(mf)
                wA = self.warp(A_s, flow[:, 0:2])
                wB = self.warp(B_s, flow[:, 2:4])
                if withP:
                    wP = self.warp(P_s, flow[:, 0:2] * kP + flow[:, 4:6])
                    tmp = blocks[i].forward_level(torch.cat((wA, wB, wP, t_s, mf, flow), 1))
                else:
                    tmp = blocks[i].forward_level(torch.cat((wA, wB, t_s, mf, flow), 1))
                flow = flow + tmp[:, :6]
                mf = tmp[:, 6:]
        # 앱으로 넘기는 flow: [t→0, t→1, t→−1(외삽+잔차)] — 워프 커널이 그대로 쓰게 합쳐서 낸다
        flowOut = torch.cat((flow[:, 0:4], flow[:, 0:2] * kP + flow[:, 4:6]), 1)
        return flowOut, mf[:, 0:2]


def from_v3(v3, warp, mp_bias=-20.0):
    """permute_input_channels()를 마친 v3 IFNet의 가중치를 MF로 이식한다 (zero-init extension).
    새 입력 채널(P 묶음·mP·P 잔차 flow)의 가중치 = 0, 새 출력(P 잔차 flow) = 0, mP 바이어스 = mp_bias.
    이러면 MF의 flow[:, :4]·mask[:, :1]이 v3와 같아야 한다 — ane_gate.py가 확인한다."""
    mf = IFNetMF(warp)   # 기본 구성(p_blocks=5, p_feat=True)만 이식 대상
    mf.encode.load_state_dict(v3.encode.state_dict())
    # 입력 채널 대응 (v3 → MF). block0: [A 0:7, B 7:14, t 14] → [A 0:7, B 7:14, P 14:21, t 21]
    in0 = {i: i for i in range(14)}; in0[14] = 21
    # block1~4: v3 [wA 0:7, wB 7:14, t 14, mask 15, feat 16:24, flow 24:28]
    #        → MF [wA 0:7, wB 7:14, wP 14:21, t 21, m 22, mP 23, feat 24:32, flow 32:36, Pres 36:38]
    inN = {i: i for i in range(14)}; inN[14] = 21; inN[15] = 22
    inN.update({16 + k: 24 + k for k in range(8)}); inN.update({24 + k: 32 + k for k in range(4)})
    # 출력(레벨) 채널: v3 [flow 0:4, mask 4, feat 5:13] → MF [flow 0:4, Pres 4:6, m 6, mP 7, feat 8:16]
    outmap = {k: k for k in range(4)}; outmap[4] = 6; outmap.update({5 + k: 8 + k for k in range(8)})
    with torch.no_grad():
        for name in ["block0", "block1", "block2", "block3", "block4"]:
            b3, bm = getattr(v3, name), getattr(mf, name)
            imap = in0 if name == "block0" else inN
            c3, cm = b3.conv0[0][0], bm.conv0[0][0]
            cm.weight.zero_()
            for i3, im in imap.items():
                cm.weight[:, im] = c3.weight[:, i3]
            cm.bias.copy_(c3.bias)
            bm.conv0[1].load_state_dict(b3.conv0[1].state_dict())
            bm.convblock.load_state_dict(b3.convblock.state_dict())
            l3, lm = b3.lastconv[0], bm.lastconv[0]          # ConvTranspose2d: weight (in, out, 4, 4)
            lm.weight.zero_(); lm.bias.zero_()
            for o3, om in outmap.items():                     # PixelShuffle(2): 레벨 채널 j ← convT 채널 4j..4j+3
                lm.weight[:, 4 * om:4 * om + 4] = l3.weight[:, 4 * o3:4 * o3 + 4]
                lm.bias[4 * om:4 * om + 4] = l3.bias[4 * o3:4 * o3 + 4]
            lm.bias[4 * 7:4 * 7 + 4] = mp_bias                # mP 로짓 → sigmoid ≈ 0
    return mf

