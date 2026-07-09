"""앵커 정확도 vs flow 해상도 — 멀티-t 보간의 미측정 설계 질문 결판.
(i, i+4) 쌍(66.7ms 갭 = 저fps 대모션 체제)에서 GT t=0.25/0.5/0.75 실프레임으로:
  288exact  : 288 flow, t별 exact predict (3 predict)
  360a2     : 360 flow, 앵커 2개(0.375, 0.75) + 선형 스케일 근사
  432a1     : 432 flow, 앵커 1개(0.5) + 스케일 근사   (v1.1.4 배포 동작)
  540a1     : 540 flow, 앵커 1개(0.5) + 스케일 근사
  432exact  : 432 flow, t별 exact (상한 참고 — 실시간 예산 밖)
스케일 근사 = 배포 워프와 동일: f0×(t/anchor), f1×((1-t)/(1-anchor)).
사용: python anchor_vs_res.py <dumps_dir> [seq제한]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "../quality_eval")
from model_compare import load_net, ssim, warp
from temporal_ui_spike import load

SCALES = [16, 8, 4, 2, 1]

def pad64(n): return ((n + 63) // 64) * 64

def flow_at(net, a, b, short, t):
    """배포 경로: short로 축소 → predict(timestep=t) → 풀해상도 flow/mask."""
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        fl, mask, _ = net(torch.cat((alo, blo), 1), timestep=float(t), scale_list=SCALES)
    flow = F.interpolate(fl[4], size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return flow, m

def compose(a, b, flow, m, s0, s1):
    wa = warp(a, flow[:, :2] * s0)
    wb = warp(b, flow[:, 2:4] * s1)
    return wa * m + wb * (1 - m)

if __name__ == '__main__':
    dumps = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    net = load_net("../v425/train_log/flownet.pkl").eval()
    seqs = []
    for d in sorted(glob.glob(f"{dumps}/2026*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) >= 6 and Image.open(fr[0]).size == (1920, 1080):
            seqs.append([load(p) for p in fr])
    seqs = seqs[:lim]
    TS = [0.25, 0.5, 0.75]
    res = {k: [] for k in ("288exact", "360a2", "432a1", "540a1", "432exact")}
    npairs = 0
    for imgs in seqs:
        for i in range(0, len(imgs) - 4, 2):
            a, b = imgs[i], imgs[i + 4]
            gts = {0.25: imgs[i + 1], 0.5: imgs[i + 2], 0.75: imgs[i + 3]}
            npairs += 1
            # 288 exact (t별 predict)
            for t in TS:
                fl, m = flow_at(net, a, b, 288, t)
                res["288exact"].append(ssim(compose(a, b, fl, m, 1, 1), gts[t]))
            # 432 exact (상한)
            for t in TS:
                fl, m = flow_at(net, a, b, 432, t)
                res["432exact"].append(ssim(compose(a, b, fl, m, 1, 1), gts[t]))
            # 360 앵커 2 (0.375: t 0.25/0.5 담당, 0.75: exact)
            fl_a, m_a = flow_at(net, a, b, 360, 0.375)
            fl_b, m_b = flow_at(net, a, b, 360, 0.75)
            for t, (fl, m, anc) in ((0.25, (fl_a, m_a, 0.375)), (0.5, (fl_a, m_a, 0.375)), (0.75, (fl_b, m_b, 0.75))):
                res["360a2"].append(ssim(compose(a, b, fl, m, t / anc, (1 - t) / (1 - anc)), gts[t]))
            # 432/540 앵커 1 (0.5)
            for short, key in ((432, "432a1"), (540, "540a1")):
                fl, m = flow_at(net, a, b, short, 0.5)
                for t in TS:
                    res[key].append(ssim(compose(a, b, fl, m, t / 0.5, (1 - t) / 0.5), gts[t]))
    print(f"쌍 {npairs}개 × t3  (66.7ms 갭 = 저fps 대모션 체제)\n")
    print(f"{'variant':<10} {'SSIM':>8}   predict비용(30fps 예산 26.6ms, v2 실측)")
    cost = {"288exact": "3×7.2=21.6ms ✅", "360a2": "2×10.9=21.8ms ✅", "432a1": "1×16.7ms ✅",
            "540a1": "1×26.1ms ⚠️경계", "432exact": "3×16.7=50ms ❌(참고상한)"}
    for k in ("288exact", "360a2", "432a1", "540a1", "432exact"):
        print(f"{k:<10} {np.mean(res[k]):8.4f}   {cost[k]}")
