"""N5 스파이크 — 멀티프레임(과거 컨텍스트)이 최악틴을 사주는가, 훈련 없이 오라클로 상한 판별.
아이디어: 2프레임 interp가 놓치는 오클루전/큰모션 픽셀을 과거 프레임이 복원 가능한지.
  R_narrow = interp(t0, t1) @ t         — 현재 2프레임 배포
  R_wide   = interp(과거, t1) @ (1+t)/2 — 과거를 좌측 프레임으로 끌어온 멀티프레임 프록시
  R_oracle = 픽셀별 min|·-GT|(narrow, wide) — 완벽한 과거통합의 상한(달성불가 상한선)
headroom = SSIM(oracle) - SSIM(narrow). 큰 값 = 과거가 실제로 복원 = N5 훈련 가치.
~0 = 2프레임이 진짜 천장. wide 브래킷을 66ms로 유지하려 30fps 케이던스(i,i+2) 사용,
과거=i-2, GT 틴=i+1(t=0.5). 실 60fps GT.
사용: python multiframe_spike.py <gtframes> [모션min] [모션max]
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
TIER = 432
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def pad64(n): return ((n + 63) // 64) * 64

def flow_at(net, a, b, short, t):
    H, W = a.shape[-2:]
    s = short / min(H, W)
    lH, lW = pad64(int(round(H * s))), pad64(int(round(W * s)))
    alo = F.interpolate(a, size=(lH, lW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(lH, lW), mode='bilinear', align_corners=False)
    with torch.no_grad():
        fl, mask, _ = net(torch.cat((alo, blo), 1), timestep=float(t), scale_list=SCALES)
    flow = F.interpolate(fl[4], size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / lW, H / lH
    flow = flow * torch.tensor([sx, sy, sx, sy], device=a.device).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return flow, m

def interp_tier(net, a, b, t, short):
    fl, m = flow_at(net, a, b, short, t)
    return warp(a, fl[:, :2]) * m + warp(b, fl[:, 2:4]) * (1 - m)

def interp(net, a, b, t):
    return interp_tier(net, a, b, t, TIER)

def oracle(rn, rw, gt):
    """픽셀별로 GT에 더 가까운 재구성 선택 (luma 오차 기준)."""
    en = (rn - gt).abs().mean(1, keepdim=True)
    ew = (rw - gt).abs().mean(1, keepdim=True)
    pick_n = (en <= ew).float()
    return rn * pick_n + rw * (1 - pick_n)

if __name__ == '__main__':
    gtdir = sys.argv[1]
    mmin = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    mmax = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    net = load_net("../v425/train_log/flownet.pkl").eval().to(DEV)
    segs = []
    for d in sorted(glob.glob(f"{gtdir}/seg*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) >= 8:
            segs.append([load(p).to(DEV) for p in fr])
    print(f"[dev={DEV}] 세그 {len(segs)}개, 모션 p90 {mmin}~{mmax}px (30fps 케이던스 i,i+2)\n")

    rows = {"narrow": [], "wide": [], "oracle": [], "orc_tier": []}
    motions = []
    for imgs in segs:
        n = len(imgs)
        for i in range(2, n - 2):                     # 과거 i-2, 쌍 (i,i+2), GT i+1
            t0, t1, past, gt = imgs[i], imgs[i + 2], imgs[i - 2], imgs[i + 1]
            with torch.no_grad():
                fl, _ = flow_at(net, t0, t1, 288, 0.5)
            mag = torch.quantile(fl[:, :2].pow(2).sum(1).sqrt().flatten().cpu(), 0.9).item()
            motions.append(mag)
            if mag < mmin or mag > mmax:
                continue
            rn = interp(net, t0, t1, 0.5)             # 2프레임: 틴 i+1은 (t0,t1) 중점
            rw = interp(net, past, t1, 0.75)          # 과거~t1 브래킷서 i+1은 국소t=3/4
            rn2 = interp_tier(net, t0, t1, 0.5, 288)  # 같은 프레임, 다른 티어 (노이즈 바닥용)
            rows["narrow"].append(ssim(rn, gt))
            rows["wide"].append(ssim(rw, gt))
            rows["oracle"].append(ssim(oracle(rn, rw, gt), gt))       # 과거 통합 상한
            rows["orc_tier"].append(ssim(oracle(rn, rn2, gt), gt))    # 같은프레임 cherry-pick 바닥

    m = np.array(motions)
    npass = len(rows["narrow"])
    print(f"모션 p90: 중앙{np.median(m):.1f} 범위{m.min():.1f}~{m.max():.1f}px | 밴드통과 {npass}쿼드\n")
    if npass:
        rn, rw, ro, rt = (np.array(rows[k]) for k in ("narrow", "wide", "oracle", "orc_tier"))
        k = max(1, npass // 4)
        worst = np.argsort(rn)[:k]                     # narrow 하위 25% = 아티팩트 후보
        print(f"{'':16} {'평균SSIM':>9} {'최악25%':>9}")
        print(f"{'narrow(2프레임)':17} {rn.mean():9.4f} {rn[worst].mean():9.4f}")
        print(f"{'wide(과거단독)':17} {rw.mean():9.4f} {rw[worst].mean():9.4f}")
        print(f"{'oracle:같은프레임':17} {rt.mean():9.4f} {rt[worst].mean():9.4f}  ← cherry-pick 노이즈 바닥")
        print(f"{'oracle:과거통합':17} {ro.mean():9.4f} {ro[worst].mean():9.4f}  ← 상한(달성불가)")
        raw = ro - rn                                  # 총 oracle 이득
        floor = rt - rn                                # 같은프레임 cherry-pick 이득
        net_past = raw - floor                         # 과거의 순수 기여 (노이즈 제거)
        print(f"\n▶ 과거 프레임 순 기여 (총 headroom - cherry-pick 바닥):")
        print(f"   평균  {raw.mean():+.4f} - {floor.mean():+.4f} = {net_past.mean():+.4f}")
        print(f"   최악25%  {raw[worst].mean():+.4f} - {floor[worst].mean():+.4f} = {net_past[worst].mean():+.4f}")
        print(f"   → 상한. 훈련모델은 이 30~60% 포착 예상 = 최악틴 ~{net_past[worst].mean()*0.45:+.3f} 현실치")
        print(f"   판정: 순기여 최악25% >~0.015면 N5 가치 有")
