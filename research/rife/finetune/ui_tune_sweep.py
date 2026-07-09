"""정지-UI 프리즈 파라미터 스윕 — 실시간 EMA와 동일한 마스크로 (alpha, clo, chi, res_div)
격자를 돌려 인게임채팅 ROI 이득 vs 전체 무회귀를 실측. '튜닝 천장' 확인용.
사용: python ui_tune_sweep.py <dumps_dir> <short>
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "../quality_eval")
from model_compare import load_net, rife_low, ssim
from temporal_ui_spike import load, gray, gblur

DEV = torch.device("cpu")

def ema_mask(frames, alpha, clo, chi, res_div, sgate=(0.004, 0.02)):
    """실시간 검출기와 동일: 소스 1/res_div에서 hp EMA mean/sq → 일관성 → smoothstep.
    마지막 프레임 시점의 마스크 반환 (실시간이 그 순간 쓰는 것과 동치)."""
    H, W = frames[0].shape[-2:]
    mh, mw = max(64, H // res_div), max(64, W // res_div)
    mean = None; sq = None
    for f in frames:
        g = F.interpolate(gray(f), size=(mh, mw), mode='bilinear', align_corners=False)
        hp = g - gblur(g, 5, 1.5)
        if mean is None: mean = hp.clone(); sq = hp * hp
        else:
            mean = mean * (1 - alpha) + hp * alpha
            sq = sq * (1 - alpha) + hp * hp * alpha
    var = (sq - mean * mean).clamp(min=0)
    cons = mean.abs() / (var.sqrt() + 0.004)
    def ss(x, a, b): return ((x - a) / (b - a)).clamp(0, 1)
    m = ss(cons, clo, chi) * ss(mean.abs(), sgate[0], sgate[1])
    return F.interpolate(gblur(m, 5, 2), size=(H, W), mode='bilinear', align_corners=False)

def roi_ssim(x, y, box):
    y0, y1, x0, x1 = box
    return ssim(x[:, :, y0:y1, x0:x1], y[:, :, y0:y1, x0:x1])

if __name__ == '__main__':
    dumps, short = sys.argv[1], int(sys.argv[2])
    net = load_net("../v425/train_log/flownet.pkl").eval()
    seqs = []
    for d in sorted(glob.glob(f"{dumps}/2026*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) >= 6 and Image.open(fr[0]).size == (1920, 1080):
            seqs.append([load(p) for p in fr])
    print(f"시퀀스 {len(seqs)}개. baseline 계산...")
    # baseline interp 캐시 (파라미터 무관)
    base = []   # (interp, gt, a, b, roi)
    for imgs in seqs:
        H, W = imgs[0].shape[-2:]
        roi = (int(H*0.44), int(H*0.74), 0, int(W*0.42))
        for i in range(len(imgs) - 2):
            a, gt, b = imgs[i], imgs[i+1], imgs[i+2]
            with torch.no_grad(): interp = rife_low(net, a, b, short)
            base.append((interp, gt, a, b, roi, imgs, i))
    b_full = np.mean([ssim(x[0], x[1]) for x in base])
    b_roi = np.mean([roi_ssim(x[0], x[1], x[4]) for x in base])
    print(f"baseline: 전체 {b_full:.4f}  인게임ROI {b_roi:.4f}\n")
    print(f"{'alpha':>6} {'clo':>4} {'chi':>4} {'div':>3} {'cover%':>6} {'전체Δ':>8} {'ROIΔ':>8}")
    for alpha in (0.02, 0.04, 0.08):
        for clo in (0.5, 0.8, 1.2):
            for div in (2, 1):
                chi = clo + 1.2
                # 시퀀스별 마스크 1회 (마지막 프레임)
                masks = {}
                fulls, rois, covs = [], [], []
                for (interp, gt, a, b, roi, imgs, i) in base:
                    key = id(imgs)
                    if key not in masks: masks[key] = ema_mask(imgs, alpha, clo, chi, div)
                    M = masks[key]
                    src = 0.5 * (a + b)
                    fr = interp * (1 - M) + src * M
                    fulls.append(ssim(fr, gt)); rois.append(roi_ssim(fr, gt, roi)); covs.append(M.mean().item())
                print(f"{alpha:6.2f} {clo:4.1f} {chi:4.1f} {div:3d} {np.mean(covs)*100:6.1f} "
                      f"{np.mean(fulls)-b_full:+8.4f} {np.mean(rois)-b_roi:+8.4f}")
