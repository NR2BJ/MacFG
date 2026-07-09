"""(b) 시간축 UI 레이어 재구성 — 합성 방식 3안을 오프라인 GT로 비교 (설계 결정용).
uiMean = 프레임 픽셀의 EMA (정지 구조는 선명하게 남고, 이동 배경은 평균화되어 뭉개짐).
  v1 freeze-src   : out = interp(1-M) + 0.5(A+B)·M          (현재 배포 — 기준)
  v2 freeze-mean  : out = interp(1-M) + uiMean·M            (복원 레이어로 통째 교체)
  v3 detail-stamp : out = interp + k·M·(uiMean - blur(uiMean))  (안정 텍스트 엣지만 이식,
                     투과 배경은 계속 보간되어 움직임 — 반투명에 이론상 최적)
  v4 = v1 + detail-stamp (프리즈 위에 안정 엣지 보강)
실시간과 동일하게 상태를 프레임 순서대로 누적(B 시점까지)해 각 삼중항 평가.
사용: python ui_layer_variants.py <dumps_dir> <short>
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "../quality_eval")
from model_compare import load_net, rife_low, ssim
from temporal_ui_spike import load, gray, gblur

A_MASK = 0.04   # 일관성 EMA율 (배포와 동일)
A_MEAN = 0.08   # uiMean EMA율 (12프레임 시퀀스라 조금 빠르게)
CLO, CHI = 0.5, 1.7

def roi_ssim(x, y, box):
    y0, y1, x0, x1 = box
    return ssim(x[:, :, y0:y1, x0:x1], y[:, :, y0:y1, x0:x1])

def run_sequence(net, imgs, short):
    H, W = imgs[0].shape[-2:]
    roi = (int(H*0.44), int(H*0.74), 0, int(W*0.42))
    mh, mw = H // 2, W // 2
    mean_hp = None; sq_hp = None; ui_mean = None
    rows = {k: [] for k in ("v0", "v1", "v2", "v3", "v4")}
    rois = {k: [] for k in rows}
    for i, f in enumerate(imgs):
        g = F.interpolate(gray(f), size=(mh, mw), mode='bilinear', align_corners=False)
        hp = g - gblur(g, 5, 1.5)
        if mean_hp is None:
            mean_hp, sq_hp = hp.clone(), hp * hp
            ui_mean = f.clone()
        else:
            mean_hp = mean_hp * (1 - A_MASK) + hp * A_MASK
            sq_hp = sq_hp * (1 - A_MASK) + hp * hp * A_MASK
            ui_mean = ui_mean * (1 - A_MEAN) + f * A_MEAN
        if i < 2:
            continue
        a, gt, b = imgs[i - 2], imgs[i - 1], imgs[i]
        var = (sq_hp - mean_hp * mean_hp).clamp(min=0)
        cons = mean_hp.abs() / (var.sqrt() + 0.004)
        def ss(x, lo, hi): return ((x - lo) / (hi - lo)).clamp(0, 1)
        m = ss(cons, CLO, CHI) * ss(mean_hp.abs(), 0.004, 0.02)
        M = F.interpolate(gblur(m, 5, 2), size=(H, W), mode='bilinear', align_corners=False)
        with torch.no_grad():
            interp = rife_low(net, a, b, short)
        src = 0.5 * (a + b)
        um_detail = ui_mean - torch.cat([gblur(ui_mean[:, c:c+1], 11, 3.0) for c in range(3)], 1)
        outs = {
            "v0": interp,
            "v1": interp * (1 - M) + src * M,
            "v2": interp * (1 - M) + ui_mean * M,
            "v3": (interp + M * um_detail).clamp(0, 1),
            "v4": ((interp * (1 - M) + src * M) + 0.5 * M * um_detail).clamp(0, 1),
        }
        for k, o in outs.items():
            rows[k].append(ssim(o, gt))
            rois[k].append(roi_ssim(o, gt, roi))
    return rows, rois

if __name__ == '__main__':
    dumps, short = sys.argv[1], int(sys.argv[2])
    net = load_net("../v425/train_log/flownet.pkl").eval()
    agg = {k: [] for k in ("v0", "v1", "v2", "v3", "v4")}
    agg_roi = {k: [] for k in agg}
    n = 0
    for d in sorted(glob.glob(f"{dumps}/2026*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) < 6 or Image.open(fr[0]).size != (1920, 1080):
            continue
        imgs = [load(p) for p in fr]
        rows, rois = run_sequence(net, imgs, short)
        for k in agg:
            agg[k] += rows[k]; agg_roi[k] += rois[k]
        n += 1
    print(f"시퀀스 {n}개, 삼중항 {len(agg['v0'])}개  (마스크 clo{CLO}/chi{CHI}, uiMean a={A_MEAN})\n")
    print(f"{'variant':<14} {'전체SSIM':>9} {'Δ':>8}   {'인게임ROI':>9} {'Δ':>8}")
    b_full, b_roi = np.mean(agg["v0"]), np.mean(agg_roi["v0"])
    names = {"v0": "baseline", "v1": "freeze-src(현)", "v2": "freeze-mean", "v3": "detail-stamp", "v4": "v1+stamp"}
    for k in ("v0", "v1", "v2", "v3", "v4"):
        f_, r_ = np.mean(agg[k]), np.mean(agg_roi[k])
        print(f"{names[k]:<14} {f_:9.4f} {f_-b_full:+8.4f}   {r_:9.4f} {r_-b_roi:+8.4f}")
