"""30fps 멀티플라이어 상한 무인 벤치 — 실 60fps GT (~/Movies 화면녹화).
저fps 소스를 실 60fps 프레임으로 합성 (보수적: 실제 30fps보다 틴당 모션이 큼):
  4x: (i,i+4) 소스, GT i+1,2,3 (t=.25/.5/.75) → 15→60. 30→120의 하한(틴당 모션 2배).
  3x: (i,i+3) 소스, GT i+1,2  (t=1/3,2/3)     → 20→60. 30→90의 하한(1.5배).
  2x: (i,i+2) 소스, GT i+1    (t=.5)          → 30→60. 현 배포 그대로.
각 멀티플라이어에서 (티어 × exact앵커수) 구성별 per-t SSIM + 최악 틴(아티팩트 지표).
앵커 그룹핑/근사는 RIFEEngine 배포 코드와 동일: 그룹 j=[j*n/A .. (j+1)*n/A-1], 앵커=그룹중앙 t,
근사 틴 = f0·(t/anc), f1·((1-t)/(1-anc)). 예산: sum(exact)=A·티어비용 ≤ gap·0.8.
사용: python multiplier_sweep.py <gtframes_dir> [모션최소px]
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
COST = {288: 7.2, 360: 10.9, 432: 16.7, 540: 26.1}   # v2 ANE predict p50 (ms), 이 세션 실측
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

def compose(a, b, flow, m, s0, s1):
    return warp(a, flow[:, :2] * s0) * m + warp(b, flow[:, 2:4] * s1) * (1 - m)

def groups(n, A):
    """배포 코드와 동일한 그룹 분할 → [(anchor_t_idx, [covered_idx...])]."""
    out = []
    for j in range(A):
        lo = j * n // A
        hi = (j + 1) * n // A - 1
        out.append(((lo + hi) // 2, list(range(lo, hi + 1))))
    return out

def evaluate(net, a, b, gts, ts, tier, A, cache):
    """구성(tier,A)로 ts 전부 합성, 각 t의 SSIM 리스트 반환."""
    per_t = [None] * len(ts)
    for anc_i, cov in groups(len(ts), A):
        anc = ts[anc_i]
        key = (tier, round(anc, 4))
        if key not in cache:
            cache[key] = flow_at(net, a, b, tier, anc)
        fl, m = cache[key]
        for ti in cov:
            t = ts[ti]
            out = compose(a, b, fl, m, t / anc, (1 - t) / (1 - anc))
            per_t[ti] = ssim(out, gts[ti])
    return per_t

# 멀티플라이어별 (소스 stride, t 리스트, GT 오프셋, 라벨)
MULTS = {
    "2x (30→60)": (2, [0.5], [1]),
    "3x (20→60)": (3, [1/3, 2/3], [1, 2]),
    "4x (15→60)": (4, [0.25, 0.5, 0.75], [1, 2, 3]),
}
# 각 멀티플라이어에서 볼 구성: (티어, exact앵커수). ref=예산초과 상한 참고.
CONFIGS = {
    "2x (30→60)": [(432, 1), (540, 1)],
    "3x (20→60)": [(288, 2), (432, 1), (360, 2), (540, 1), (432, 2)],
    "4x (15→60)": [(288, 3), (432, 1), (360, 2), (540, 1), (432, 3)],
}
BUDGET30 = 33.3 * 0.8   # 30fps 갭 예산 (ms)

if __name__ == '__main__':
    gtdir = sys.argv[1]
    motion_min = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    motion_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9   # 장면전환(과대 flow) 배제
    net = load_net("../v425/train_log/flownet.pkl").eval().to(DEV)
    segs = []
    for d in sorted(glob.glob(f"{gtdir}/seg*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) >= 6:
            segs.append([load(p).to(DEV) for p in fr])
    print(f"[dev={DEV}] 세그먼트 {len(segs)}개, 모션≥{motion_min}px 쿼드만\n")

    acc = {mk: {c: {"per_t": [], "worst": []} for c in CONFIGS[mk]} for mk in MULTS}
    motions = []
    nquad = {mk: 0 for mk in MULTS}
    for imgs in segs:
        n = len(imgs)
        for mk, (stride, ts, gtoff) in MULTS.items():
            for i in range(0, n - stride, stride):   # 비중첩 쿼드
                a, b = imgs[i], imgs[i + stride]
                gts = [imgs[i + o] for o in gtoff]
                # 모션 게이트: 소스 갭 flow 크기 — 평균은 정적 배경에 희석되므로
                # p90(움직이는 영역 대표)으로 판단. 정적 씬 배제, 실동작만 측정.
                with torch.no_grad():
                    fl, _ = flow_at(net, a, b, 288, 0.5)
                mago = fl[:, :2].pow(2).sum(1).sqrt().flatten().cpu()
                mag = torch.quantile(mago, 0.9).item()
                if mk == "4x (15→60)":
                    motions.append(mag)
                if mag < motion_min or mag > motion_max:
                    continue
                nquad[mk] += 1
                cache = {}
                for tier, A in CONFIGS[mk]:
                    pt = evaluate(net, a, b, gts, ts, tier, A, cache)
                    acc[mk][(tier, A)]["per_t"].append(pt)
                    acc[mk][(tier, A)]["worst"].append(min(pt))

    if motions:
        m = np.array(motions)
        npass = int((m >= motion_min).sum())
        print(f"4x 소스갭(15fps) 모션 p90 분포(전 {len(m)}쿼드): 중앙{np.median(m):.1f}px 평균{m.mean():.1f} "
              f"범위 {m.min():.1f}~{m.max():.1f}px | 게이트≥{motion_min} 통과 {npass}/{len(m)} "
              f"(→30fps 4x는 이 1/2 수준)\n")
    for mk in MULTS:
        ts = MULTS[mk][1]
        print(f"══ {mk}  쿼드 {nquad[mk]}개 ══")
        print(f"{'구성':<16} {'cost':>6} {'30fps':>6} {'평균SSIM':>8} {'최악틴':>7}  per-t(t=" +
              "/".join(f"{t:.2f}" for t in ts) + ")")
        rows = []
        for (tier, A) in CONFIGS[mk]:
            pt = np.array(acc[mk][(tier, A)]["per_t"])  # (쿼드, len ts)
            if len(pt) == 0: continue
            cost = A * COST[tier]
            feas = "✅" if cost <= BUDGET30 else "✗예산"
            per_t_mean = pt.mean(0)
            rows.append((tier, A, cost, feas, pt.mean(), np.array(acc[mk][(tier,A)]["worst"]).mean(), per_t_mean))
        rows.sort(key=lambda r: -r[4])
        for tier, A, cost, feas, mean, worst, per_t_mean in rows:
            lab = f"{tier}×{A}ex" + ("+ap" if A < len(ts) else "")
            pts = " ".join(f"{v:.3f}" for v in per_t_mean)
            print(f"{lab:<16} {cost:5.1f}m {feas:>6} {mean:8.4f} {worst:7.4f}  {pts}")
        print()
