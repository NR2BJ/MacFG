"""실영상 프레임 윈도우에서 VFI 삼중항 선별.
각 윈도우의 연속 프레임 (i,i+1,i+2)를 (a, gt, b)로 보고:
  - d_ab = mean|a-b|  (움직임 크기; 클수록 보간 변별력↑)
  - 장면전환 배제: d_ab 상한 + a→gt, gt→b 대칭성(컷이면 한쪽만 급변)
모션 큰 순으로 윈도우별 분산 선택 → triplets/tNN/{a,gt,b}.png
사용: python select_triplets.py <frames_dir> <out_dir> <N>
"""
import sys, os, shutil
import numpy as np
from PIL import Image

frames_dir, out_dir, N = sys.argv[1], sys.argv[2], int(sys.argv[3])

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

cands = []  # (score, window, i, [paths])
for win in sorted(os.listdir(frames_dir)):
    wdir = os.path.join(frames_dir, win)
    if not os.path.isdir(wdir): continue
    fs = sorted(f for f in os.listdir(wdir) if f.endswith(".png"))
    imgs = [load(os.path.join(wdir, f)) for f in fs]
    for i in range(len(imgs) - 2):
        a, gt, b = imgs[i], imgs[i+1], imgs[i+2]
        d_ab = float(np.mean(np.abs(a - b)))
        d_agt = float(np.mean(np.abs(a - gt)))
        d_gtb = float(np.mean(np.abs(gt - b)))
        # 움직임 있음(>1.5%) + 컷 아님(<12%) + 대칭(양쪽 변화 비슷)
        if d_ab < 0.010 or d_ab > 0.15: continue
        sym = abs(d_agt - d_gtb) / (d_agt + d_gtb + 1e-6)
        if sym > 0.55: continue
        cands.append((d_ab, win, i, [os.path.join(wdir, fs[i+k]) for k in range(3)]))

# 윈도우별 라운드로빈으로 분산 (한 윈도우 독점 방지), 각 윈도우 내 모션 큰 순
from collections import defaultdict
bywin = defaultdict(list)
for c in cands: bywin[c[1]].append(c)
for w in bywin: bywin[w].sort(key=lambda c: -c[0])
order = []
wins = sorted(bywin)
idx = {w: 0 for w in wins}
while len(order) < N and any(idx[w] < len(bywin[w]) for w in wins):
    for w in wins:
        if idx[w] < len(bywin[w]):
            order.append(bywin[w][idx[w]]); idx[w] += 1
            if len(order) >= N: break

os.makedirs(out_dir, exist_ok=True)
for j, (score, win, i, paths) in enumerate(order):
    td = os.path.join(out_dir, f"t{j:02d}")
    os.makedirs(td, exist_ok=True)
    for name, src in zip(["a", "gt", "b"], paths):
        shutil.copy(src, os.path.join(td, f"{name}.png"))
    print(f"t{j:02d}: {win} i={i} motion={score*100:.1f}%")
print(f"\n선택 {len(order)}개 → {out_dir}  (후보 {len(cands)}개 중)")
