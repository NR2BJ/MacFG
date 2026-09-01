"""A5 — 배수 천장: 앵커 1개 근사가 '극단 t'에서 무너지는가.

물음: 24fps 소스를 144Hz에 ×6으로 채우면 틴이 5장(t=1/6..5/6)이다. 배포 경로는 앵커
하나(0.5)를 뽑아 f0×(t/0.5), f1×((1-t)/0.5)로 스케일한다. t가 0.5에서 멀어질수록
이 근사가 무너진다면 ×6은 성능이 아니라 화질로 막히는 것이다.

**교락 제거가 이 스크립트의 요점이다.** 60fps GT에서 틴을 늘리면 브래킷이 같이 커진다
((i,i+2)=33ms에 틴1 / (i,i+6)=100ms에 틴5). 그래서 '배수'와 '모션 크기'가 섞인다.
여기서는 **브래킷을 고정한 채** 같은 쌍 안에서 중앙 t와 극단 t를 비교한다 —
브래킷이 같으므로 차이는 전적으로 t 위치, 즉 근사 탓이다.

천장으로 t별 exact predict를 같이 잰다. exact도 같이 나빠지면 원인은 근사가 아니라
브래킷(대모션)이고, 앵커를 늘려도 안 낫는다.

사용: python tin_density.py <dumps_dir> [seq제한]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, glob
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "../quality_eval")
from model_compare import load_net, ssim, warp
from temporal_ui_spike import load
from anchor_vs_res import flow_at, compose

if __name__ == '__main__':
    dumps = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    net = load_net("../v425/train_log/flownet.pkl").eval()
    seqs = []
    for d in sorted(glob.glob(f"{dumps}/2026*")):
        fr = sorted(glob.glob(f"{d}/frame_*.png"))
        if len(fr) >= 7 and Image.open(fr[0]).size == (1920, 1080):
            seqs.append([load(p) for p in fr])
    seqs = seqs[:lim]
    if not seqs:
        sys.exit("7장 이상 1920x1080 시퀀스 없음")

    SHORT = 360           # 배포 티어
    # (스트라이드, 그 브래킷 안에서 볼 t들) — 스트라이드가 곧 브래킷이다.
    CASES = [(2, [0.5]), (3, [1/3, 2/3]), (4, [0.25, 0.5, 0.75]),
             (6, [1/6, 2/6, 3/6, 4/6, 5/6])]
    out = {}
    for stride, ts in CASES:
        anc = {}   # t -> [ssim]
        exa = {}
        n = 0
        for imgs in seqs:
            for i in range(0, len(imgs) - stride, 2):
                a, b = imgs[i], imgs[i + stride]
                n += 1
                fl0, m0 = flow_at(net, a, b, SHORT, 0.5)      # 앵커 1개
                for t in ts:
                    gt = imgs[i + int(round(t * stride))]
                    anc.setdefault(t, []).append(
                        ssim(compose(a, b, fl0, m0, t / 0.5, (1 - t) / 0.5), gt))
                    fl, m = flow_at(net, a, b, SHORT, t)      # 천장: t별 exact
                    exa.setdefault(t, []).append(ssim(compose(a, b, fl, m, 1, 1), gt))
        out[stride] = (n, anc, exa)

    print(f"flow short={SHORT}, 시퀀스 {len(seqs)}개\n")
    print(f"{'브래킷':<22}{'t':>6}{'앵커1':>9}{'exact':>9}{'근사손실':>10}")
    print("-" * 58)
    for stride, ts in CASES:
        n, anc, exa = out[stride]
        ms = stride * 1000.0 / 60
        for t in ts:
            A, E = np.mean(anc[t]), np.mean(exa[t])
            print(f"{f'x{stride} ({ms:.1f}ms, n={n})':<22}{t:>6.3f}{A:>9.4f}{E:>9.4f}{A-E:>+10.4f}")
        print()
    print("판정: 같은 브래킷 안에서 극단 t의 '근사손실'이 중앙 t보다 크게 나쁘면")
    print("      앵커 1개가 원인이다(=앵커를 늘리면 낫는다). 둘이 비슷하면 브래킷 탓이라")
    print("      앵커를 늘려도 소용없고 배수 천장은 화질로 굳는다.")
