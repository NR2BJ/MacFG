"""A1 최종 판정 평가 — **CPU에서 결정적으로**.

MPS 평가는 비결정적으로 값이 오염된다(2026-10-01 실측: 같은 샘플의 A/B 블렌드 PSNR이 한 실행에선 25.118,
다른 실행에선 24.367 — 같은 호출의 전체 출력은 멀쩡. 출발 가중치가 0이라 달라질 수 없는 값이다. CPU는 매번 같다).
학습 손실은 평균이라 버티지만 **판정 수치는 CPU로만** 낸다.
팔: v3(출발 모델의 A/B 블렌드 = v3와 정확히 같음) / 체크포인트들(각자 P 소스: past 또는 A=대조군).
출력: 샘플별 PSNR(전체·움직임 영역) JSON + 요약(Δ중앙·평균·최악25%·움직임 영역).
사용: python eval_cpu.py --data <mf> --ckpt run1=<pt>:past --ckpt ctrl=<pt>:A [--n 60] [--p-blocks 3] --out <json>
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, os, random, math
import numpy as np
import torch
import torch.nn.functional as F
import train_mf as T
T.DEV = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() - 1))

def luma(x): return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

class Hybrid(torch.nn.Module):
    """flow는 fnet에서, mask는 mnet에서 — 파인튜닝 이득/흐림이 flow와 블렌딩 중 어디서 오는지 가른다."""
    def __init__(self, fnet, mnet):
        super().__init__(); self.fnet, self.mnet = fnet, mnet
    def forward(self, x, t, scales):
        f, _ = self.fnet(x, t, scales); _, m = self.mnet(x, t, scales)
        return f, m

def outputs(net, P, A, B, t):
    with torch.no_grad():
        x = torch.cat((T.to_model(P), T.to_model(A), T.to_model(B)), 1)
        flow, mask = net(x, torch.full((1, 1, 1, 1), t), T.SCALES)
        H, W = A.shape[-2:]
        fl = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        sx, sy = W / T.MW, H / T.MH
        fl = fl * torch.tensor([sx, sy, sx, sy, sx, sy]).view(1, 6, 1, 1)
        mk = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        out, base = T.synth(A, B, P, fl, mk)
        return out.clamp(0, 1), base.clamp(0, 1)

def grad_energy(x):
    l = luma(x) * 255.0
    gx = (l[:, :, 1:-1:2, 2::2] - l[:, :, 1:-1:2, 0:-2:2]).abs()
    gy = (l[:, :, 2::2, 1:-1:2] - l[:, :, 0:-2:2, 1:-1:2]).abs()
    h = min(gx.shape[2], gy.shape[2]); w = min(gx.shape[3], gy.shape[3])
    return float((gx[:, :, :h, :w] + gy[:, :, :h, :w]).mean())

def sharp(o, G):
    """출력/GT 그래디언트 에너지 비 — 1.0 = 정답만큼 선명, <1 = 흐림 (InterpBench sharp=와 같은 정의)."""
    return grad_energy(o) / max(grad_energy(G), 1e-9)

def psnr(o, G, m=None):
    e = ((o - G) ** 2).mean(1, keepdim=True)
    mse = float(e.mean()) if m is None else float((e * m).sum() / m.sum().clamp(min=1))
    return -10 * math.log10(max(mse, 1e-10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="이름=경로:past|A")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--p-blocks", type=int, default=3)
    ap.add_argument("--p-feat", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    ev = [w["id"] for w in man["eval"]]
    random.Random(1).shuffle(ev)
    ev = ev[:a.n]
    v3 = T.IFNetV3()
    sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    v3.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    v3 = v3.eval().permute_input_channels()
    base_net, _ = T.from_v3(v3, T.warp_manual, p_blocks=a.p_blocks, p_feat=a.p_feat)
    base_net.eval()
    def load_arm(path):
        if path == "v3":
            return base_net
        sd = torch.load(path, map_location='cpu')
        if any(k.startswith("net.") for k in sd):          # --arch v3 체크포인트(V3AsMF 래퍼)
            net = T.V3AsMF(T.IFNetV3())
        else:
            net, _ = T.from_v3(v3, T.warp_manual, p_blocks=a.p_blocks, p_feat=a.p_feat)
        net.load_state_dict(sd)
        return net.eval()
    arms = []
    for spec in a.ckpt:
        name, rest = spec.split("=", 1)
        path, src = rest.rsplit(":", 1)
        if "+" in path:
            pf, pm = path.split("+", 1)
            arms.append((name, Hybrid(load_arm(pf), load_arm(pm)).eval(), src))
        else:
            arms.append((name, load_arm(path), src))
    rows = []
    for i, wid in enumerate(ev):
        for ks, t in (((1, 3, 4, 5), 0.5), ((0, 3, 4, 6), 1 / 3)):
            P, A, G, B = [T.load_jpg(os.path.join(a.data, "eval", f"{wid}_{k}.jpg")).unsqueeze(0) for k in ks]
            d = torch.maximum((luma(A) - luma(B)).abs(), (luma(A) - luma(P)).abs())
            moving = (F.max_pool2d(d, 5, 1, 2) >= 0.02).float()
            _, b3 = outputs(base_net, P, A, B, t)       # A/B 블렌드 = v3
            r = {"id": wid, "t": t, "v3": psnr(b3, G), "v3_mov": psnr(b3, G, moving), "v3_sh": sharp(b3, G)}
            for name, net, src in arms:
                o, _ = outputs(net, A if src == "A" else P, A, B, t)
                r[name] = psnr(o, G); r[name + "_mov"] = psnr(o, G, moving); r[name + "_sh"] = sharp(o, G)
            rows.append(r)
        print(f"  {i + 1}/{len(ev)} " + " ".join(f"{k}={rows[-1][k]:.2f}" for k in rows[-1]
                                                  if k not in ("id", "t") and not k.endswith("_mov") and not k.endswith("_sh")), flush=True)
    json.dump(rows, open(a.out, "w"))
    v = np.array([r["v3"] for r in rows]); vm = np.array([r["v3_mov"] for r in rows])
    worst = np.argsort(v)[:max(1, len(v) // 4)]
    print(f"\n삼중항 {len(rows)}개 (CPU, 결정적) — v3 PSNR 중앙 {np.median(v):.3f} 평균 {v.mean():.3f}  선명도 {np.mean([r['v3_sh'] for r in rows]):.3f}")
    for name, _, src in arms:
        x = np.array([r[name] for r in rows]); xm = np.array([r[name + "_mov"] for r in rows])
        d, dm = x - v, xm - vm
        print(f"  {name}(P={src}): Δ중앙 {np.median(d):+.3f} Δ평균 {d.mean():+.3f}  최악25% Δ중앙 {np.median(d[worst]):+.3f} "
              f"Δ평균 {d[worst].mean():+.3f}  움직임영역 Δ중앙 {np.median(dm):+.3f}  (Δ<−0.1dB {int((d < -0.1).sum())}/{len(d)})  "
              f"선명도 {np.mean([r[name + '_sh'] for r in rows]):.3f}")
