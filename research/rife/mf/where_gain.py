"""A1 교란 점검 — MF의 PSNR 이득이 **어디서** 나오는가.

의심: 평가 영상(오버워치 방송)은 채팅·스코어보드·HUD 같은 정적 UI가 크다. P가 있으면 모델이 "세 프레임이
같은 곳은 제자리"를 배워 정적 UI 번짐을 줄일 수 있는데, 앱은 그 영역을 이미 층 분리(UILayer)로 처리한다.
그러면 앱에서의 실제 이득은 이 평가보다 작다. 그래서 이득을 둘로 나눈다:
  정적 픽셀: P·A·B가 서로 거의 같다(루마 차 < --static-thr, 5x5 최대 필터로 경계 포함)  — 층이 맡는 영역에 가깝다
  움직임 픽셀: 나머지                                                               — 엔진 몫
출력: 영역별 MSE 기여·PSNR 변화, 이득 지도(평균 |v3−GT|−|MF−GT|) PNG, 이득 상·하위 예시 크롭.
사용: pixi run ... python where_gain.py --data <mf데이터> --ckpt <mf_N.pt> --out <dir> [--n 60] [--p-blocks 3]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, json, os, random, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import train_mf as T

def luma(x): return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])

def run(net, P, A, B, t, base_only):
    with torch.no_grad():
        x = torch.cat((T.to_model(P), T.to_model(A), T.to_model(B)), 1)
        flow, mask = net(x, torch.full((1, 1, 1, 1), t, device=T.DEV), T.SCALES)
        H, W = A.shape[-2:]
        fl = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        sx, sy = W / T.MW, H / T.MH
        fl = fl * torch.tensor([sx, sy, sx, sy, sx, sy], device=T.DEV).view(1, 6, 1, 1)
        mk = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        out, base = T.synth(A, B, P, fl, mk)
        return (base if base_only else out).clamp(0, 1), torch.sigmoid(mk[:, 1:2])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--p-blocks", type=int, default=3)
    ap.add_argument("--p-feat", action="store_true")
    ap.add_argument("--static-thr", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    ev = [w["id"] for w in man["eval"]]
    random.Random(1).shuffle(ev)
    ev = ev[:a.n]
    v3 = T.IFNetV3()
    sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    v3.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    v3 = v3.eval().permute_input_channels()
    net, _ = T.from_v3(v3, T.warp_manual, p_blocks=a.p_blocks, p_feat=a.p_feat)
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    net = net.to(T.DEV).eval()
    # 출발 모델(= v3의 A/B 블렌드)을 따로 만든다 — 같은 구성, 이식 직후 가중치
    base_net, _ = T.from_v3(v3, T.warp_manual, p_blocks=a.p_blocks, p_feat=a.p_feat)
    base_net = base_net.to(T.DEV).eval()

    acc = {"static": [0.0, 0.0, 0], "moving": [0.0, 0.0, 0]}   # [sse v3, sse MF, 픽셀수]
    gain_map = None; gate_map = None; stat_map = None
    per = []
    for wid in ev:
        for ks, t in (((1, 3, 4, 5), 0.5), ((0, 3, 4, 6), 1 / 3)):
            P, A, G, B = [T.load_jpg(os.path.join(a.data, "eval", f"{wid}_{k}.jpg")).unsqueeze(0).to(T.DEV) for k in ks]
            o3, _ = run(base_net, P, A, B, t, True)
            om, g = run(net, P, A, B, t, False)
            la, lb, lp = luma(A), luma(B), luma(P)
            d = torch.maximum((la - lb).abs(), (la - lp).abs())
            static = (F.max_pool2d(d, 5, 1, 2) < a.static_thr).float()
            e3 = ((o3 - G) ** 2).mean(1, keepdim=True); em = ((om - G) ** 2).mean(1, keepdim=True)
            for name, m in (("static", static), ("moving", 1 - static)):
                acc[name][0] += float((e3 * m).sum()); acc[name][1] += float((em * m).sum()); acc[name][2] += int(m.sum())
            gm = ((o3 - G).abs() - (om - G).abs()).mean(1, keepdim=True)
            small = lambda x: F.interpolate(x, size=(270, 480), mode='area')[0, 0].cpu().numpy()
            gain_map = small(gm) if gain_map is None else gain_map + small(gm)
            gate_map = small(g) if gate_map is None else gate_map + small(g)
            stat_map = small(static) if stat_map is None else stat_map + small(static)
            p3 = -10 * math.log10(max(float(e3.mean()), 1e-10)); pm = -10 * math.log10(max(float(em.mean()), 1e-10))
            mv = 1 - static
            pm3 = -10 * math.log10(max(float((e3 * mv).sum() / mv.sum().clamp(min=1)), 1e-10))
            pmm = -10 * math.log10(max(float((em * mv).sum() / mv.sum().clamp(min=1)), 1e-10))
            per.append((wid, t, p3, pm, pm3, pmm, float(static.mean())))
    n = len(per)
    print(f"삼중항 {n}개 — 정적 픽셀 비율 평균 {np.mean([r[6] for r in per]):.1%}")
    tot3 = acc["static"][0] + acc["moving"][0]; totm = acc["static"][1] + acc["moving"][1]
    for name in ("static", "moving"):
        s3, sm, cnt = acc[name]
        print(f"  {name:6s}: 픽셀 {cnt / (cnt + 1e-9) and cnt:>12d}  v3 MSE {s3 / cnt:.6f} → MF {sm / cnt:.6f}  "
              f"(영역 PSNR {-10 * math.log10(s3 / cnt):.3f} → {-10 * math.log10(sm / cnt):.3f})  "
              f"전체 오차 감소분 중 {100 * (s3 - sm) / max(tot3 - totm, 1e-12):.0f}%")
    d_all = np.array([r[3] - r[2] for r in per]); d_mov = np.array([r[5] - r[4] for r in per])
    print(f"  프레임 PSNR Δ중앙 {np.median(d_all):+.3f} / 움직임 영역만 PSNR Δ중앙 {np.median(d_mov):+.3f} Δ평균 {d_mov.mean():+.3f}")
    worst = np.argsort([r[4] for r in per])[:max(1, n // 4)]
    print(f"  움직임 영역 기준 최악25%: Δ중앙 {np.median(d_mov[worst]):+.3f} Δ평균 {d_mov[worst].mean():+.3f}")
    def save_map(m, name, lo, hi):
        v = np.clip((m - lo) / (hi - lo), 0, 1)
        Image.fromarray((v * 255).astype(np.uint8)).resize((960, 540), Image.NEAREST).save(os.path.join(a.out, name))
    gm = gain_map / n
    lim = max(abs(float(np.percentile(gm, 1))), abs(float(np.percentile(gm, 99))), 1e-6)
    # 이득 지도: 회색 0, 밝음=MF가 나음, 어두움=MF가 나쁨
    save_map(gm, "gain_map.png", -lim, lim)
    save_map(gate_map / n, "gate_map.png", 0, max(float((gate_map / n).max()), 1e-6))
    save_map(stat_map / n, "static_map.png", 0, 1)
    json.dump([list(map(lambda x: x if isinstance(x, str) else float(x), r)) for r in per], open(os.path.join(a.out, "per.json"), "w"))
    print(f"  지도 저장: {a.out}/gain_map.png (회색=0, 밝음=MF 우세, ±{lim:.4f}), gate_map.png, static_map.png")
