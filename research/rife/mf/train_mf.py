"""A1 ② 멀티프레임 IFNet 파인튜닝 — v3에서 zero-init 이식해 출발, 과거 프레임 P를 쓰는 법을 배운다.

데이터: extract_windows.py의 7프레임 창(학습 = 모델 해상도 640x384).
  30fps 케이던스: P=k1, A=k3, GT=k4(t=½), B=k5      20fps: P=k0, A=k3, GT=k4/k5(t=⅓/⅔), B=k6
모드: --train new  = 새 채널만(기존 채널 기울기 0 — 백본 동결, 망각 없음)
      --train all  = 전체(새 채널 lr, 기존 채널 lr×--base-lr-mul)
손실: 모델 해상도 합성 프레임의 L1 + Laplacian 피라미드(train_finetune.py와 같은 구성).
평가(--eval-every마다 + 끝): 1080p 평가 창에서 **배포 경로**(늘려 넣기 → flow 업샘플 → 풀해상도 워프·블렌드)
  PSNR, 기준 = 같은 창에서 출발 모델(= v3). 판정은 차이의 중앙값과 **기준이 나쁜 25%(최악틴)** 에서의 차이.
MPS: grid_sample backward가 없어 gather 기반 수동 워프(train_finetune.warp_manual)를 쓴다.
사용: pixi run ... python train_mf.py --data ~/Documents/MacFG/datasets/mf --out <dir> [--steps 3000]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time, argparse, os, json, random, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "../finetune")
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def warp_manual(img, flow):
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing='ij')
    x = xx[None] + flow[:, 0]; y = yy[None] + flow[:, 1]
    x0 = torch.floor(x); y0 = torch.floor(y)
    wx = (x - x0).unsqueeze(1); wy = (y - y0).unsqueeze(1)
    x0i = x0.long().clamp(0, W - 1); x1i = (x0.long() + 1).clamp(0, W - 1)
    y0i = y0.long().clamp(0, H - 1); y1i = (y0.long() + 1).clamp(0, H - 1)
    flat = img.reshape(B, C, H * W)
    def g(yi, xi):
        idx = (yi * W + xi).reshape(B, 1, H * W).expand(B, C, H * W)
        return torch.gather(flat, 2, idx).reshape(B, C, H, W)
    Ia, Ib, Ic, Id = g(y0i, x0i), g(y0i, x1i), g(y1i, x0i), g(y1i, x1i)
    return Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp_manual
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, '../v425/train_log')
from IFNet_HDv3_coreml_v3 import IFNet as IFNetV3
from ifnet_mf import from_v3

SCALES = (16, 8, 4, 2, 1)

class V3AsMF(torch.nn.Module):
    """v3를 MF 인터페이스로 감싼다: 입력 [P, A, B] 9ch 중 P는 무시, 출력 flow 6ch(P=t→0 복사)·mask 2ch(mP=−1e4)."""
    def __init__(self, v3):
        super().__init__()
        self.net = v3
    def forward(self, x, t, scale_list=SCALES):
        fl, m, _ = self.net(x[:, 3:9], timestep=t, scale_list=list(scale_list))
        f = fl[4]
        return torch.cat((f, f[:, 0:2]), 1), torch.cat((m, torch.full_like(m, -1e4)), 1)
MW, MH = 640, 384

def load_jpg(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)

def synth(A, B, P, flow, mask):
    """flow 6ch [t→0, t→1, t→−1], mask 2ch [m, mP] — 같은 해상도에서 합성."""
    wA = warp_manual(A, flow[:, 0:2]); wB = warp_manual(B, flow[:, 2:4]); wP = warp_manual(P, flow[:, 4:6])
    m = torch.sigmoid(mask[:, 0:1]); g = torch.sigmoid(mask[:, 1:2])
    base = wA * m + wB * (1 - m)
    return base * (1 - g) + wP * g, base

def gauss_kernel(ch, dev):
    k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
    k = (k[:, None] * k[None, :]); k /= k.sum()
    return k.expand(ch, 1, 5, 5).contiguous()

def lap_loss(x, y, levels=4):
    ker = gauss_kernel(3, x.device)
    def down(t):
        return F.conv2d(F.pad(t, (2, 2, 2, 2), mode='reflect'), ker, stride=2, groups=3)
    loss = 0
    for _ in range(levels):
        xd, yd = down(x), down(y)
        xu = F.interpolate(xd, size=x.shape[-2:], mode='bilinear', align_corners=False)
        yu = F.interpolate(yd, size=y.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + (x - xu - (y - yu)).abs().mean()
        x, y = xd, yd
    return loss

class Windows:
    def __init__(self, root, ids):
        self.root, self.ids = root, ids
    def sample(self, rng):
        wid = rng.choice(self.ids)
        if rng.random() < 0.6:
            ks, t = (1, 3, 4, 5), 0.5                    # P, A, GT, B — 30fps
        elif rng.random() < 0.5:
            ks, t = (0, 3, 4, 6), 1 / 3                  # 20fps t=⅓
        else:
            ks, t = (0, 3, 5, 6), 2 / 3                  # 20fps t=⅔
        fr = [load_jpg(os.path.join(self.root, "train", f"{wid}_{k}.jpg")) for k in ks]
        if rng.random() < 0.5:
            fr = [f.flip(-1) for f in fr]
        return fr, t

def batch(ds, rng, bs):
    P, A, G, B, T = [], [], [], [], []
    for _ in range(bs):
        (p, a, g, b), t = ds.sample(rng)
        P.append(p); A.append(a); G.append(g); B.append(b); T.append(t)
    st = lambda L: torch.stack(L).to(DEV)
    return st(P), st(A), st(G), st(B), torch.tensor(T, dtype=torch.float32, device=DEV).view(-1, 1, 1, 1)

# ── 평가: 1080p 배포 경로
def to_model(x):
    return F.interpolate(x, size=(MH, MW), mode='bilinear', align_corners=False)

def deploy_eval(net, P, A, G, B, t):
    """P/A/G/B: (1,3,1080,1920) on DEV. 반환 (psnr_mf, psnr_base) — base = 같은 모델의 A/B 블렌드만(P 게이트 0)."""
    with torch.no_grad():
        x = torch.cat((to_model(P), to_model(A), to_model(B)), 1)
        flow, mask = net(x, torch.full((1, 1, 1, 1), t, device=DEV), SCALES)
        H, W = A.shape[-2:]
        fl = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        sx, sy = W / MW, H / MH
        fl = fl * torch.tensor([sx, sy, sx, sy, sx, sy], device=DEV).view(1, 6, 1, 1)
        mk = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        out, base = synth(A, B, P, fl, mk)
        ps = lambda o: -10 * math.log10(max(float(((o.clamp(0, 1) - G) ** 2).mean()), 1e-10))
        return ps(out), ps(base)

def load_eval(root, ids, n):
    # 1080p float로 미리 올리면 60창×7장 ≈ 10GB — 경로만 두고 평가 때 디코드한다
    return [(wid, os.path.join(root, "eval")) for wid in ids[:n]]

P_SOURCE = "past"

def run_eval(net, items):
    net.eval()
    rows = []
    for wid, d in items:
        for ks, t in (((1, 3, 4, 5), 0.5), ((0, 3, 4, 6), 1 / 3)):
            P, A, G, B = [load_jpg(os.path.join(d, f"{wid}_{k}.jpg")).unsqueeze(0).to(DEV) for k in ks]
            if P_SOURCE == "A": P = A
            rows.append(deploy_eval(net, P, A, G, B, t))
    net.train()
    return rows

def summarize(rows, ref_rows):
    mf = np.array([r[0] for r in rows]); r0 = np.array([r[0] for r in ref_rows])
    d = mf - r0
    worst = np.argsort(r0)[:max(1, len(r0) // 4)]
    return (f"PSNR 중앙 {np.median(mf):.3f} (기준 {np.median(r0):.3f})  Δ중앙 {np.median(d):+.3f}  Δ평균 {d.mean():+.3f}  "
            f"최악25% Δ중앙 {np.median(d[worst]):+.3f} Δ평균 {d[worst].mean():+.3f}  (Δ<−0.1dB {int((d < -0.1).sum())}/{len(d)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-lr-mul", type=float, default=0.1)
    ap.add_argument("--train", choices=["new", "all"], default="new")
    ap.add_argument("--p-blocks", type=int, default=3)
    ap.add_argument("--p-feat", action="store_true")
    ap.add_argument("--mp-bias", type=float, default=-6.0)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="이어 학습할 체크포인트(.pt) — 같은 arch/구성이어야 한다")
    ap.add_argument("--arch", choices=["mf", "v3"], default="mf",
                    help="v3 = 과거 프레임 없이 v3 그래프 가중치만 파인튜닝(도메인 적응 단독 — 배포 시 연산 +0%%, Swift 무변경)")
    ap.add_argument("--p-source", choices=["past", "A"], default="past",
                    help="A = 대조군: P 자리에 A를 넣는다(시간 정보 없이 같은 새 파라미터·같은 학습 — 도메인 적응분 분리)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global P_SOURCE
    P_SOURCE = a.p_source
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    tr_ids = [w["id"] for w in man["train"]]; ev_ids = [w["id"] for w in man["eval"]]
    random.Random(1).shuffle(ev_ids)
    print(f"학습 창 {len(tr_ids)} / 평가 창 {len(ev_ids)} (평가에 {min(a.eval_n, len(ev_ids))}개 × 2 케이던스)", flush=True)

    v3 = IFNetV3()
    sd = torch.load('../v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    v3.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    v3 = v3.eval().permute_input_channels()
    if a.arch == "v3":
        net, masks = V3AsMF(v3), {}
        a.train = "all"                                   # v3 단독은 전체(기존) 파라미터만 있다
        a.base_lr_mul = 1.0
    else:
        net, masks = from_v3(v3, warp_manual, mp_bias=a.mp_bias, p_blocks=a.p_blocks, p_feat=a.p_feat)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location='cpu'))
        print(f"초기화: {a.init}", flush=True)
    net = net.to(DEV).train()
    masks = {k: v.to(DEV) for k, v in masks.items()}
    named = dict(net.named_parameters())
    new_params, base_params = [], []
    for n, p in named.items():
        if n in masks:
            if a.train == "new":
                p.register_hook(lambda g, m=masks[n]: g * m)
            new_params.append(p)                         # 부분 새 채널 텐서 — 새 채널 lr
        elif a.train == "all":
            base_params.append(p)
        else:
            p.requires_grad_(False)
    groups = [{"params": new_params, "lr": a.lr}] if new_params else []
    if base_params: groups.append({"params": base_params, "lr": a.lr * a.base_lr_mul})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999))
    ntrain = sum(int(masks[n].sum()) for n in masks) if a.train == "new" else sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"구성 arch={a.arch} p_blocks={a.p_blocks} p_feat={a.p_feat} mP바이어스={a.mp_bias} 학습={a.train} P={a.p_source} (학습 파라미터 {ntrain / 1e3:.1f}k)", flush=True)

    ds = Windows(a.data, tr_ids)
    ev_items = load_eval(a.data, ev_ids, a.eval_n)
    ref = run_eval(net, ev_items)
    # 출발 시점의 A/B 블렌드(P 게이트 무시) = **순수 v3와 정확히 같다**(새 입력 가중치 0) — 이것이 기준선.
    v3_rows = [(r[1], r[1]) for r in ref]
    print(f"[eval 0] 출발 모델(σ(mP) 누설 포함) vs v3: {summarize(ref, v3_rows)}", flush=True)
    ema, t0 = None, time.time()
    for step in range(1, a.steps + 1):
        # 코사인 감쇠
        for g in opt.param_groups:
            g.setdefault("lr0", g["lr"])
            g["lr"] = g["lr0"] * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        P, A, G, B, T = batch(ds, rng, a.bs)
        if a.p_source == "A": P = A
        flow, mask = net(torch.cat((P, A, B), 1), T, SCALES)
        out, _ = synth(A, B, P, flow, mask)
        loss = (out - G).abs().mean() + lap_loss(out, G)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema = float(loss) if ema is None else ema * 0.98 + float(loss) * 0.02
        if step % 50 == 0:
            gate = float(torch.sigmoid(mask[:, 1:2]).mean())
            print(f"  step {step:5d} loss {ema:.5f}  σ(mP) 평균 {gate:.4f}  {(time.time() - t0) / step:.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            rows = run_eval(net, ev_items)
            print(f"[eval {step}] vs v3: {summarize(rows, v3_rows)}", flush=True)
            torch.save(net.state_dict(), os.path.join(a.out, f"mf_{step}.pt"))
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))

if __name__ == "__main__":
    main()
