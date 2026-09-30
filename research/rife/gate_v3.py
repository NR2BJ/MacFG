"""RIFE v3 그래프 관문 — 움직임 큰 실삼중항에서 원본(v1)·배포(v2)·후보(v3)의 배포 경로 화질.

export_coreml_v3.py의 ①은 쌍 하나(중앙 시퀀스 frame 0/2)로 재는데, 그 쌍은 움직임이 ±1px뿐이라
"피라미드 체이닝이 bilinear 왕복과 다르다"는 차이가 드러날 수가 없다 — 관문이 약하다.
여기선 코퍼스 전체에서 움직임이 큰 삼중항 (a, gt, b)을 골라, **앱과 같은 경로**로 가운데 프레임을 만든다:
  모델 입력 크기로 늘려 넣기(rifePack) → flow/mask → 원 해상도로 bilinear 업샘플·px 스케일
  → 가장자리 고정 bilinear 워프 + sigmoid 마스크 블렌드(rifeWarp의 핵심 식) → GT 대비 PSNR.
팔(arm):
  torch v1(원본 그래프, 기준) / torch v2 / torch v3  — fp32, 그래프 차이만
  coreml v2(배포 Models/) / coreml v3(후보)         — fp16, CPU_AND_NE (앱 경로)
판정: coreml v3 − coreml v2 의 PSNR 차이(중앙값·최악)와 flow 차이(v3 vs v1, 모델 px).
사용: pixi run ... python gate_v3.py --v3 <dir> [--shorts 288,360,540] [--n 40]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, argparse, os, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# export_coreml_v3.py와 같은 warp를 v1/v2/v3 그래프에 주입 (수식은 원본 warplayer와 동일)
_WARP_CONSTS = {}
def warp(img, flow):
    _, _, H, W = flow.shape
    key = (H, W)
    if key not in _WARP_CONSTS:
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
        ng = torch.stack((2.0 * xx / max(W - 1, 1) - 1.0,
                          2.0 * yy / max(H - 1, 1) - 1.0), 0).unsqueeze(0)
        sc = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)],
                          dtype=torch.float32).view(1, 2, 1, 1)
        _WARP_CONSTS[key] = (ng, sc)
    ng, sc = _WARP_CONSTS[key]
    g = (flow * sc + ng).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = warp
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, 'v425/train_log')
from IFNet_HDv3_coreml import IFNet as IFNetV1
from IFNet_HDv3_coreml_v2 import IFNet as IFNetV2
from IFNet_HDv3_coreml_v3 import IFNet as IFNetV3

SCALES = [16, 8, 4, 2, 1]
ROOT = os.path.abspath("../..")

def pad64(n): return ((n + 63) // 64) * 64

def load_net(cls):
    net = cls()
    sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
    net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return net.eval()

def load_img(p, scale=1):
    im = Image.open(p).convert("RGB")
    if scale != 1:
        im = im.resize((im.width // scale, im.height // scale), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

def select_triplets(dirs, n, per_dir):
    """움직임 큰 순. select_triplets.py와 같은 기준(컷 배제 + 대칭) — 1/4 축소로 빠르게 판정."""
    cands = []
    for d in dirs:
        fs = sorted(glob.glob(os.path.join(d, "frame_*.png")))
        if len(fs) < 3: continue
        small = [load_img(f, 4) for f in fs]
        local = []
        for i in range(len(fs) - 2):
            a, g, b = small[i], small[i + 1], small[i + 2]
            if a.shape != b.shape or a.shape != g.shape: continue
            d_ab = float(np.mean(np.abs(a - b)))
            d_ag = float(np.mean(np.abs(a - g)))
            d_gb = float(np.mean(np.abs(g - b)))
            if d_ab < 0.02 or d_ab > 0.15: continue
            if abs(d_ag - d_gb) / (d_ag + d_gb + 1e-6) > 0.35: continue
            local.append((d_ab, d, i, fs[i], fs[i + 1], fs[i + 2]))
        local.sort(reverse=True)
        # 같은 시퀀스의 인접 삼중항은 거의 같은 장면이라 간격을 둔다
        picked = []
        for c in local:
            if all(abs(c[2] - p[2]) >= 6 for p in picked):
                picked.append(c)
            if len(picked) >= per_dir: break
        cands += picked
    cands.sort(reverse=True)
    return cands[:n]

def to_t(a): return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

def pack(a, b, mH, mW):
    # rifePack: uv=(gid+0.5)/size 에서 bilinear 샘플 = align_corners=False, 안티앨리어스 없음
    x = torch.cat((a, b), 1)
    return F.interpolate(x, size=(mH, mW), mode='bilinear', align_corners=False)

def synth(a, b, flow, mask):
    """rifeWarp 핵심식: uv=(gid+0.5)/size, f=bilinear(flow)*scale, uvA=uv+f/size, clamp_to_edge, m=sigmoid(bilinear(mask))."""
    _, _, H, W = a.shape
    mH, mW = flow.shape[-2:]
    fl = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    fl = fl * torch.tensor([W / mW, H / mH, W / mW, H / mH], dtype=fl.dtype).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
    def wp(img, f):
        gx = (xx + 0.5 + f[:, 0]) / W * 2 - 1
        gy = (yy + 0.5 + f[:, 1]) / H * 2 - 1
        return F.grid_sample(img, torch.stack((gx, gy), -1), mode='bilinear', padding_mode='border', align_corners=False)
    return wp(a, fl[:, :2]) * m + wp(b, fl[:, 2:4]) * (1 - m), fl

def psnr(x, y):
    mse = float(((x.clamp(0, 1) - y) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-10))

class TorchArm:
    def __init__(self, net): self.net = net
    def __call__(self, x):
        with torch.no_grad():
            fl, mk, _ = self.net(x, timestep=0.5, scale_list=SCALES)
        return fl[4], mk

class CoreMLArm:
    def __init__(self, path):
        import coremltools as ct
        self.m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    def __call__(self, x):
        t = np.full((1, 1, 1, 1), 0.5, np.float16)
        p = self.m.predict({"x": x.numpy().astype(np.float16), "t": t})
        return torch.from_numpy(p["flow"].astype(np.float32)), torch.from_numpy(p["mask"].astype(np.float32))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True, help="후보 v3 mlpackage 디렉터리")
    ap.add_argument("--v2", default=os.path.join(ROOT, "Models"), help="배포 v2 mlpackage 디렉터리")
    ap.add_argument("--shorts", default="360")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per-dir", type=int, default=4)
    ap.add_argument("--dirs", default="", help="쉼표 구분 추가 시퀀스 디렉터리 (frame_NNN.png)")
    ap.add_argument("--no-torch", action="store_true", help="torch v1/v2/v3 팔 생략 (coreml만)")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "bench_frames", "2026*")) if os.path.isdir(d))
    dirs += [d for d in a.dirs.split(",") if d]
    trips = select_triplets(dirs, a.n, a.per_dir)
    print(f"삼중항 {len(trips)}개 (시퀀스 {len(set(t[1] for t in trips))}개, d_ab {trips[-1][0]:.3f}~{trips[0][0]:.3f})")

    arms = {}
    if not a.no_torch:
        arms["t.v1"] = TorchArm(load_net(IFNetV1))
        arms["t.v2"] = TorchArm(load_net(IFNetV2))
        arms["t.v3"] = TorchArm(load_net(IFNetV3).permute_input_channels())

    for short in [int(s) for s in a.shorts.split(",")]:
        mH, mW = pad64(short), pad64(short * 16 // 9)
        arms_s = dict(arms)
        arms_s["c.v2"] = CoreMLArm(os.path.join(a.v2, f"rife{short}.mlpackage"))
        arms_s["c.v3"] = CoreMLArm(os.path.join(a.v3, f"rife{short}.mlpackage"))
        names = list(arms_s)
        rows = []
        for (d_ab, d, i, pa, pg, pb) in trips:
            A, G, B = to_t(load_img(pa)), to_t(load_img(pg)), to_t(load_img(pb))
            x = pack(A, B, mH, mW)
            ps, fls = {}, {}
            for k in names:
                f, m = arms_s[k](x)
                out, _ = synth(A, B, f, m)
                ps[k] = psnr(out, G)
                fls[k] = f
            ref = "t.v1" if "t.v1" in fls else "c.v2"
            _, flref = synth(A, B, fls[ref], torch.zeros_like(fls[ref][:, :1]))
            mag = float(flref.abs().amax(1).flatten().quantile(0.95))   # 원 해상도 px, 95퍼센타일
            fd = {k: (fls[k] - fls[ref]).abs() for k in names if k != ref}
            rows.append((os.path.basename(d), i, d_ab, mag, ps, {k: (float(v.mean()), float(v.flatten().quantile(0.99)), float(v.max())) for k, v in fd.items()}))
            print(f"  {os.path.basename(d)[:22]:22s} #{i:03d} d_ab={d_ab:.3f} |flow|p95={mag:6.1f}px  " +
                  " ".join(f"{k}={ps[k]:.2f}" for k in names), flush=True)

        print(f"\n── rife{short} ({mW}x{mH}) — 삼중항 {len(rows)}개")
        for k in names:
            v = [r[4][k] for r in rows]
            print(f"  {k:5s} PSNR 중앙값 {np.median(v):.3f}  평균 {np.mean(v):.3f}")
        def delta(k1, k0):
            dv = [r[4][k1] - r[4][k0] for r in rows]
            return np.median(dv), np.mean(dv), min(dv), max(dv)
        pairs = [("c.v3", "c.v2")]
        if "t.v1" in names: pairs += [("t.v2", "t.v1"), ("t.v3", "t.v1"), ("c.v2", "t.v2"), ("c.v3", "t.v3")]
        for k1, k0 in pairs:
            md, mn, lo, hi = delta(k1, k0)
            print(f"  Δ {k1}−{k0}: 중앙값 {md:+.3f}  평균 {mn:+.3f}  범위 {lo:+.3f}~{hi:+.3f} dB")
        big = [r for r in rows if r[3] >= 16]
        if big and "t.v1" in names:
            dv = [r[4]["t.v3"] - r[4]["t.v1"] for r in big]
            dc = [r[4]["c.v3"] - r[4]["c.v2"] for r in big]
            print(f"  |flow|p95>=16px 인 {len(big)}개: Δ t.v3−t.v1 중앙값 {np.median(dv):+.3f} (최악 {min(dv):+.3f}), "
                  f"Δ c.v3−c.v2 중앙값 {np.median(dc):+.3f} (최악 {min(dc):+.3f})")
        for k in [k for k in names if k != ("t.v1" if "t.v1" in names else "c.v2")]:
            mm = [r[5][k] for r in rows]
            print(f"  flow 차이 {k} vs 기준(모델 px): mean 중앙값 {np.median([m[0] for m in mm]):.3f}  "
                  f"p99 중앙값 {np.median([m[1] for m in mm]):.3f}  max 최악 {max(m[2] for m in mm):.2f}")
