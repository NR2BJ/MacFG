"""임의 CoreML flow 모델(.mlpackage)의 실배포 화질 — 앱과 동일 경로로 SSIM 측정.
  CoreML(고정 크기)로 flow+mask 예측 → 풀해상도 업스케일·벡터스케일 → torch warp+blend → GT.
모델 입력 x:(1,6,mH,mW) fp16, t:(1,1,1,1). 출력 flow:(1,4,mH,mW), mask:(1,1,mH,mW) pre-sigmoid.
사용: python coreml_deploy_eval.py <model.mlpackage> <triplets_dir>   → mean/고모션/최악5 SSIM
공정성: 양자화·모델 변형 비교는 전부 이 하네스로 (절대값은 torch 경로와 소폭 다르나 상대비교 정확).
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys
import numpy as np
import torch, torch.nn.functional as F
import coremltools as ct
from PIL import Image

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

def warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vg = grid + flow
    vx = 2.0 * vg[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vg[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

def _gauss(ks=11, sig=1.5):
    c = torch.arange(ks) - ks // 2
    g = torch.exp(-(c ** 2) / (2 * sig ** 2)); g /= g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, ks, ks)
_W = _gauss()

def ssim(x, y):
    lx = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    ly = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mx = F.conv2d(lx, _W, padding=5); my = F.conv2d(ly, _W, padding=5)
    vx = F.conv2d(lx * lx, _W, padding=5) - mx * mx
    vy = F.conv2d(ly * ly, _W, padding=5) - my * my
    cxy = F.conv2d(lx * ly, _W, padding=5) - mx * my
    return (((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))).mean().item()

def model_input_size(mlmodel):
    for inp in mlmodel.get_spec().description.input:
        if inp.name == "x":
            sh = [int(d) for d in inp.type.multiArrayType.shape]
            return sh[2], sh[3]   # mH, mW
    raise RuntimeError("x 입력 없음")

def deploy_pred(m, mH, mW, a, b):
    H, W = a.shape[-2:]
    alo = F.interpolate(a, size=(mH, mW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(mH, mW), mode='bilinear', align_corners=False)
    x = torch.cat((alo, blo), 1).numpy().astype(np.float16)
    t = np.full((1, 1, 1, 1), 0.5, dtype=np.float16)
    out = m.predict({"x": x, "t": t})
    flow = torch.from_numpy(out["flow"].astype(np.float32))
    mask = torch.from_numpy(out["mask"].astype(np.float32))
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / mW, H / mH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    sig = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return warp(a, flow[:, :2]) * sig + warp(b, flow[:, 2:4]) * (1 - sig)

def evaluate(path, tdir, verbose=True):
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    mH, mW = model_input_size(m)
    subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
    rows = []
    for name in subs:
        d = os.path.join(tdir, name)
        a, b, gt = load(f"{d}/a.png"), load(f"{d}/b.png"), load(f"{d}/gt.png")
        mo = torch.mean(torch.abs(a - b)).item()
        pred = deploy_pred(m, mH, mW, a, b)
        rows.append((name, ssim(pred, gt), mo))
    med = float(np.median([r[2] for r in rows]))
    alls = [r[1] for r in rows]
    hi = [r[1] for r in rows if r[2] >= med]
    worst = sorted(rows, key=lambda r: r[1])[:5]
    if verbose:
        print(f"모델 {os.path.basename(path)}  입력 {mW}x{mH}")
        print(f"  SSIM mean={np.mean(alls):.4f}  고모션={np.mean(hi):.4f}  최악5={np.mean([w[1] for w in worst]):.4f}")
        print(f"  최악: " + " ".join(f"{w[0]}={w[1]:.3f}" for w in worst))
    return dict(mean=float(np.mean(alls)), himotion=float(np.mean(hi)),
                worst5=float(np.mean([w[1] for w in worst])), worst=worst)

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
