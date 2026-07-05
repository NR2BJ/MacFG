"""CoreML(.mlpackage, fp16) 실프레임 엔드투엔드 패리티 — PyTorch 레퍼런스(pred_rife_low432)와
같은 파이프라인(직접 리사이즈→flow→업스케일→풀해상도 warp→sigmoid blend)으로 합성해
GT SSIM과 레퍼런스 대비 유사도를 확인. GPU/ANE 유닛별.
사용: python parity_coreml.py <triplets_dir>
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
import coremltools as ct

MODEL = "../../../Models/rife432.mlpackage"
LH, LW = 448, 768

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
def save(t, p):
    Image.fromarray((t.clamp(0,1)[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(p)
def psnr(x, y):
    mse = torch.mean((x - y) ** 2).item()
    return 99.0 if mse < 1e-8 else 10 * np.log10(1.0 / mse)

def real_warp(img, flow):
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing='ij')
    grid = torch.stack((xx, yy), 0).unsqueeze(0)
    vgrid = grid + flow
    vx = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vy = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    g = torch.cat((vx, vy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)

def compose(ml, a, b):
    H, W = a.shape[-2:]
    alo = F.interpolate(a, size=(LH, LW), mode='bilinear', align_corners=False)
    blo = F.interpolate(b, size=(LH, LW), mode='bilinear', align_corners=False)
    x = torch.cat((alo, blo), 1).numpy().astype(np.float16)
    t = np.full((1,1,1,1), 0.5, dtype=np.float16)
    out = ml.predict({"x": x, "t": t})
    flow = torch.from_numpy(out["flow"].astype(np.float32))
    mask = torch.from_numpy(out["mask"].astype(np.float32))
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    sx, sy = W / LW, H / LH
    flow = flow * torch.tensor([sx, sy, sx, sy]).view(1, 4, 1, 1)
    m = torch.sigmoid(F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False))
    return real_warp(a, flow[:, :2]) * m + real_warp(b, flow[:, 2:4]) * (1 - m)

tdir = sys.argv[1]
subs = sorted(d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d)))
for unit_name, cu in [("GPU", ct.ComputeUnit.CPU_AND_GPU), ("ANE", ct.ComputeUnit.CPU_AND_NE)]:
    ml = ct.models.MLModel(MODEL, compute_units=cu)
    d_ref, d_gt = [], []
    for name in subs:
        d = os.path.join(tdir, name)
        a, b, gt = load(f"{d}/a.png"), load(f"{d}/b.png"), load(f"{d}/gt.png")
        ref = load(f"{d}/pred_rife_low432.png")
        pred = compose(ml, a, b)
        if unit_name == "GPU": save(pred, f"{d}/pred_rife_cml432.png")
        d_ref.append(psnr(pred, ref)); d_gt.append(psnr(pred, gt))
    print(f"[{unit_name}] vs PyTorch레퍼런스: {np.mean(d_ref):.2f}dB (min {min(d_ref):.2f}) | vs GT: {np.mean(d_gt):.2f}dB")
print("게이트: vs레퍼런스 ≥35dB면 패리티 합격 (fp16 노이즈 수준)")
