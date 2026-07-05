"""고모션 삼중항의 움직임 영역을 크롭해 GT vs 엔진별 예측을 나란히 비교.
사용: python montage.py <triplets_dir> <t1> <t2> <out.png>
"""
import sys, numpy as np
from PIL import Image, ImageDraw

tdir, names, outp = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
COLS = [("gt", "GT"), ("blend", "Blend"), ("metalflow", "MetalFlow"),
        ("applefi", "AppleFI"), ("rife", "RIFE Swift 432"), ("rife_full", "RIFE-full")]
CROP, TILE, LBL = 360, 320, 26

def load(p): return np.asarray(Image.open(p).convert("RGB"))

rows = []
for nm in names:
    d = f"{tdir}/{nm}"
    a, b = load(f"{d}/a.png").astype(np.float32), load(f"{d}/b.png").astype(np.float32)
    H, W = a.shape[:2]
    # 움직임 최대 영역 탐색 (블록 다운샘플 argmax)
    diff = np.abs(a - b).sum(2)
    bs = 60
    dh, dw = H // bs, W // bs
    small = diff[:dh*bs, :dw*bs].reshape(dh, bs, dw, bs).mean((1, 3))
    my, mx = np.unravel_index(small.argmax(), small.shape)
    cy, cx = my * bs + bs // 2, mx * bs + bs // 2
    y0 = min(max(cy - CROP // 2, 0), H - CROP); x0 = min(max(cx - CROP // 2, 0), W - CROP)
    tiles = []
    for key, _ in COLS:
        img = Image.open(f"{d}/pred_{key}.png") if key != "gt" else Image.open(f"{d}/gt.png")
        crop = img.convert("RGB").crop((x0, y0, x0 + CROP, y0 + CROP)).resize((TILE, TILE), Image.NEAREST)
        tiles.append(crop)
    rows.append(tiles)

n = len(COLS)
W_out = n * TILE
H_out = LBL + len(rows) * (TILE + LBL)
canvas = Image.new("RGB", (W_out, H_out), (24, 24, 28))
draw = ImageDraw.Draw(canvas)
for i, (_, label) in enumerate(COLS):
    draw.text((i * TILE + 6, 4), label, fill=(230, 230, 235))
y = LBL
for ri, tiles in enumerate(rows):
    draw.text((6, y - 2), f"{names[ri]}", fill=(255, 210, 120))
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * TILE, y))
    y += TILE + LBL
canvas.save(outp)
print(f"✅ {outp}  ({W_out}x{H_out})")
