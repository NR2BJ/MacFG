"""사실적 채팅/HUD 오버레이 합성 — 실제 폰트 텍스트 + 반투명 어두운 패널. crude 사각형
버전이 진짜 채팅에 전이 안 돼서(외형 격차) 실채팅 외형에 근접시킨다:
어두운 반투명 패널(코너 편향) + 소형 밝은 텍스트 여러 줄(유저명+메시지 유사).
A/GT/B 동일 합성(정지) → 배경 실모션 + 오버레이 정지 = 채팅 실패모드.
"""
import random, string
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
]
def _rand_text(rng, n):
    words = []
    for _ in range(rng.randint(1, 5)):
        L = rng.randint(2, 9)
        words.append("".join(rng.choice(string.ascii_letters + "0123456789가나다라마바사아") for _ in range(L)))
    return " ".join(words)[:n]

def make_overlay(H, W, rng):
    """RGBA numpy [H,W,4] float(0..1) — 정지 오버레이 (배경 위 알파합성)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    n_panels = rng.randint(1, 2)
    for _ in range(n_panels):
        pw = rng.randint(int(W * 0.16), int(W * 0.42))
        ph = rng.randint(int(H * 0.10), int(H * 0.30))
        # 코너 편향 (채팅/HUD 위치): 60% 좌하단/모서리
        if rng.random() < 0.6:
            x0 = rng.choice([rng.randint(0, int(W * 0.08)), W - pw - rng.randint(0, int(W * 0.08))])
            y0 = rng.choice([rng.randint(0, int(H * 0.1)), H - ph - rng.randint(0, int(H * 0.1))])
        else:
            x0 = rng.randint(0, max(1, W - pw)); y0 = rng.randint(0, max(1, H - ph))
        # 어두운 반투명 패널
        pa = rng.randint(60, 150)
        dr.rectangle([x0, y0, x0 + pw, y0 + ph], fill=(rng.randint(0, 25), rng.randint(0, 25), rng.randint(0, 30), pa))
        # 텍스트 여러 줄
        try:
            fs = rng.randint(max(9, ph // 12), max(11, ph // 7))
            font = ImageFont.truetype(rng.choice(_FONTS), fs)
        except Exception:
            font = ImageFont.load_default()
        y = y0 + rng.randint(2, 6)
        while y < y0 + ph - fs:
            tcol = rng.choice([(230, 230, 230), (200, 220, 255), (255, 230, 180), (180, 255, 180)])
            ta = rng.randint(180, 255)
            dr.text((x0 + rng.randint(3, 10), y), _rand_text(rng, pw // (fs // 2 + 1)), font=font, fill=tcol + (ta,))
            y += fs + rng.randint(1, 5)
    return np.asarray(img, dtype=np.float32) / 255.0   # [H,W,4]

def apply_overlay(a, gt, b, rng):
    """a/gt/b [1,3,H,W] torch에 동일 오버레이 알파합성. 같은 device 반환."""
    _, _, H, W = a.shape
    ov = make_overlay(H, W, rng)
    dev = a.device
    rgb = torch.from_numpy(ov[..., :3]).permute(2, 0, 1).unsqueeze(0).to(dev)
    al = torch.from_numpy(ov[..., 3:4]).permute(2, 0, 1).unsqueeze(0).to(dev)
    def comp(img): return img * (1 - al) + rgb * al
    return comp(a), comp(gt), comp(b)
