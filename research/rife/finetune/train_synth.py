"""도메인 파인튜닝 — 합성 반투명 오버레이 증강. 학습가능성 probe가 양성(t15 +0.055)이라
데이터 부족만 남음. 해법: 게임 배경 삼중항 위에 '정지된 반투명 패널'(=채팅/HUD 실패모드)을
A/GT/B 동일하게 합성 → 배경은 실모션, 패널은 정지. 완벽한 GT로 '반투명 정지 오버레이는
안 끌린다'를 무한 학습. clean 삼중항과 5:5 혼합(일반 보간 망각·전면 프리즈 방지).
검증: 합성 없는 '진짜' 게임 채팅(t15,t03,…)에 전이되는지 = 진짜 성공 지표.

사용: python train_synth.py <dumps_dir> <val_triplets_dir> <short> <steps> <lr> <out.pkl>
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, time, argparse, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from train_finetune import (warp_manual, load_net, deploy_forward, lap_loss, ssim, RIFE, DEV)
from synth_overlay_v2 import apply_overlay as synth_overlay_v2

def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

def _fhash(p):
    im = Image.open(p).convert("RGB").resize((64, 36))
    return hash(np.asarray(im, dtype=np.uint8).tobytes())

def build_bg_pool(dumps_dir, exclude_hashes):
    """덤프 시퀀스에서 연속 (i,i+1,i+2) 삼중항 전부 → 배경 풀. 검증 프레임과 콘텐츠
    해시가 겹치는 삼중항은 제외 (진짜 채팅이 정지 타깃으로 학습되는 누수 차단)."""
    pool = []; skipped = 0
    for d in sorted(os.listdir(dumps_dir)):
        dd = os.path.join(dumps_dir, d)
        if not os.path.isdir(dd): continue
        frames = sorted(f for f in os.listdir(dd) if f.startswith("frame_") and f.endswith(".png"))
        if frames:   # 게임(1920x1080)만 — vidwin 테스트패턴 등 비게임 배경 제외
            if Image.open(f"{dd}/{frames[0]}").size != (1920, 1080): continue
        hs = {f: _fhash(f"{dd}/{f}") for f in frames}
        for i in range(len(frames) - 2):
            tri = frames[i:i+3]
            if any(hs[f] in exclude_hashes for f in tri):
                skipped += 1; continue
            pool.append(tuple(f"{dd}/{f}" for f in tri))
    if skipped: print(f"  (누수 제외 {skipped} 삼중항 — 검증 프레임 겹침)")
    return pool

def synth_overlay(a, gt, b, rng):
    """A/GT/B에 동일한 반투명 패널 합성 (정지). in-place 아님. [1,3,H,W]."""
    _, _, H, W = a.shape
    ov = torch.zeros(1, 3, H, W, device=a.device)     # 패널 색 (어두움)
    al = torch.zeros(1, 1, H, W, device=a.device)      # 알파
    n_panels = rng.randint(1, 3)
    for _ in range(n_panels):
        pw = rng.randint(int(W * 0.15), int(W * 0.4))
        ph = rng.randint(int(H * 0.08), int(H * 0.25))
        x0 = rng.randint(0, max(1, W - pw)); y0 = rng.randint(0, max(1, H - ph))
        base = rng.uniform(0.0, 0.15)                  # 어두운 반투명 배경
        alpha = rng.uniform(0.25, 0.6)
        al[:, :, y0:y0+ph, x0:x0+pw] = alpha
        ov[:, :, y0:y0+ph, x0:x0+pw] = base
        # 텍스트 유사 밝은 가로 세그먼트 여러 줄
        line_h = max(2, ph // rng.randint(5, 10))
        y = y0 + line_h
        while y < y0 + ph - line_h:
            seg_x = x0 + rng.randint(2, max(3, pw // 6))
            while seg_x < x0 + pw - 4:
                seg_w = rng.randint(4, max(5, pw // 4))
                bright = rng.uniform(0.6, 1.0)
                ov[:, :, y:y+max(1, line_h//2), seg_x:min(x0+pw, seg_x+seg_w)] = bright
                al[:, :, y:y+max(1, line_h//2), seg_x:min(x0+pw, seg_x+seg_w)] = rng.uniform(0.5, 0.9)
                seg_x += seg_w + rng.randint(3, 12)
            y += line_h
    def comp(img): return img * (1 - al) + ov * al
    return comp(a), comp(gt), comp(b)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps"); ap.add_argument("val"); ap.add_argument("short", type=int)
    ap.add_argument("steps", type=int); ap.add_argument("lr", type=float); ap.add_argument("out")
    ap.add_argument("--synth_p", type=float, default=0.5)
    ap.add_argument("--valnames", default="t15,t03,t02,t14,t08")
    a = ap.parse_args()
    rng = random.Random(0)

    valn = a.valnames.split(",")
    val = {n: tuple(load_img(f"{a.val}/{n}/{k}.png").to(DEV) for k in ("a", "gt", "b")) for n in valn}
    excl = set()
    for n in valn:
        for k in ("a", "gt", "b"): excl.add(_fhash(f"{a.val}/{n}/{k}.png"))
    bg = build_bg_pool(a.dumps, excl)
    print(f"배경 풀 {len(bg)} 삼중항 (덤프 {a.dumps})")

    net = load_net(f"{RIFE}/flownet.pkl").to(DEV)
    net.eval()
    def eval_val():
        with torch.no_grad():
            return {n: ssim(deploy_forward(net, val[n][0], val[n][2], a.short), val[n][1]) for n in valn}
    base = eval_val()
    print("진짜 채팅 기준선: " + " ".join(f"{n}={base[n]:.3f}" for n in valn))

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    net.train()
    t0 = time.time()
    cache = {}
    def get(paths):
        if paths not in cache:
            if len(cache) > 40: cache.clear()
            cache[paths] = tuple(load_img(p).to(DEV) for p in paths)
        return cache[paths]
    best = dict(base); best_step = 0
    for step in range(a.steps):
        aimg, gt, bimg = get(rng.choice(bg))
        if rng.random() < a.synth_p:
            aimg, gt, bimg = synth_overlay_v2(aimg, gt, bimg, rng)
        pred = deploy_forward(net, aimg, bimg, a.short)
        loss = F.l1_loss(pred, gt) + 0.5 * lap_loss(pred, gt, DEV)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, a.steps // 15) == 0 or step == a.steps - 1:
            net.eval(); cur = eval_val(); net.train()
            mean = np.mean([cur[n] for n in valn])
            if mean > np.mean([best[n] for n in valn]): best = dict(cur); best_step = step + 1
            dl = " ".join(f"{n}={cur[n]:.3f}(Δ{cur[n]-base[n]:+.3f})" for n in valn)
            print(f"step {step+1:4d} loss={loss.item():.3f} 진짜채팅평균={mean:.4f}  {dl}")
    torch.save(net.state_dict(), a.out)
    print(f"\n완료 {time.time()-t0:.0f}s → {a.out}")
    print(f"진짜 채팅 평균 SSIM: 기준선 {np.mean([base[n] for n in valn]):.4f} → 최종 {np.mean([eval_val()[n] for n in valn]):.4f} (best {np.mean([best[n] for n in valn]):.4f} @step{best_step})")
