"""A1 ② 학습 코퍼스 — 사용자 제공 오버워치 영상(worklog 2026-09-02)에서 7프레임 창(i-3..i+3)을 뽑는다.

한 창으로 두 케이던스를 만든다:
  30fps: P=i-2, A=i, GT=i+1(t=½), B=i+2        20fps: P=i-3, A=i, GT=i+1/i+2(t=⅓/⅔), B=i+3
**시간 분할(평가 누수 방지)**: 기존 도구가 쓰던 구간(seq_green 1261~1266s, seq_stairs 1785~1790s,
h2954 1794~1797s, ow_fhd 600s 부근)은 전부 평가 쪽으로 간다.
  만약…webm(AV1 1080p 59.94, 36.5분): 학습 [0,1200)s − [560,640)s / 평가 [1200, 끝)
  위.webm(VP9 4K 60, 10.6분):          학습 [0,180)∪[240,끝) / 평가 [180,240)
저장: 학습 = 모델 해상도 640x384로 **앱처럼 늘려 넣고 안티앨리어스 없는 bilinear**(rifePack과 같은 샘플링),
      평가 = 1080p 원본(배포 경로 PSNR용). JPEG q95. 디스크 여유가 작아(19GB) 창을 30프레임(0.5s)마다 하나만.
필터: 정지(|A−B| < 0.01) · 컷(인접 프레임 차 > 0.15 또는 비대칭) 창은 버린다.
사용: python extract_windows.py <out_root> [--every 30] [--eval-every 120]
"""
import os, sys, subprocess, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MOVIES = os.path.expanduser("~/Movies")
SOURCES = [
    # (파일, fps, 학습 구간들, 평가 구간들, 디코드 시 1080p로 줄이기)
    ("만약 이 영상이 레전드가 아니라면 오버워치 유튜버 은퇴 하겠습니다 ＊어그로X.webm", 19001 / 317,
     [(0, 560), (640, 1200)], [(1200, 1e9)], False),
    ("위.webm", 60.0, [(0, 180), (240, 1e9)], [(180, 240)], True),
]
# 도메인 밖(OOD) 점검용 — 오버워치가 아닌 60fps 녹화. 전부 평가(1080p)로만 쓴다.
OOD_SOURCES = [
    ("2025-01-10 18-45-28.mp4", 60.0, [], [(0, 1e9)], "pix1440"),   # 픽셀아트 게임 2560x1440 (HUD/채팅 위 캐릭터·눈 입자)
    ("26-04-02 11-05-43.mp4", 60.0, [], [(0, 1e9)], "desk4k"),      # 4K 데스크탑 녹화(브라우저 영상 + 주변 정지 UI)
]
MW, MH = 640, 384

def in_ranges(t, rs): return any(a <= t < b for a, b in rs)

def to_model(fr):
    x = torch.from_numpy(fr).permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(MH, MW), mode="bilinear", align_corners=False, antialias=False)
    return y[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy()

MIN_DAB = 0.01

def ok_window(win_small):
    """정지·컷 필터 — 1/4 축소 루마로 판정."""
    L = [f.astype(np.float32).mean(2) / 255.0 for f in win_small]
    d = [float(np.abs(L[k + 1] - L[k]).mean()) for k in range(len(L) - 1)]
    if max(d) > 0.15: return False, d
    dab = float(np.abs(L[5] - L[3]).mean())            # 30fps 쌍 A=i(3), B=i+2(5)
    if dab < MIN_DAB: return False, d
    # 컷은 한쪽만 급변 — 인접 차의 최대/중앙값 비로 거른다
    med = float(np.median(d))
    if med > 0 and max(d) / med > 4.0: return False, d
    return True, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=30, help="학습 창 간격(프레임)")
    ap.add_argument("--eval-every", type=int, default=120, help="평가 창 간격(프레임)")
    ap.add_argument("--ood", action="store_true", help="OOD 소스(오버워치 아님)만, 전부 평가 창으로")
    ap.add_argument("--min-dab", type=float, default=0.01, help="정지 판정 하한(화면 대부분이 정지한 녹화는 낮춘다)")
    a = ap.parse_args()
    global MIN_DAB
    MIN_DAB = a.min_dab
    root = os.path.abspath(a.out)
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "eval"), exist_ok=True)
    manifest = {"train": [], "eval": []}
    srcs = [(f, fps, tr, ev, dn, ("ow4k" if dn else "ow1080")) for f, fps, tr, ev, dn in SOURCES] if not a.ood else \
           [(f, fps, tr, ev, True, tag) for f, fps, tr, ev, tag in OOD_SOURCES]
    for fname, fps, tr, ev, down, tag in srcs:
        path = os.path.join(MOVIES, fname)
        vf = f"select='lt(mod(n\\,{a.every})\\,7)'"
        if down: vf += ",scale=1920:1080:flags=area"
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1920 * 1080 * 3 * 8)
        fsz = 1920 * 1080 * 3
        g = 0; kept = {"train": 0, "eval": 0}; seen = 0
        while True:
            win = []
            for _ in range(7):
                buf = p.stdout.read(fsz)
                if len(buf) < fsz: break
                win.append(np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3))
            if len(win) < 7: break
            n0 = g * a.every; g += 1; seen += 1
            tc = (n0 + 3) / fps
            split = "train" if in_ranges(tc, tr) else ("eval" if in_ranges(tc, ev) else None)
            if split is None: continue
            if split == "eval" and (n0 % a.eval_every) != 0: continue
            small = [f[::4, ::4] for f in win]
            ok, d = ok_window(small)
            if not ok: continue
            wid = f"{tag}_{n0:07d}"
            if split == "train":
                for k, f in enumerate(win):
                    Image.fromarray(to_model(f)).save(os.path.join(root, "train", f"{wid}_{k}.jpg"), quality=95)
            else:
                for k, f in enumerate(win):
                    Image.fromarray(f).save(os.path.join(root, "eval", f"{wid}_{k}.jpg"), quality=95)
            manifest[split].append({"id": wid, "t": round(tc, 3), "d": [round(x, 4) for x in d]})
            kept[split] += 1
            if seen % 200 == 0:
                print(f"  {tag} t={tc:7.1f}s 창 {seen} → 학습 {kept['train']} / 평가 {kept['eval']}", flush=True)
        p.wait()
        print(f"{tag}: 학습 {kept['train']} / 평가 {kept['eval']}", flush=True)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(f"완료: 학습 {len(manifest['train'])} / 평가 {len(manifest['eval'])} 창")

if __name__ == "__main__":
    main()
