"""시간축 화면정지 UI 검출 스파이크 (비-ML, 오프라인 검증).
가설: 화면고정 UI(채팅/HUD)는 수 프레임에 걸쳐 '같은 화면좌표에 지속되는 구조'다.
  - edge(p)   = 시간평균 고주파 에너지 (텍스트/구조는 큼, 평탄배경 작음)
  - tmotion(p)= 시간평균 |프레임차| (이동배경 큼, 정지UI 작음; 반투명은 배경비침으로 중간)
  static_ui = 구조 있음(edge↑) AND 시간적 이동 적음(tmotion↓) → 워프 대신 소스 프리즈.
검증: 12프레임 시퀀스의 내부 삼중항 (i,i+2)→i+1 RIFE 보간에 정지마스크로 소스 프리즈,
  GT(i+1) 대비 SSIM. 전체(무회귀) + 좌하단 ROI(채팅 프록시) 분리 측정.
사용: python temporal_ui_spike.py <dumps_dir> <short> [mlo mhi elo ehi]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, types
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "../quality_eval")
from model_compare import load_net, rife_low, ssim, warp  # 배포경로 재사용

DEV = torch.device("cpu")   # model_compare.warp가 CPU 그리드 생성 — 추론 전용이라 CPU로 통일

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

def gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

def gblur(x, k=11, s=3.0):
    c = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-(c ** 2) / (2 * s * s)); g /= g.sum()
    ker = (g[:, None] * g[None, :]).view(1, 1, k, k)
    return F.conv2d(F.pad(x, (k//2,)*4, mode='reflect'), ker)

def static_ui_mask(frames, mlo, mhi, elo, ehi):
    """frames: list [1,3,H,W]. → 정지UI 마스크 [1,1,H,W] 0..1 (화면좌표)."""
    g = [gray(f) for f in frames]
    hp = [gi - gblur(gi) for gi in g]
    edge = torch.stack([h.abs() for h in hp], 0).mean(0)                    # 시간평균 고주파
    tmot = torch.stack([(g[i] - g[i-1]).abs() for i in range(1, len(g))], 0).mean(0)  # 시간평균 이동
    edge = gblur(edge, 15, 4); tmot = gblur(tmot, 15, 4)                    # 공간 평활(코히런스)
    def ss(x, a, b): return ((x - a) / (b - a)).clamp(0, 1)
    m = ss(edge, elo, ehi) * (1 - ss(tmot, mlo, mhi))                       # 구조↑ AND 이동↓
    return gblur(m, 9, 3)                                                    # 마스크 부드럽게

def static_ui_mask_consistency(frames, clo, chi):
    """일관성 신호 — 흐린 정지 텍스트도 잡기 위해 '고주파의 시간적 일관성' 사용.
    hp_t의 시간평균 |mean| 대비 시간표준편차: 정지구조=일관(큼), 이동배경=비일관(작음)."""
    g = [gray(f) for f in frames]
    hp = torch.stack([gi - gblur(gi) for gi in g], 0)          # [T,1,1,H,W] (각 gi=[1,1,H,W])
    mean_hp = hp.mean(0).abs()                                  # [1,1,H,W] 시간평균 고주파 크기
    std_hp = hp.std(0)                                          # [1,1,H,W] 시간표준편차 (이동=큼)
    cons = mean_hp / (std_hp + 0.01)                            # 일관성 (정지구조=큼)
    cons = gblur(cons, 15, 4)
    def ss(x, a, b): return ((x - a) / (b - a)).clamp(0, 1)
    return gblur(ss(cons, clo, chi), 9, 3)

def roi_ssim(x, y, box):
    y0, y1, x0, x1 = box
    return ssim(x[:, :, y0:y1, x0:x1], y[:, :, y0:y1, x0:x1])

if __name__ == '__main__':
    dumps = sys.argv[1]; short = int(sys.argv[2])
    mlo, mhi, elo, ehi = (float(x) for x in (sys.argv[3:7] if len(sys.argv) >= 7 else (0.01, 0.05, 0.02, 0.08)))
    net = load_net("../v425/train_log/flownet.pkl").to(DEV).eval()
    seqs = []
    for d in sorted(os.listdir(dumps)):
        dd = os.path.join(dumps, d)
        if not os.path.isdir(dd): continue
        fr = sorted(f for f in os.listdir(dd) if f.startswith("frame_") and f.endswith(".png"))
        if len(fr) >= 5 and Image.open(f"{dd}/{fr[0]}").size == (1920, 1080):
            seqs.append([f"{dd}/{f}" for f in fr])
    print(f"게임 시퀀스 {len(seqs)}개 (params mlo{mlo} mhi{mhi} elo{elo} ehi{ehi})")

    b_full, f_full, b_roi, f_roi, cover = [], [], [], [], []
    dumped = False
    for paths in seqs:
        imgs = [load(p).to(DEV) for p in paths]
        H, W = imgs[0].shape[-2:]
        roi = (int(H*0.44), int(H*0.74), int(W*0.0), int(W*0.42))   # 좌중단 = 인게임 채팅(진짜 하드)
        M = static_ui_mask_consistency(imgs, mlo, mhi) if os.environ.get("CONS") else static_ui_mask(imgs, mlo, mhi, elo, ehi)
        for i in range(len(imgs) - 2):
            a, gt, b = imgs[i], imgs[i+1], imgs[i+2]
            with torch.no_grad():
                interp = rife_low(net, a, b, short)   # 배포경로 보간(baseline), MPS
            src = 0.5 * (a + b)                                            # 정지영역 A≈B≈GT
            frozen = interp * (1 - M) + src * M
            b_full.append(ssim(interp, gt)); f_full.append(ssim(frozen, gt))
            b_roi.append(roi_ssim(interp, gt, roi)); f_roi.append(roi_ssim(frozen, gt, roi))
            cover.append(M.mean().item())
        if not dumped:  # 마스크 시각화 1회
            SP = os.path.expanduser("/private/tmp/claude-501/-Users-nr2bj-Documents-MacFG/aab2d6a7-cb87-4570-a406-affc491ed109/scratchpad")
            vis = (M[0,0].cpu().numpy()*255).astype('uint8')
            Image.fromarray(vis).save(f"{SP}/ui_mask.png")
            over = imgs[0].clone(); over[:,0] += M[:,0]*0.5   # 빨강 오버레이
            Image.fromarray((over[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype('uint8')).save(f"{SP}/ui_mask_over.png")
            dumped = True
    import numpy as np
    print(f"마스크 커버리지 평균 {np.mean(cover)*100:.1f}%")
    print(f"전체  SSIM: baseline {np.mean(b_full):.4f} → +프리즈 {np.mean(f_full):.4f} (Δ{np.mean(f_full)-np.mean(b_full):+.4f})")
    print(f"채팅ROI SSIM: baseline {np.mean(b_roi):.4f} → +프리즈 {np.mean(f_roi):.4f} (Δ{np.mean(f_roi)-np.mean(b_roi):+.4f})")
