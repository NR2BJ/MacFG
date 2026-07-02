import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # flownet.pkl 상대경로 보정
# RIFE IFNet 성능 벤치 (MPS/GPU) — go/no-go 1차 지표
import torch, time
from ifnet import load_ifnet

net, _, _ = load_ifnet('flownet.pkl')
net.eval()
dev = torch.device('mps')
net = net.to(dev)

def pad64(n): return ((n + 63) // 64) * 64  # 32의 배수 + 스케일 여유

def bench(inW, inH, scale_list, iters=40, label=''):
    pH, pW = pad64(inH), pad64(inW)
    img0 = torch.rand(1, 3, pH, pW, device=dev)
    img1 = torch.rand(1, 3, pH, pW, device=dev)
    ts = torch.full((1, 1, pH, pW), 0.5, device=dev)
    with torch.no_grad():
        for _ in range(8):  # 워밍업
            net(img0, img1, ts, scale_list=scale_list)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(img0, img1, ts, scale_list=scale_list)
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1000
    print(f"{label} {inW}x{inH} scale={scale_list}: {dt:.2f} ms/frame  ({1000/dt:.0f} fps 상한)")

# 입력 해상도별. 4K는 저해상도 flow(scale 큼)로 계산하는 게 실전 (LS 방식)
bench(1280, 720, (4, 2, 1), label='720p full')
bench(1920, 1080, (4, 2, 1), label='1080p full')
bench(1920, 1080, (8, 4, 2), label='1080p coarse')
bench(3840, 2160, (8, 4, 2), label='4K full')
bench(960, 540, (4, 2, 1), label='540p(→4K flow up)')
print(f"\nMPS 백엔드 GPU 상한 (배포는 CoreML/ANE). 8.3ms=120fps 예산, 16.7ms=60fps 예산")
