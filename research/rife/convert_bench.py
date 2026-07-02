import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # flownet.pkl 상대경로 보정
# RIFE IFNet → CoreML 변환 + ANE/GPU predict 벤치 (진짜 배포 성능 지표)
import torch, time, sys
import numpy as np
import coremltools as ct
from ifnet import IFNet, load_ifnet

net, _, _ = load_ifnet('flownet.pkl')
net.eval()

def pad64(n): return ((n + 63) // 64) * 64

class Wrapped(torch.nn.Module):
    """고정 scale_list로 trace 가능하게 감쌈. 입력 6ch(img0,img1) + timestep 1ch."""
    def __init__(self, net, scale_list):
        super().__init__()
        self.net = net
        self.scale_list = scale_list
    def forward(self, img0, img1, ts):
        merged, flow, sig = self.net(img0, img1, ts, scale_list=self.scale_list)
        return merged

def convert_and_bench(inW, inH, scale_list, label, units):
    pH, pW = pad64(inH), pad64(inW)
    wrapped = Wrapped(net, scale_list).eval()
    img0 = torch.rand(1, 3, pH, pW)
    img1 = torch.rand(1, 3, pH, pW)
    ts = torch.full((1, 1, pH, pW), 0.5)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (img0, img1, ts))
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="img0", shape=img0.shape, dtype=np.float32),
                    ct.TensorType(name="img1", shape=img1.shape, dtype=np.float32),
                    ct.TensorType(name="ts", shape=ts.shape, dtype=np.float32)],
            compute_units=units,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        print(f"{label} ({inW}x{inH}) 변환 실패: {str(e)[:200]}")
        return
    i0 = {"img0": img0.numpy(), "img1": img1.numpy(), "ts": ts.numpy()}
    for _ in range(5):
        mlmodel.predict(i0)
    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        mlmodel.predict(i0)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"{label} {inW}x{inH} scale={scale_list} [{units}]: {dt:.2f} ms/frame ({1000/dt:.0f} fps)")

unit = ct.ComputeUnit.ALL if len(sys.argv) < 2 else {
    'all': ct.ComputeUnit.ALL, 'ane': ct.ComputeUnit.CPU_AND_NE,
    'gpu': ct.ComputeUnit.CPU_AND_GPU}[sys.argv[1]]
print(f"=== CoreML predict 벤치 (units={unit}) ===")
convert_and_bench(1280, 720, (4, 2, 1), 'full 720p', unit)
convert_and_bench(1920, 1080, (8, 4, 2), 'coarse 1080p', unit)
convert_and_bench(960, 540, (4, 2, 1), '540p→up', unit)
print("8.3ms=120fps, 16.7ms=60fps 예산")
