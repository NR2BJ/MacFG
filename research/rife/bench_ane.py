import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # flownet.pkl 상대경로 보정
# warp(grid_sample) 제거한 순수 conv IFNet의 ANE 성능 — go/no-go 결정 지표.
# warp는 배포 시 Metal 셰이더로 뺄 것이므로, NN(conv)이 ANE에서 예산 내인지가 핵심.
import torch, time, sys
import numpy as np
import coremltools as ct
import ifnet as M
from ifnet import IFNet, load_ifnet

# warp를 identity로 몽키패치 (성능 측정 전용 — 정확도 무시, conv 그래프는 동일)
M.warp = lambda img, flow: img
import ifnet
ifnet.warp = M.warp

net, _, _ = load_ifnet('flownet.pkl')
net.eval()

def pad64(n): return ((n + 63) // 64) * 64

class Wrapped(torch.nn.Module):
    def __init__(self, net, scale_list):
        super().__init__()
        self.net = net; self.scale_list = scale_list
    def forward(self, img0, img1, ts):
        m, flow, sig = self.net(img0, img1, ts, scale_list=self.scale_list)
        return flow  # flow만 출력 (warp는 밖에서)

def bench(inW, inH, scale_list, label, units):
    pH, pW = pad64(inH), pad64(inW)
    w = Wrapped(net, scale_list).eval()
    a = torch.rand(1,3,pH,pW); b = torch.rand(1,3,pH,pW); ts = torch.full((1,1,pH,pW),0.5)
    with torch.no_grad():
        traced = torch.jit.trace(w, (a,b,ts))
    try:
        ml = ct.convert(traced,
            inputs=[ct.TensorType(name="i0",shape=a.shape,dtype=np.float32),
                    ct.TensorType(name="i1",shape=b.shape,dtype=np.float32),
                    ct.TensorType(name="ts",shape=ts.shape,dtype=np.float32)],
            compute_units=units, minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16)
    except Exception as e:
        print(f"{label}: 변환 실패 {str(e)[:150]}"); return
    inp = {"i0":a.numpy(),"i1":b.numpy(),"ts":ts.numpy()}
    for _ in range(8): ml.predict(inp)
    N=40; t0=time.perf_counter()
    for _ in range(N): ml.predict(inp)
    dt=(time.perf_counter()-t0)/N*1000
    print(f"{label} {inW}x{inH} scale={scale_list}: {dt:.2f} ms ({1000/dt:.0f} fps)  [conv-only, warp 제외]")

unit = {'all':ct.ComputeUnit.ALL,'ane':ct.ComputeUnit.CPU_AND_NE,'gpu':ct.ComputeUnit.CPU_AND_GPU}[sys.argv[1] if len(sys.argv)>1 else 'all']
print(f"=== 순수 conv IFNet (warp 제거) [{unit}] ===")
bench(1280,720,(4,2,1),'720p',unit)
bench(1920,1080,(8,4,2),'1080p coarse',unit)
bench(960,540,(4,2,1),'540p',unit)
bench(3840,2160,(16,8,4),'4K coarse',unit)
print("8.3ms=120fps, 16.7ms=60fps")
