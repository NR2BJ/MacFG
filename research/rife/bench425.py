import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 상대경로 보정
# RIFE v4.25 conv-only(ANE) 벤치 — warp는 Metal로 뺄 것이므로 identity 치환.
# timestep은 CoreML 입력 텐서로 유지 (arbitrary-t 보존 확인 겸).
import sys, types, time
import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

# model.warplayer 셔밍 — 벤치는 warp=identity (그래프의 conv 부분만 측정)
pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = lambda img, flow: img
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl

sys.path.insert(0, 'v425/train_log')
from IFNet_HDv3_coreml import IFNet

net = IFNet()
sd = torch.load('v425/train_log/flownet.pkl', map_location='cpu', weights_only=True)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
net.load_state_dict(sd, strict=False)
net.eval()

class Wrapped(torch.nn.Module):
    def __init__(self, net, scale_list):
        super().__init__()
        self.net = net
        self.scale_list = scale_list
    def forward(self, x, timestep):
        flow_list, mask, merged = self.net(x, timestep=timestep, scale_list=self.scale_list)
        return flow_list[4], mask  # 최종 flow + mask (warp/합성은 Metal에서)

def pad64(n): return ((n + 63) // 64) * 64

def bench(inW, inH, scale_list, label, units):
    pH, pW = pad64(inH), pad64(inW)
    w = Wrapped(net, scale_list).eval()
    x = torch.rand(1, 6, pH, pW)
    ts = torch.full((1, 1, 1, 1), 0.5)
    with torch.no_grad():
        traced = torch.jit.trace(w, (x, ts))
    try:
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
                    ct.TensorType(name="t", shape=ts.shape, dtype=np.float32)],
            compute_units=units,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        print(f"{label}: 변환 실패 {str(e)[:180]}")
        return None
    inp = {"x": x.numpy(), "t": ts.numpy()}
    for _ in range(6):
        ml.predict(inp)
    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        ml.predict(inp)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"{label} {inW}x{inH} scale={scale_list}: {dt:.2f} ms ({1000/dt:.0f} fps)  [conv-only]")
    return ml

units = {'all': ct.ComputeUnit.ALL, 'ane': ct.ComputeUnit.CPU_AND_NE,
         'gpu': ct.ComputeUnit.CPU_AND_GPU}[sys.argv[1] if len(sys.argv) > 1 else 'ane']
print(f"=== RIFE v4.25 conv-only [{units}] ===")
bench(960, 540, [16, 8, 4, 2, 1], '540p', units)
bench(1280, 720, [16, 8, 4, 2, 1], '720p', units)
bench(640, 360, [16, 8, 4, 2, 1], '360p', units)
# coarse 옵션: 위쪽 스케일만 (품질↓ 속도↑)
bench(960, 540, [32, 16, 8, 4, 2], '540p coarse', units)
print("참고: predict 고정비 ~2-3ms(540p)/3ms(720p) 포함. 8.3ms=120fps, 16.7ms=60fps")
