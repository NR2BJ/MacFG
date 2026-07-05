"""M1 환산용: RIFE v4.25 flow(conv-only) ANE 비용을 저해상도 타깃에서 재측정 + int8 양자화 레버.
4K 워프는 GPU(별도), 병목은 ANE flow 추론. 이 M4 수치를 ANE TOPS비(M4 38 / M1 11 ≈ 3.5x,
실측 모델 스피드업은 보통 2.5~3x)로 환산해 M1 가늠.
사용: pixi ... python bench_m1est.py
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, types, time
import numpy as np, torch
import coremltools as ct

pkg = types.ModuleType('model'); pkg.__path__ = []
wl = types.ModuleType('model.warplayer'); wl.warp = lambda img, flow: img  # 벤치는 warp=identity
sys.modules['model'] = pkg; sys.modules['model.warplayer'] = wl
sys.path.insert(0, "../v425/train_log")
from IFNet_HDv3_coreml import IFNet

net = IFNet()
sd = torch.load("../v425/train_log/flownet.pkl", map_location='cpu', weights_only=True)
net.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
net.eval()

class W(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x, t):
        fl, m, _ = s.n(x, timestep=t, scale_list=[16, 8, 4, 2, 1]); return fl[4], m

def pad64(n): return ((n + 63) // 64) * 64

def make_ml(inW, inH, quant=False):
    pH, pW = pad64(inH), pad64(inW)
    x = torch.rand(1, 6, pH, pW); ts = torch.full((1, 1, 1, 1), 0.5)
    with torch.no_grad(): tr = torch.jit.trace(W(net).eval(), (x, ts))
    ml = ct.convert(tr,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32),
                ct.TensorType(name="t", shape=ts.shape, dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16)
    if quant:
        import coremltools.optimize.coreml as cto
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        ml = cto.linear_quantize_weights(ml, cfg)
    return ml, {"x": x.numpy(), "t": ts.numpy()}

def timeit(ml, inp, N=40):
    for _ in range(8): ml.predict(inp)
    t0 = time.perf_counter()
    for _ in range(N): ml.predict(inp)
    return (time.perf_counter() - t0) / N * 1000

print(f"=== RIFE v4.25 flow conv-only, ANE [{ct.__version__}] — 이 기기(M4) ===")
targets = [(512, 288, "288p"), (640, 360, "360p"), (768, 432, "432p")]
for inW, inH, lbl in targets:
    ml, inp = make_ml(inW, inH)
    ms = timeit(ml, inp)
    line = f"{lbl:5} ({inW}x{inH}): fp16 {ms:5.2f} ms"
    try:
        mlq, inpq = make_ml(inW, inH, quant=True)
        msq = timeit(mlq, inpq)
        line += f"  |  int8w {msq:5.2f} ms  ({(1-msq/ms)*100:+.0f}%)"
    except Exception as e:
        line += f"  |  int8 실패: {str(e)[:60]}"
    print(line)
print("\nM1 환산 = M4 × ~2.5–3배(ANE) + 워프 GPU ~5ms(M1). 예산: 60→120=16.7ms/쌍, 30→60=33ms/쌍")
