"""양자화 스윕 — v2 flow 모델(360/432)을 fp16 baseline + palettize{8,6,4}bit + int8-linear로
변형해 각각 ANE p50 속도 + 실배포 SSIM 측정. 목표: 432를 60fps 예산(~13ms)에 넣을 수 있나 /
품질 손실 없이 속도·용량 이득이 있나.

전제(정직): ANE는 fp16 네이티브라 가중치 양자화는 주로 대역폭/용량 이득 — 연산 속도는 크게 안
줄 수 있음. 실측으로 확인.
사용: python quant_sweep.py <src_dir(v2 mlpackage들)> <triplets_dir> <out_dir> [360 432]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, sys, time, shutil
import numpy as np
import coremltools as ct
from coremltools.optimize.coreml import (
    palettize_weights, linear_quantize_weights,
    OpPalettizerConfig, OpLinearQuantizerConfig, OptimizationConfig,
)
sys.path.insert(0, "quality_eval")
import coreml_deploy_eval as ev

def ane_p50(path, n=30, warm=6):
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    mH, mW = ev.model_input_size(m)
    x = np.random.rand(1, 6, mH, mW).astype(np.float16)
    t = np.full((1, 1, 1, 1), 0.5, dtype=np.float16)
    for _ in range(warm): m.predict({"x": x, "t": t})
    ts = []
    for _ in range(n):
        s = time.perf_counter(); m.predict({"x": x, "t": t}); ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    return ts[len(ts) // 2]

def variants(base_mlmodel):
    yield "fp16", base_mlmodel
    for nbits in (8, 6, 4):
        # uniform 모드 = sklearn(kmeans) 불요 · 결정적. 품질은 kmeans보다 소폭 낮으나 첫 스윕엔 충분.
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(nbits=nbits, mode="uniform"))
        yield f"pal{nbits}", palettize_weights(base_mlmodel, cfg)
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    yield "int8", linear_quantize_weights(base_mlmodel, cfg)

def sweep(src_dir, tdir, out_dir, shorts):
    os.makedirs(out_dir, exist_ok=True)
    budget60 = 16.7 * 0.8   # predict 예산 (워프/팩 별도 ~2ms 감안하면 실질 더 빡빡)
    for short in shorts:
        src = os.path.join(src_dir, f"rife{short}.mlpackage")
        base = ct.models.MLModel(src)
        print(f"\n==== rife{short} (v2 baseline) — 60fps predict 예산 ~{budget60:.1f}ms ====")
        print(f"{'variant':<8} {'ANE p50':>9} {'size MB':>8} {'SSIM':>7} {'고모션':>7} {'최악5':>7}")
        for tag, mdl in variants(base):
            path = os.path.join(out_dir, f"rife{short}_{tag}.mlpackage")
            if os.path.exists(path): shutil.rmtree(path)
            mdl.save(path)
            szmb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6
            p50 = ane_p50(path)
            q = ev.evaluate(path, tdir, verbose=False)
            flag = " ✅60fps" if p50 <= budget60 else ""
            print(f"{tag:<8} {p50:8.1f}ms {szmb:7.1f} {q['mean']:7.4f} {q['himotion']:7.4f} {q['worst5']:7.4f}{flag}")

if __name__ == '__main__':
    src_dir = sys.argv[1]; tdir = sys.argv[2]; out_dir = sys.argv[3]
    shorts = [int(s) for s in sys.argv[4:]] or [360, 432]
    sweep(src_dir, tdir, out_dir, shorts)
