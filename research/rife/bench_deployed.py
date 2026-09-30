"""배포된 Models/rife{N}.mlpackage의 predict 시간 (coremltools, ALL / CPU_AND_NE). v3 비교 기준선."""
import os, sys, time
import numpy as np
import coremltools as ct
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Models")
def pad64(n): return ((n + 63) // 64) * 64
for s in [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "288,360,432").split(",")]:
    path = os.path.join(root if len(sys.argv) < 3 else sys.argv[2], f"rife{s}.mlpackage")
    H, W = pad64(s), pad64(s * 16 // 9)
    x = np.random.rand(1, 6, H, W).astype(np.float16); t = np.full((1, 1, 1, 1), 0.5, np.float16)
    for label, cu in [("ALL", ct.ComputeUnit.ALL), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        m = ct.models.MLModel(path, compute_units=cu)
        inp = {"x": x, "t": t}
        for _ in range(10): m.predict(inp)
        N = 60; t0 = time.perf_counter()
        for _ in range(N): m.predict(inp)
        print(f"rife{s} {W}x{H} [{label}] {(time.perf_counter() - t0) / N * 1000:.2f} ms")
