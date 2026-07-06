"""rife{N}.mlpackage의 연산별 디바이스 배치 프로파일 (MLComputePlan, coremltools 9).

어떤 op가 ANE에서 떨어져 GPU/CPU로 가는지 + 추정 비용을 집계 — ANE 효율 재수출의 타겟 특정.
사용: python profile_ane.py <model.mlpackage> [ALL|CPU_AND_NE|CPU_AND_GPU]
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys, collections
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.compute_device import MLNeuralEngineComputeDevice, MLGPUComputeDevice, MLCPUComputeDevice

path = sys.argv[1]
cu_name = sys.argv[2] if len(sys.argv) > 2 else "ALL"
cu = getattr(ct.ComputeUnit, cu_name)

# .mlpackage → 컴파일 필요
if path.endswith(".mlpackage"):
    compiled = ct.models.MLModel(path).get_compiled_model_path()
else:
    compiled = path
print(f"모델: {path}\n컴파일: {compiled}\ncompute_units: {cu_name}\n")

plan = MLComputePlan.load_from_path(path=compiled, compute_units=cu)
program = plan.model_structure.program
fn = program.functions["main"]

def device_name(dev):
    if isinstance(dev, MLNeuralEngineComputeDevice): return "ANE"
    if isinstance(dev, MLGPUComputeDevice): return "GPU"
    if isinstance(dev, MLCPUComputeDevice): return "CPU"
    return type(dev).__name__

# 집계: (op타입, 디바이스) → count, cost합
by_type_dev = collections.defaultdict(lambda: [0, 0.0])
by_dev_cost = collections.defaultdict(float)
seq = []   # 실행 순서의 디바이스 시퀀스 (경계 횟수 계산)
total_cost = 0.0
for op in fn.block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    cost = plan.get_estimated_cost_for_mlprogram_operation(op)
    if usage is None:
        continue
    dev = device_name(usage.preferred_compute_device)
    w = cost.weight if cost is not None else 0.0
    by_type_dev[(op.operator_name, dev)][0] += 1
    by_type_dev[(op.operator_name, dev)][1] += w
    by_dev_cost[dev] += w
    total_cost += w
    if op.operator_name not in ("const",):
        seq.append((op.operator_name, dev))

# 디바이스 경계(전환) 횟수 — 전환마다 텐서 복사 비용
transitions = sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i-1][1])

print(f"=== 디바이스별 추정 비용 비중 (총 {total_cost:.3f}) ===")
for dev, w in sorted(by_dev_cost.items(), key=lambda kv: -kv[1]):
    print(f"  {dev}: {w/max(total_cost,1e-9)*100:5.1f}%")
print(f"\n=== 디바이스 전환(경계) 횟수: {transitions} (전환마다 텐서 복사) ===")

print("\n=== op타입 × 디바이스 (비용 내림차순, 상위 25) ===")
rows = sorted(by_type_dev.items(), key=lambda kv: -kv[1][1])[:25]
for (name, dev), (cnt, w) in rows:
    print(f"  {name:28} {dev:4} ×{cnt:3}  cost={w/max(total_cost,1e-9)*100:5.1f}%")

print("\n=== ANE 밖으로 떨어진 op 전체 (타입별 개수) ===")
off = collections.Counter()
for (name, dev), (cnt, _) in by_type_dev.items():
    if dev != "ANE":
        off[f"{name}→{dev}"] += cnt
for k, v in off.most_common():
    print(f"  {k}: {v}")

# 전환 시퀀스 요약 (연속 같은 디바이스 뭉치기 — 파이프라인 모양)
print("\n=== 실행 시퀀스 (디바이스 뭉침, 앞 60세그먼트) ===")
segs = []
for name, dev in seq:
    if segs and segs[-1][0] == dev:
        segs[-1][1] += 1
    else:
        segs.append([dev, 1])
print("  " + " → ".join(f"{d}×{n}" for d, n in segs[:60]) + (" …" if len(segs) > 60 else ""))
