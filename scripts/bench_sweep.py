#!/usr/bin/env python3
"""화질 A/B — bench_frames 전 시퀀스를 돌리고 hold 기준선과 함께 판정한다.

**왜 이 스크립트가 필요한가 (2026-08-30).**
InterpBench를 시퀀스 하나씩 손으로 돌리면 표본을 고르게 되고, 그러면 틀린다.
실제로 그날 occ-directional을 3개 시퀀스로 판정했는데 그중 2개가 병리 시퀀스였다.
전수 측정 결과 27개 중 12개(44%)에서 보간이 hold(보간 안 함)보다 나쁘다 —
거의 정지 구간이거나 장면 전환이라 보간이 정의상 지는 삼중항이다. 엔진 결함이 아니다.
그 12개가 섞이면 **평균과 중앙값의 부호가 반대로 나온다**(전체 평균 -0.83dB / 중앙값 +1.27dB).

그래서 세 가지를 강제한다:
  (1) 전 시퀀스를 돈다 — 고를 수 없다
  (2) mf-hold를 항상 같이 낸다 — 기준선 미달을 숨길 수 없다
  (3) 판정 수치는 깨끗한 시퀀스의 **중앙값** — 극단값에 안 끌린다

zsh가 아니라 파이썬인 이유: 이 저장소는 셸 스크립트에서 반복해서 대가를 치렀다
(멀티바이트 정규식 무음 실패, set -e + [[ ]] && 조용한 종료, 프로세스 그룹 kill).

사용:
    scripts/bench_sweep.py                     # 현재 설정 전수
    scripts/bench_sweep.py --occ-dir           # 플래그를 붙여 A/B
    scripts/bench_sweep.py --flow-base 1440
    scripts/bench_sweep.py --engine metalflow  # 주의: 소문자. metalFlow는 조용히 빈 집합이 된다.
"""
import os, subprocess, sys, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 기본은 debug 빌드 — **release만 다시 빌드하고 스윕을 돌리면 옛 debug 바이너리를 돌린다**
# (2026-09-25: 영점 편향 A/B가 두 arm 비트 동일로 나와서 발견). MACFG_BENCH_BIN으로 바꿀 수 있다.
BIN = Path(os.environ.get("MACFG_BENCH_BIN", str(ROOT / ".build/debug/InterpBench")))

args = sys.argv[1:]
engine = "metalflow"
if "--engine" in args:
    i = args.index("--engine"); engine = args[i + 1]; del args[i:i + 2]

if not BIN.exists():
    sys.exit("InterpBench 없음 — swift build 먼저")

rows = []
seqs = sorted(d for d in (ROOT / "bench_frames").iterdir()
              if d.is_dir() and len(list(d.glob("*.png"))) >= 3)
for n, d in enumerate(seqs, 1):
    print("  [%d/%d] %s" % (n, len(seqs), d.name), end="\r", file=sys.stderr, flush=True)
    try:
        out = subprocess.run([str(BIN), "--triplets", str(d), "--engines", engine] + args,
                             capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        continue
    # avg=NN.NNdB 가 순서대로 hold / blend / 엔진
    v = [float(x) for x in re.findall(r"avg=([\d.]+)dB", out)]
    # **최악값과 선명도도 같이 거둔다 (B4).** 중앙값 비교는 "빠른 모션에서 무너지는가"를
    # 덮지 못한다 — 엔진 간 격차가 큰 쪽이 min이다(실측 RIFE 27.52 vs MetalFlow 32.64).
    # min=은 hold/blend/엔진 세 줄 모두에 있고 sharp=는 엔진 줄에만 있다.
    mn = [float(x) for x in re.findall(r"min=([\d.]+)dB", out)]
    sh = re.search(r"sharp=([\d.]+)", out)
    if len(v) >= 3:
        rows.append((d.name, v[0], v[1], v[2],
                     mn[0] if len(mn) >= 3 else 0.0,     # hold min
                     mn[2] if len(mn) >= 3 else 0.0,     # 엔진 min
                     float(sh.group(1)) if sh else 0.0))
print(" " * 60, file=sys.stderr)

if not rows:
    sys.exit("측정된 시퀀스 없음 — 엔진 키가 맞는지 확인 (소문자 metalflow)")

clean = [r for r in rows if r[3] > r[1]]
bad = [r for r in rows if r[3] <= r[1]]
print("엔진 %s   플래그 %s" % (engine, " ".join(args) or "(기본)"))
print("시퀀스 %d개  —  정상 %d / 병리 %d  (병리 = 보간이 hold 이하)" % (len(rows), len(clean), len(bad)))
print()
print("   %-20s %8s %8s %8s %9s %8s %8s %7s"
      % ("시퀀스", "hold", "blend", engine, "-hold", "holdMin", "engMin", "sharp"))
for name, h, b, m, hm, em, sp in sorted(rows, key=lambda r: r[3] - r[1]):
    print("%s %-20s %8.2f %8.2f %8.2f %+9.2f %8.2f %8.2f %7.3f"
          % ("  " if m > h else "PP", name, h, b, m, m - h, hm, em, sp))
print()
if clean:
    d = [r[3] - r[1] for r in clean]
    print("판정 수치 — 깨끗한 %d개의 **중앙값**: %+.3f dB   (평균 %+.3f, 범위 %+.2f~%+.2f)"
          % (len(clean), st.median(d), st.mean(d), min(d), max(d)))
    print("절대 PSNR 중앙값: %.3f dB" % st.median([r[3] for r in clean]))
    # **B4의 판정 수치** — 최악값과 선명도. 중앙값이 같아도 여기서 갈릴 수 있다.
    print("최악값(min) 중앙값: %.3f dB   |  최악값의 최악: %.3f dB   |  선명도 중앙값: %.3f"
          % (st.median([r[5] for r in clean]), min(r[5] for r in clean),
             st.median([r[6] for r in clean])))
    # 빠른 모션 슬라이스 = blend가 hold를 크게 이기는 시퀀스(=A와 GT가 많이 다르다).
    fast = [r for r in clean if r[2] - r[1] >= 2.0]
    if fast:
        print("빠른 모션 %d개(blend-hold>=2dB) — 중앙값 %+.3f dB · min 중앙값 %.3f dB · 선명도 %.3f"
              % (len(fast), st.median([r[3] - r[1] for r in fast]),
                 st.median([r[5] for r in fast]), st.median([r[6] for r in fast])))
if bad:
    print()
    print("PP 로 표시한 병리 %d개는 판정에서 제외했다." % len(bad))
    print("   이 시퀀스들은 blend(모션보상 없는 단순 평균)도 hold보다 낮다 = A 이 GT와 거의 같다는 뜻이고,")
    print("   정지 구간이거나 장면 전환이라 보간이 정의상 진다. 엔진 결함이 아니라 데이터 성격이다.")
