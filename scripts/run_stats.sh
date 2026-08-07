#!/bin/bash
# 특정 실행 구간의 슬롯당 예산 + 결과. 사용: scripts/run_stats.sh <구간번호>
# 구간 번호는 scripts/runs.sh 로 확인한다.
set -u
LOG=/tmp/MacFG_diag.log
IDX=${1:-1}
python3 - "$LOG" "$IDX" <<'PY'
import re, sys, statistics
log, idx = sys.argv[1], int(sys.argv[2])
lines = open(log, encoding="utf-8", errors="replace").read().split("\n")
starts = [i for i, l in enumerate(lines) if "MacFG 실행 시작" in l]
if not starts or idx < 1 or idx > len(starts):
    print(f"구간 {idx} 없음 (총 {len(starts)}개)"); sys.exit(1)
a = starts[idx-1]; b = starts[idx] if idx < len(starts) else len(lines)
seg = lines[a:b]
print(f"=== 구간 #{idx}: {lines[a].strip()}")
for l in seg[:40]:
    if "[AUTO] capturing" in l or "[DISPLAY] 부착" in l: print("  " + l.strip()[:130])
slot=[]; sched=[]
for l in seg:
    m = re.search(r'슬롯당\(([\d.]+)ms\): cb1=([\d.]+) cb2=([\d.]+) present=([\d.]+) 합=([\d.]+)\((\d+)%\)', l)
    if m: slot.append(tuple(float(x) for x in m.groups()))
    if '[SCHED]' in l:
        def g(p,c=float):
            mm=re.search(p,l); return c(mm.group(1)) if mm else None
        t=g(r'tick=([\d.]+)Hz'); pr=g(r'present=(\d+)',int); dr=g(r'=(\d+)\)',int); e=g(r'e2e=(\d+)',int)
        if None not in (t,pr,dr,e) and t>1: sched.append((t,pr,dr,e,240.0/t))
act = [s for s in slot if s[2] > 0.5]
if act:
    md=lambda i: statistics.median([s[i] for s in act])
    print(f"  슬롯당(보간 활성 {len(act)}창): cb1={md(1):.2f} cb2={md(2):.2f} present={md(3):.2f} 합={md(4):.2f} ({md(5):.0f}%)")
if sched:
    md=lambda i: statistics.median([s[i] for s in sched])
    sp=statistics.median([s[4] for s in sched])
    print(f"  결과({len(sched)}창): tick={md(0):.1f}Hz  present={md(1)/sp:.1f}/s  표시={(md(1)-md(2))/sp:.1f}/s  e2e={md(3):.0f}ms")
PY
