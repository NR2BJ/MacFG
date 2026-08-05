#!/bin/bash
# 마지막 AB 마커 이후의 SCHED/STAGE 요약. 다중바이트 문자는 절대 정규식에 넣지 않는다
# (σ, ± 를 awk/grep 패턴에 쓰면 조용히 매칭 실패해 두 번이나 틀린 결론을 냈다).
set -u
awk '/^=== AB /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' /tmp/MacFG_diag.log > /tmp/ab_window.log
label=$(grep -o '=== AB .* ===' /tmp/ab_window.log | tail -1)
echo "── ${label:-(마커 없음)} ──"
grep '\[SCHED\]' /tmp/ab_window.log | tail -n "${1:-6}" | awk '{
  for (i=1;i<=NF;i++) {
    if ($i ~ /^tick=/) t=$i; if ($i ~ /^gap=/) g=$i; if ($i ~ /^interpEnc=/) ie=$i;
    if ($i ~ /^present=/) p=$i; if ($i ~ /^work=/) w=$i; if ($i ~ /^e2e=/) e=$i;
    if ($i ~ /^avg=/) ga=$i; if ($i ~ /^max=/) gm=$i;
  }
  print "  " t, g, ie, p, w, e, "glass" ga, gm
}'
echo "── STAGE ──"
grep '\[STAGE\]' /tmp/ab_window.log | tail -3 | sed 's/^\[[^]]*\] //'
echo "── GOV / SPIKE ──"
grep -c '\[GOV\] [0-9]' /tmp/ab_window.log | xargs -I{} echo "  GOV 전이 {}회"
grep '\[SPIKE\]' /tmp/ab_window.log | tail -3 | sed 's/^\[[^]]*\] //'
