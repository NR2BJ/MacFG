#!/bin/bash
# 로그에 쌓인 실행 구간 목록. 개발자 모드를 켠 뒤의 모든 앱 실행이 한 파일에 누적되므로,
# A/B는 "구간 N vs 구간 M"으로 비교한다 (재시작해도 앞 조건이 안 지워진다).
set -u
LOG=${1:-/tmp/MacFG_diag.log}
[ -f "$LOG" ] || { echo "로그 없음: $LOG (개발자 모드가 꺼져 있으면 파일이 지워진다)"; exit 1; }
grep -n "MacFG 실행 시작" "$LOG" | nl -w2 -s'  ' | while read -r idx rest; do
  line=${rest%%:*}
  start=$(sed -n "${line}p" "$LOG" | sed 's/.*시작 //; s/ (pid.*//')
  # 이 구간의 SCHED 창 수와 첫/마지막 시각
  next=$(grep -n "MacFG 실행 시작" "$LOG" | awk -F: -v l="$line" '$1>l{print $1; exit}')
  end=${next:-$(wc -l < "$LOG")}
  n=$(sed -n "${line},${end}p" "$LOG" | grep -c '\[SCHED\]')
  cfg=$(sed -n "${line},${end}p" "$LOG" | grep -m1 '\[AUTO\] capturing' | sed 's/.*capturing //' | cut -c1-46)
  printf "  #%-3s 줄%-7s SCHED %-5s %s\n       %s\n" "$idx" "$line" "$n" "$start" "${cfg:-(캡처 없음)}"
done
