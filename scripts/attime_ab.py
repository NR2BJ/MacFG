#!/usr/bin/env python3
"""atTime 교대 A/B 집계. 창(240틱)마다 모드가 뒤집히므로 콘텐츠 드리프트는 양쪽에 동일하게 실린다."""
import re, sys, statistics
rows = {0: [], 1: []}
for line in open("/tmp/MacFG_diag.log", encoding="utf-8", errors="replace"):
    if '[SCHED]' not in line: continue
    def g(p, c=float):
        m = re.search(p, line); return c(m.group(1)) if m else None
    at = g(r'atTime=([01])', int)
    tick = g(r'tick=([\d.]+)Hz'); pres = g(r'present=(\d+)', int); drop = g(r'=(\d+)\)', int)
    gap = g(r'gap=(\d+)', int); e2e = g(r'e2e=(\d+)', int); dup = g(r'dupSlot=(\d+)', int)
    sl = re.search(r'slip=(\d+)/(\d+)/(\d+)/(\d+)', line)
    if None in (at, tick, pres, drop, gap, e2e) or tick <= 0: continue
    span = 240.0 / tick
    slip = [int(x) for x in sl.groups()] if sl else [0,0,0,0]
    rows[at].append(dict(tick=tick, pres=pres/span, disp=(pres-drop)/span,
                         eff=(pres-drop)/pres if pres else 0, gap=gap, e2e=e2e,
                         dup=dup or 0, slip=slip))
if not rows[0] or not rows[1]:
    print(f"창 부족 (atTime=0:{len(rows[0])} 1:{len(rows[1])}) — 더 돌려야 한다"); sys.exit(0)
def med(v): return statistics.median(v) if v else 0
print(f"{'모드':<12}{'창':>4}{'tick':>8}{'present/s':>11}{'표시/s':>9}{'효율':>8}{'갭':>6}{'e2e':>6}{'dupSlot':>9}")
for at in (0, 1):
    r = rows[at]
    name = "atTime 고정" if at else "plain(현재)"
    print(f"{name:<12}{len(r):>4}{med([x['tick'] for x in r]):>8.1f}{med([x['pres'] for x in r]):>11.1f}"
          f"{med([x['disp'] for x in r]):>9.1f}{med([x['eff'] for x in r]):>8.2f}"
          f"{med([x['gap'] for x in r]):>6.0f}{med([x['e2e'] for x in r]):>6.0f}{med([x['dup'] for x in r]):>9.0f}")
print("\nslip 분포 (표시가 목표 슬롯에서 몇 칸 밀렸나: 0 / 1 / 2 / 3+):")
for at in (0, 1):
    tot = [sum(x['slip'][i] for x in rows[at]) for i in range(4)]
    n = sum(tot) or 1
    name = "atTime 고정" if at else "plain(현재)"
    print(f"  {name:<12} " + "  ".join(f"{t*100/n:5.1f}%" for t in tot) + f"   (n={n})")
