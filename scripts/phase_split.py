#!/usr/bin/env python3
"""시각 구간별로 SCHED 창을 갈라 평균을 낸다. 인자: 이름=HH:MM:SS ... (마지막 구간은 로그 끝까지)"""
import re, sys
def sec(s):
    h,m,x=s.split(":"); return int(h)*3600+int(m)*60+int(x)
marks=[(a.split("=")[0], sec(a.split("=")[1])) for a in sys.argv[1:]]
rows=[]
for line in open("/tmp/MacFG_diag.log", encoding="utf-8", errors="replace"):
    m=re.match(r'^\[(\d\d):(\d\d):(\d\d)\.', line)
    if not m or '[SCHED]' not in line: continue
    t=int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))
    def g(p,c=float):
        mm=re.search(p,line); return c(mm.group(1)) if mm else None
    tick=g(r'tick=([\d.]+)Hz'); pres=g(r'present=(\d+)',int); drop=g(r'=(\d+)\)',int)
    gap=g(r'gap=(\d+)',int); e2e=g(r'e2e=(\d+)',int); mouse=g(r'mouse=(\d+)',int)
    interp=g(r'interpEnc=(\d+)',int)
    if None in (tick,pres,drop,gap,e2e) or tick<=0: continue
    span=240.0/tick
    rows.append((t,tick,(pres-drop)/span,pres/span,gap,e2e,mouse or 0,(interp or 0)/span))
print(f"{'구간':<14}{'창':>4}{'tick':>9}{'표시/s':>9}{'present/s':>11}{'갭':>7}{'e2e':>7}{'mouse':>8}{'보간/s':>8}")
for i,(name,start) in enumerate(marks):
    end = marks[i+1][1] if i+1 < len(marks) else 10**9
    sel=[r for r in rows if start<=r[0]<end]
    if not sel: print(f"{name:<14}  (창 없음)"); continue
    n=len(sel); f=lambda i2: sum(r[i2] for r in sel)/n
    print(f"{name:<14}{n:>4}{f(1):>9.1f}{f(2):>9.1f}{f(3):>11.1f}{f(4):>7.1f}{f(5):>7.1f}{f(6):>8.0f}{f(7):>8.1f}")
