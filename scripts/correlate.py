#!/usr/bin/env python3
"""한 번의 실행 안에서 창별 지표를 뽑아 상관을 낸다.

왜 '한 실행 안'인가: 조건을 바꿔가며 실행을 나누면 방송 콘텐츠가 흘러가서 조건 효과와
콘텐츠 효과가 섞인다(오늘 실제로 그 교란으로 잘못된 결론을 한 번 냈다). 같은 실행 안에서는
설정이 고정이므로, 창마다 달라지는 것은 콘텐츠/시스템 상태뿐이다.

주의: 다중바이트 문자를 정규식에 넣지 않는다(awk/grep에서 조용히 실패한 전력).
파이썬은 유니코드가 안전하지만 일관성을 위해 ASCII 키로만 매칭한다.
"""
import re, sys, math

def parse(path):
    sched, stage = [], []
    ts_re = re.compile(r'^\[(\d\d):(\d\d):(\d\d)\.(\d+)\]')
    for line in open(path, encoding='utf-8', errors='replace'):
        m = ts_re.match(line)
        if not m: continue
        h, mi, s, ms = (int(x) for x in m.groups()[:3]), None, None, None
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = hh*3600 + mm*60 + ss + int(m.group(4))/1000.0
        if '[SCHED]' in line:
            def g(pat, cast=float):
                mm2 = re.search(pat, line)
                return cast(mm2.group(1)) if mm2 else None
            tick = g(r'tick=([\d.]+)Hz')
            pres = g(r'present=(\d+)', int)
            drop = g(r'=(\d+)\)', int)          # 미표시=N) 의 N — ASCII만으로 잡는다
            gap  = g(r'gap=(\d+)', int)
            e2e  = g(r'e2e=(\d+)', int)
            ie   = g(r'interpEnc=(\d+)', int)
            stale= g(r'staleDrop:(\d+)', int)
            if None in (tick, pres, drop, gap, e2e) or tick <= 0: continue
            span = 240.0/tick
            sched.append(dict(t=t, tick=tick, gap=gap, e2e=e2e,
                              pres_s=pres/span, disp_s=(pres-drop)/span,
                              eff=(pres-drop)/pres if pres else 0,
                              interp_s=(ie or 0)/span, stale_s=(stale or 0)/span))
        elif '[STAGE]' in line:
            def g2(pat):
                mm2 = re.search(pat, line)
                return float(mm2.group(1)) if mm2 else None
            stage.append(dict(t=t, ci=g2(r'capIngest=([\d.]+)'), c1=g2(r'cb1gpu=([\d.]+)'),
                              c2=g2(r'cb2gpu=([\d.]+)'), pg=g2(r'present=([\d.]+)\(n='),
                              work=g2(r'work=([\d.]+)')))
    # SCHED 창마다 시간상 가장 가까운 STAGE를 붙인다
    for w in sched:
        if not stage: break
        near = min(stage, key=lambda s: abs(s['t']-w['t']))
        if abs(near['t']-w['t']) < 3.0:
            w.update({k: near[k] for k in ('ci','c1','c2','pg','work') if near[k] is not None})
    return [w for w in sched if 'ci' in w]

def pearson(xs, ys):
    n = len(xs)
    if n < 4: return float('nan')
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)); dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx and dy else float('nan')

w = parse(sys.argv[1] if len(sys.argv) > 1 else '/tmp/MacFG_diag.log')
if len(w) < 6:
    print(f"창이 {len(w)}개뿐 — 더 오래 돌려야 한다"); sys.exit(0)
print(f"창 {len(w)}개 ({w[-1]['t']-w[0]['t']:.0f}초)\n")
print(f"{'지표':<12}{'평균':>9}{'최소':>9}{'최대':>9}")
for k, lab in [('tick','tick Hz'),('disp_s','표시/s'),('pres_s','present/s'),('eff','효율'),
               ('ci','capIngest'),('c1','cb1 GPU'),('c2','cb2 GPU'),('pg','present GPU'),
               ('gap','vsync갭'),('e2e','e2e ms'),('stale_s','폐기/s')]:
    v = [x[k] for x in w if k in x]
    if v: print(f"{lab:<12}{sum(v)/len(v):>9.2f}{min(v):>9.2f}{max(v):>9.2f}")
print("\n표시량(표시/s)과의 상관:")
for k, lab in [('ci','capIngest'),('c2','cb2 GPU'),('c1','cb1 GPU'),('pg','present GPU'),
               ('pres_s','present/s'),('interp_s','보간생성/s'),('gap','vsync갭'),('work','work')]:
    xs = [x[k] for x in w if k in x]; ys = [x['disp_s'] for x in w if k in x]
    if len(xs) >= 6: print(f"  r(표시, {lab:<12}) = {pearson(xs,ys):+.3f}")
print("\nvsync갭과의 상관:")
for k, lab in [('ci','capIngest'),('c2','cb2 GPU'),('pres_s','present/s'),('work','work')]:
    xs = [x[k] for x in w if k in x]; ys = [x['gap'] for x in w if k in x]
    if len(xs) >= 6: print(f"  r(갭,   {lab:<12}) = {pearson(xs,ys):+.3f}")
