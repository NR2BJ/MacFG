#!/bin/bash
# 마지막 N개 SCHED 창을 초당으로 환산. 창은 정확히 240틱이므로 창길이 = 240/tickHz.
set -u
grep '\[SCHED\]' /tmp/MacFG_diag.log | tail -n "${1:-5}" | awk '{
  for(i=1;i<=NF;i++){
    if($i~/^tick=/){t=$i; gsub(/tick=|Hz/,"",t)}
    if($i~/^present=/){p=$i; gsub(/present=/,"",p)}
    if($i~/^미표시=/){d=$i; gsub(/미표시=|\)/,"",d)}
    if($i~/^gap=/){g=$i; gsub(/gap=|\(.*/,"",g)}
    if($i~/^e2e=/){e=$i; gsub(/e2e=|ms/,"",e)}
    if($i~/^interpEnc=/){ie=$i; gsub(/interpEnc=/,"",ie)}
  }
  span=240.0/t
  printf "  tick=%6.1fHz  gap=%3d  present=%5.1f/s  표시=%5.1f/s  효율=%3.0f%%  interp=%5.1f/s  e2e=%sms\n", \
    t, g, p/span, (p-d)/span, (p>0?100*(p-d)/p:0), ie/span, e
  sp+=p/span; sd+=(p-d)/span; st+=t; n++
} END { if(n>0) printf "  ── 평균: tick %.1fHz, present %.1f/s, 표시 %.1f/s (효율 %.0f%%)\n", st/n, sp/n, sd/n, 100*sd/sp }'
