#!/bin/zsh
# Claude Code → Codex(GPT) 헤드리스 질의.
#
# 왜 필요한가: 컴퓨터 제어로 ChatGPT 데스크탑을 몰면 사용자의 화면·포커스를 뺏는다.
# codex 바이너리는 ChatGPT.app 안에 번들돼 있고 `exec` 서브커맨드가 비대화형이라,
# 백그라운드에서 같은 계정·같은 모델로 질의할 수 있다. 사용자는 자기 컴을 계속 쓴다.
#
# 사용:
#   scripts/ask-codex.sh "질문"                     # 이 저장소 기준
#   scripts/ask-codex.sh -C /other/repo "질문"
#   echo "긴 질문" | scripts/ask-codex.sh -         # stdin
#   scripts/ask-codex.sh -m gpt-5.6-sol "질문"
#   scripts/ask-codex.sh -e low "간단한 질문"        # low / medium / high
#
# 속도(실측 2026-08-28, 초점 잡힌 코드 질문): low 16초 / medium 20초 / high 26초.
# 셋 다 file:line 인용까지 정확했다. 코드베이스 전체를 훑어야 하는 넓은 질문은 2분+,
# 하위 질문 여러 개 + 웹 검색이 붙으면 5분+ 걸린다 — 그때만 백그라운드가 필요하다.
#
# 답변은 stdout + $OUT 파일(기본 /tmp/codex_answer.md) 양쪽으로 나온다.
# 샌드박스는 read-only 고정 — 상담이 목적이고, 코드 수정은 이쪽(Claude)에서 한다.
set -euo pipefail

CODEX="${CODEX_BIN:-/Applications/ChatGPT.app/Contents/Resources/codex}"
[[ -x "$CODEX" ]] || { echo "codex 바이너리 없음: $CODEX" >&2; exit 1; }

CWD="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=""
EFFORT="${CODEX_EFFORT:-medium}"
# **감시 타이머 (초).** 실측(2026-08-28): 정상은 5~26초. 드물게 안 끝난다(4분 38초, 180초).
# 결정적 관찰: 행이 났을 때도 **답변 파일은 이미 정확히 채워져 있었다.** 즉 추론이 느린 게
# 아니라 프로세스 종료가 막히는 것이다. 그래서 프로세스 종료를 기다리지 않고
# **답변 파일이 완성되면 바로 끝낸다.** (notify 훅 가설은 A/B로 반증: 9초 vs 8초로 차이 없음.
#  원인 미규명이지만 이 방식이면 원인과 무관하게 정상 속도가 나온다.)
# 이 맥에는 GNU `timeout`이 없어서 직접 감시한다. 넓은 조사 질문은 -t로 올려라.
LIMIT="${CODEX_TIMEOUT:-180}"
OUT="${CODEX_OUT:-/tmp/codex_answer.md}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -C|--cd)    CWD="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    -o|--out)   OUT="$2"; shift 2 ;;
    -e|--effort) EFFORT="$2"; shift 2 ;;
    -t|--timeout) LIMIT="$2"; shift 2 ;;
    *) break ;;
  esac
done

[[ $# -gt 0 ]] || { echo "질문이 없다" >&2; exit 1; }
PROMPT="$*"
[[ "$PROMPT" == "-" ]] && PROMPT="$(cat)"

args=(exec -C "$CWD" -s read-only --skip-git-repo-check -o "$OUT" -c model_reasoning_effort="$EFFORT")
[[ -n "$MODEL" ]] && args+=(-m "$MODEL")

rm -f "$OUT"
# **stdin을 반드시 닫는다.** codex exec는 "stdin이 파이프면 프롬프트에 덧붙인다"는 사양이라,
# 백그라운드로 띄우면서 파이프를 열어두면 **영영 입력을 기다린다.** 실측한 "행"(4분 38초,
# 180초, 364초×3)이 전부 이것이었다 — 직접 호출은 같은 질문에 5초였다.
# ask-claude.sh에서 같은 실수를 이미 한 번 고쳤는데(claude -p의 3초 stdin 경고) 여기선 빠뜨렸다.
"$CODEX" "${args[@]}" "$PROMPT" </dev/null >/dev/null 2>&1 &
CPID=$!

# 답변 파일이 **완성**되면 즉시 끝낸다. 완성 판정 = 비어 있지 않고 크기가 1초간 그대로.
# (파일은 한 번에 쓰이지만, 부분 쓰기를 읽고 나가는 것을 막는 안전장치다.)
# 주의: 이 루프는 `set -e` 아래서 돈다. `[[ 조건 ]] && break`처럼 쓰면 조건이 거짓일 때
# 종료코드 1이 되어 **셸이 스크립트를 조용히 죽인다**(실측: 답변 파일은 채워졌는데 출력이 빔).
# 그래서 모든 분기를 명시적 if/then으로 쓴다.
LAST=-1
DONE=0
for (( i=0; i<LIMIT*2; i++ )); do
  if [[ -s "$OUT" ]]; then
    SZ=$(wc -c < "$OUT")
    if [[ "$SZ" == "$LAST" ]]; then DONE=1; break; fi
    LAST="$SZ"
  fi
  if ! kill -0 $CPID 2>/dev/null; then DONE=1; break; fi   # 정상 종료 → 즉시 탈출
  sleep 1
done
# **출력을 먼저, 정리를 나중에.** codex를 백그라운드로 띄우면 같은 프로세스 그룹에 묶여서
# `kill $CPID`가 스크립트 자신까지 데려간다(실측: 추적이 kill에서 끊기고 cat에 도달 못 함,
# 답변 파일에는 정답이 들어 있었다). 답을 먼저 뱉으면 뒤가 어떻게 되든 호출자는 결과를 받는다.
if [[ ! -s "$OUT" ]]; then
  kill $CPID 2>/dev/null || true
  echo "codex 응답 없음 (${LIMIT}초 초과 또는 실패). 인증은 'codex login status'로 확인." >&2
  exit 1
fi
cat "$OUT"
kill $CPID 2>/dev/null || true          # 매달린 프로세스 회수 (정상 종료면 무해)
