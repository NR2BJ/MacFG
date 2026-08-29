#!/bin/zsh
# Codex(GPT) → Claude 헤드리스 질의. ask-codex.sh의 반대 방향.
#
# 사용:
#   scripts/ask-claude.sh "질문"                       # 기본 모델
#   scripts/ask-claude.sh -m opus "어려운 질문"         # opus / sonnet / haiku
#   echo "긴 질문" | scripts/ask-claude.sh -
#
# 읽기 전용으로 돈다 — 상담이 목적이고 코드 수정은 부른 쪽에서 한다.
set -euo pipefail

CWD="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="opus"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -C|--cd)    CWD="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    *) break ;;
  esac
done

[[ $# -gt 0 ]] || { echo "질문이 없다" >&2; exit 1; }
PROMPT="$*"
STDIN_USED=0
if [[ "$PROMPT" == "-" ]]; then PROMPT="$(cat)"; STDIN_USED=1; fi

cd "$CWD"
# stdin을 이미 소비했거나 쓰지 않으면 /dev/null로 닫는다 — 안 그러면 claude -p가
# 파이프 입력을 3초 기다리며 경고를 뱉는다.
exec 0</dev/null
exec claude -p "$PROMPT" --model "$MODEL" \
  --allowed-tools "Read,Grep,Glob,Bash(git log:*),Bash(git show:*),Bash(grep:*),Bash(sed:*),Bash(cat:*),Bash(ls:*)"
