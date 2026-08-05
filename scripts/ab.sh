#!/bin/bash
# A/B 한 조건 준비 — 앱 종료 + defaults 적용 + 로그 마커.
# 실행 자체는 **Finder 더블클릭(⌘O)** 으로 해야 한다: 셸에서 띄우면 macOS 26이 그 셸을
# 메뉴바 항목 소유자로 기록해 허용 목록이 오염된다(2026-07-25 사건).
set -u
label="$1"; shift
pkill -f "/Applications/MacFG.app" 2>/dev/null; sleep 2
for kv in "$@"; do
  k="${kv%%=*}"; v="${kv#*=}"
  if [ -z "$v" ]; then defaults delete com.macfg.MacFG "env.$k" 2>/dev/null || true
  else defaults write com.macfg.MacFG "env.$k" -string "$v"; fi
done
echo "=== AB $label ===" >> /tmp/MacFG_diag.log
echo "준비됨: $label — Finder에서 ⌘O"
