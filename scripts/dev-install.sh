#!/bin/zsh
# 개발용 설치 — **릴리즈 번호를 소모하지 않는다.**
#
# install.sh는 버전 인자를 요구하는데, 내부 테스트마다 올리다 보니 하루에 1.7 → 1.9.4까지
# 갔다(2026-08-31). 그 속도면 진짜 릴리즈 전에 2.0을 써버린다.
# 개발 빌드는 `0.0.<커밋수>` 를 쓴다 — 단조 증가하되 릴리즈 대역(1.x/2.x)을 안 건드리고,
# 버전만 보고 어느 커밋인지 알 수 있다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N=$(git -C "$ROOT" rev-list --count HEAD)
exec "$ROOT/scripts/install.sh" "0.0.$N"
