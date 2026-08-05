#!/bin/zsh
# 재빌드 → /Applications 재설치. 테스트 루프용.
#
# 왜 /Applications인가: 자동화 도구가 앱을 인식하려면 LaunchServices에 정상 등록된
# 위치에 있어야 한다. dist/ 안의 번들은 "설치되지 않은 앱"이라 화면 캡처에서 필터링되고
# 클릭도 불가능하다. 번들ID는 그대로라 메뉴바 허용목록 기록에는 영향이 없다.
#
# 사용: scripts/install.sh <version>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$ROOT/scripts/make_app.sh" "$@" >/dev/null
pkill -f "/Applications/MacFG.app" 2>/dev/null || true
sleep 1
rm -rf /Applications/MacFG.app
cp -R "$ROOT/dist/MacFG.app" /Applications/
echo "설치됨: /Applications/MacFG.app ($(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' /Applications/MacFG.app/Contents/Info.plist))"
