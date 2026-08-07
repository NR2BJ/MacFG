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
# LaunchServices/Spotlight 재등록.
# rm -rf + cp -R 를 빠르게 반복하면 등록이 어긋나 **Finder 검색·Launchpad에서 앱이 사라진다**
# (실측 2026-08-07: mdls가 kMDItemDisplayName=null을 반환, 사용자가 "앱 폴더에 없다"고 제보).
# 번들 자체는 멀쩡하고 더블클릭도 되지만 찾을 수가 없으니 실사용에선 없어진 것과 같다.
LSREG=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
[ -x "$LSREG" ] && "$LSREG" -f /Applications/MacFG.app >/dev/null 2>&1
touch /Applications/MacFG.app

echo "설치됨: /Applications/MacFG.app ($(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' /Applications/MacFG.app/Contents/Info.plist))"
