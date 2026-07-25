#!/bin/zsh
# MacFG.app 번들 + DMG 생성
# 사용: scripts/make_app.sh <version>   (예: scripts/make_app.sh 1.0.0)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 버전 결정. **틀린 버전으로 배포하면 되돌릴 수 없다.**
# UpdateChecker가 번들 버전을 최신 릴리즈 태그와 비교하는데, 번들이 실제보다 낮은 버전을
# 주장하면 모든 사용자가 6시간마다 "업데이트 있음"을 영구히 보게 되고 지울 방법이 없다.
# 그래서 조용한 폴백을 두지 않는다:
#   · 인자로 주면 그대로 쓴다 (릴리즈 표준 경로)
#   · 인자가 없으면 HEAD가 **정확히** 태그 위에 있을 때만 그 태그를 쓴다.
#     `describe --abbrev=0`은 HEAD에서 도달 가능한 최신 태그를 주므로, v1.2 태그를 달기 전에
#     빌드하면 조용히 1.1.5로 찍힌다 — 이게 정확히 막으려는 사고다.
#   · 둘 다 아니면 중단한다. 1.0.0 폴백은 얕은 클론에서 같은 사고를 낸다.
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  EXACT="$(git -C "$ROOT" describe --tags --exact-match 2>/dev/null || true)"
  if [[ -z "$EXACT" ]]; then
    echo "✘ 버전을 결정할 수 없다." >&2
    echo "  HEAD가 태그 위에 있지 않다. 버전을 명시하거나 태그를 먼저 달아라:" >&2
    echo "    scripts/make_app.sh 1.2.0" >&2
    echo "  (인자 없이 빌드하면 태그 이전 버전으로 찍혀, 모든 사용자에게 지울 수 없는" >&2
    echo "   '업데이트 있음' 알림이 영구히 남는다)" >&2
    exit 1
  fi
  VERSION="${EXACT#v}"
  echo "── 버전 $VERSION (태그 $EXACT)"
fi
DIST="$ROOT/dist"
APP="$DIST/MacFG.app"

echo "── release 빌드"
cd "$ROOT"
swift build -c release

echo "── 아이콘 생성"
ICON_TMP="$DIST/icon"
rm -rf "$DIST"
mkdir -p "$ICON_TMP/MacFG.iconset"
swift "$ROOT/scripts/gen_icon.swift" "$ICON_TMP/icon_1024.png"
for s in 16 32 64 128 256 512; do
  sips -z $s $s "$ICON_TMP/icon_1024.png" --out "$ICON_TMP/MacFG.iconset/icon_${s}x${s}.png" > /dev/null
  d=$((s * 2))
  sips -z $d $d "$ICON_TMP/icon_1024.png" --out "$ICON_TMP/MacFG.iconset/icon_${s}x${s}@2x.png" > /dev/null
done
iconutil -c icns "$ICON_TMP/MacFG.iconset" -o "$ICON_TMP/MacFG.icns"

echo "── 앱 번들 구성"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/release/MacFGApp" "$APP/Contents/MacOS/MacFG"
cp "$ICON_TMP/MacFG.icns" "$APP/Contents/Resources/"

echo "── 신경망(RIFE) 모델 컴파일·번들"
# 180/216/240 = 거버너 부하 강등용 sub-288 flow 티어 (predict 3.1/5.0/6.2ms). bypass 대신
# 낮은 flow로 보간을 유지하는 데 필수 — 번들에 없으면 부하 시 다시 보간이 꺼진다.
for m in 180 216 240 288 360 432 540; do
  if [ -d "$ROOT/Models/rife$m.mlpackage" ]; then
    xcrun coremlcompiler compile "$ROOT/Models/rife$m.mlpackage" "$APP/Contents/Resources/" > /dev/null
    echo "   rife$m.mlmodelc"
  fi
done

cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>MacFG</string>
    <key>CFBundleIdentifier</key><string>com.macfg.MacFG</string>
    <key>CFBundleName</key><string>MacFG</string>
    <key>CFBundleDisplayName</key><string>MacFG</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>CFBundleIconFile</key><string>MacFG</string>
    <key>LSMinimumSystemVersion</key><string>26.0</string>
    <!-- 메뉴바 전용 앱 — 없으면 .app 실행 시 Dock 아이콘이 뜬다(앱이 런타임에
         setActivationPolicy(.accessory)를 부르지만 그 전에 이미 등록됨). 사용자가
         "메뉴바만" 설정을 끄면 런타임에 .regular로 올려 Dock에 다시 표시된다. -->
    <key>LSUIElement</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSHumanReadableCopyright</key><string>MIT License</string>
</dict>
</plist>
PLIST

# 서명 정체성: "MacFG Dev" 자체서명 인증서가 유효하면 사용 — TCC(화면 녹화/손쉬운 사용)가
# 정체성+번들ID 기준으로 유지되어 재빌드마다 권한 재허가가 필요 없어진다.
# (ad-hoc은 빌드마다 CDHash가 바뀌어 TCC가 매번 다른 앱으로 취급)
if security find-identity -v -p codesigning 2>/dev/null | grep -q "MacFG Dev"; then
  echo "── 코드사인 (MacFG Dev — TCC 영속)"
  codesign -f -s "MacFG Dev" --entitlements "$ROOT/MacFGApp.entitlements" "$APP"
else
  echo "── 코드사인 (adhoc — MacFG Dev 인증서 없음/미신뢰)"
  codesign -f -s - --entitlements "$ROOT/MacFGApp.entitlements" "$APP"
fi

echo "── DMG 생성"
STAGE="$DIST/stage"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "MacFG $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DIST/MacFG-$VERSION.dmg" > /dev/null
rm -rf "$STAGE" "$ICON_TMP"

echo "── 서명 검증"
# 이전엔 `codesign -dv | head -3`이었는데, 그건 정보 출력일 뿐 **검증이 아니다** —
# 서명이 깨져도 그대로 통과했다. 실패하면 중단하도록 진짜 게이트로 바꾼다.
codesign --verify --deep --strict --verbose=2 "$APP"
echo "── 완료"
ls -la "$DIST"
# 정보 출력일 뿐이므로 실패해도 스크립트를 죽이지 않는다. `| head`는 head가 파이프를 닫는 순간
# codesign이 SIGPIPE로 죽고, set -o pipefail이 그걸 잡아 **성공한 빌드가 종료코드 141로 끝났다**.
codesign -dv "$APP" 2>&1 | sed -n '1,3p' || true
