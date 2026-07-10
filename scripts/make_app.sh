#!/bin/zsh
# MacFG.app 번들 + DMG 생성
# 사용: scripts/make_app.sh <version>   (예: scripts/make_app.sh 1.0.0)
set -euo pipefail

VERSION="${1:-1.0.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
for m in 288 360 432 540; do
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

echo "── 완료"
ls -la "$DIST"
codesign -dv "$APP" 2>&1 | head -3
