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

echo "── 코드사인 (adhoc)"
codesign -f -s - --entitlements "$ROOT/MacFGApp.entitlements" "$APP"

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
