#!/usr/bin/env python3
"""ControlCenter 메뉴바 허용 목록에서 MacFG 오염 기록만 걷어낸다.

왜 필요한가 (2026-07-25 실측):
  macOS 26의 ControlCenter는 메뉴바 항목 권한을 앱이 아니라 **띄운 프로세스**
  (responsible process) 기준으로 기록한다. 개발 중 셸/에이전트에서 앱을 반복 실행하면
  MacFG의 메뉴 항목이 iTerm2 · Claude Code · Claude Desktop 같은 남의 소유자 기록에
  등재되고, 그중 하나라도 isAllowed=false면 같은 항목을 상충하는 허용 상태의 소유자
  여럿이 주장하는 꼴이 된다. 그러면 ControlCenter가 **아무것도 채택하지 않아서**
  NSStatusItem이 창 서버에 올라가지 못한다 — AppKit은 isVisible=true라고 보고하지만
  CGWindowList에는 우리 pid의 창이 0개다.

무엇을 지우는가:
  - 소유자가 MacFG가 아닌데 menuItemLocations에 MacFG를 끼워넣은 기록 → **MacFG만 제거**
    (그 소유자의 자기 항목은 건드리지 않는다)
  - 소유자가 그렇게 해서 항목이 하나도 안 남으면 그 기록 자체를 제거
  - .build/... 를 가리키는 adhocBinary 기록 → 통째로 제거 (번들 아닌 실행의 잔재)
  - com.macfg.MacFG 자신의 정상 기록은 **보존한다**

사용:
  python3 scripts/repair_menubar_allowlist.py --dry-run     # 무엇이 바뀔지만 출력
  python3 scripts/repair_menubar_allowlist.py --apply       # 실제 수정 (백업 생성)
"""
import os
import plistlib
import shutil
import sys
import time

PLIST = os.path.expanduser(
    "~/Library/Group Containers/group.com.apple.controlcenter/"
    "Library/Preferences/group.com.apple.controlcenter.plist"
)
NEEDLES = ("com.macfg.MacFG", "/MacFG/.build/", "MacFGApp")


def loc_name(loc):
    """소유자/항목 식별자를 사람이 읽는 문자열로."""
    if not isinstance(loc, dict):
        return str(loc)
    if "bundle" in loc:
        return str(loc["bundle"].get("_0", loc["bundle"]))
    if "adhocBinary" in loc:
        inner = loc["adhocBinary"].get("_0", {})
        return str(inner.get("relative", inner)) if isinstance(inner, dict) else str(inner)
    return str(loc)


def is_macfg(name):
    return any(n in name for n in NEEDLES)


def repair(entries):
    """(새 목록, 변경 로그) 반환. 원본은 건드리지 않는다."""
    out, log = [], []
    for e in entries:
        if not isinstance(e, dict) or "location" not in e:
            out.append(e)                     # 소유자 기록이 아닌 항목은 그대로
            continue
        owner = loc_name(e["location"])
        locs = e.get("menuItemLocations") or []

        # .build 바이너리 소유자 기록 = 번들 아닌 실행의 잔재 → 통째로 제거
        if "/MacFG/.build/" in owner:
            log.append(f"제거(adhoc 소유자): {owner}")
            continue

        # 소유자가 MacFG 자신이면 보존
        if owner == "com.macfg.MacFG":
            out.append(e)
            continue

        kept = [l for l in locs if not is_macfg(loc_name(l))]
        if len(kept) == len(locs):
            out.append(e)                     # MacFG와 무관 → 그대로
            continue

        dropped = [loc_name(l) for l in locs if is_macfg(loc_name(l))]
        if kept:
            new = dict(e)
            new["menuItemLocations"] = kept
            out.append(new)
            log.append(f"정리: {owner} (허용={e.get('isAllowed')}) 에서 {dropped} 제거")
        else:
            log.append(f"제거(빈 소유자): {owner} (허용={e.get('isAllowed')}) — 항목이 {dropped} 뿐이었음")
    return out, log


def main():
    apply = "--apply" in sys.argv
    if not apply and "--dry-run" not in sys.argv:
        print(__doc__)
        return 2
    if not os.path.exists(PLIST):
        print(f"허용 목록 없음: {PLIST}")
        return 1

    root = plistlib.load(open(PLIST, "rb"))
    entries = plistlib.loads(bytes(root["trackedApplications"]))
    new, log = repair(entries)

    print(f"소유자 기록 {len(entries)} → {len(new)}")
    for line in log:
        print("  " + line)
    if not log:
        print("  변경 없음 — 오염 기록이 없다.")
        return 0
    if not apply:
        print("\n(--dry-run: 아무것도 쓰지 않았다. 실제 적용은 --apply)")
        return 0

    backup = f"{PLIST}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(PLIST, backup)
    root["trackedApplications"] = plistlib.dumps(new, fmt=plistlib.FMT_BINARY)
    with open(PLIST, "wb") as f:
        plistlib.dump(root, f, fmt=plistlib.FMT_BINARY)
    print(f"\n적용 완료. 백업: {backup}")
    print("이제 ControlCenter를 재시작해야 반영된다:  killall ControlCenter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
