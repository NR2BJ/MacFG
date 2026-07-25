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

--reset 모드 (권장):
  MacFG 관련 기록을 **자기 기록까지 포함해 전부** 제거한다. 다음 실행이 ControlCenter에겐
  첫 실행이 되어 깨끗하게 채택된다. 실측 2026-07-25: 같은 바이너리를 번들ID만 바꿔
  (com.macfg.MacFGProbe) 띄웠더니 기록이 없는 덕에 isAllowed=True를 받고 정상 표시됐다.
  자기 기록의 isAllowed가 False로 박힌 경우(시스템 설정에서 한 번 끄면 이렇게 된다)는
  기본 모드로는 못 고친다 — 기본 모드는 자기 기록을 보존하기 때문이다.

사용:
  python3 scripts/repair_menubar_allowlist.py --dry-run             # 기본 정리 미리보기
  python3 scripts/repair_menubar_allowlist.py --reset --dry-run     # 전체 제거 미리보기
  python3 scripts/repair_menubar_allowlist.py --reset --apply       # 적용 (백업 생성)
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
APP_ID = "com.macfg.MacFG"          # 정확히 이 번들ID만. com.macfg.MacFGProbe 같은 다른 앱은 건드리지 않는다
BUILD_MARK = "/MacFG/.build/"       # 번들 없이 실행한 잔재


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
    """정확히 MacFG 본체(또는 그 .build 잔재)인가. 부분일치를 쓰면 com.macfg.MacFGProbe
    같은 **다른 앱**까지 지운다 — 실측에서 실제로 그랬다."""
    return name == APP_ID or BUILD_MARK in name


def repair(entries, reset=False):
    """(새 목록, 변경 로그) 반환. 원본은 건드리지 않는다.

    reset=True면 com.macfg.MacFG 자기 기록도 제거해, 다음 실행이 첫 실행이 되게 한다.
    자기 기록의 isAllowed가 False로 박힌 상태는 이 모드로만 풀린다.
    """
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

        # 소유자가 MacFG 자신인 기록
        if owner == "com.macfg.MacFG":
            if reset:
                log.append(f"제거(자기 기록): {owner} (허용={e.get('isAllowed')}) — 다음 실행이 첫 실행이 된다")
                continue
            out.append(e)
            if e.get("isAllowed") is False:
                log.append(f"경고: {owner} 의 허용이 False다. 기본 모드는 이걸 못 고친다 → --reset 을 쓰라")
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
    reset = "--reset" in sys.argv
    if not os.path.exists(PLIST):
        print(f"허용 목록 없음: {PLIST}")
        return 1
    if "--verify" in sys.argv:
        return verify()
    if not apply and "--dry-run" not in sys.argv:
        print(__doc__)
        return 2

    root = plistlib.load(open(PLIST, "rb"))
    entries = plistlib.loads(bytes(root["trackedApplications"]))
    new, log = repair(entries, reset=reset)
    print(f"모드: {'--reset (자기 기록까지 제거)' if reset else '기본 (자기 기록 보존)'}")

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
    print("""
다음이 **중요**하다 — ControlCenter는 이 파일을 메모리에 들고 있다:

    killall -9 ControlCenter

**반드시 -9(SIGKILL)여야 한다.** 그냥 `killall`(SIGTERM)을 쓰면 ControlCenter가 죽기 전에
자기 메모리 상태를 디스크에 flush해서 방금 쓴 편집을 **되돌려 버린다**(2026-07-25 실측:
적용 3분 뒤 오염 기록 5건이 전부 되살아나 있었다). SIGKILL은 flush 기회를 주지 않으므로,
launchd가 재시작한 ControlCenter가 우리가 쓴 파일을 읽는다.

확인:  python3 scripts/repair_menubar_allowlist.py --verify
""")
    return 0


def verify():
    """편집이 살아남았는지 확인 — ControlCenter가 되돌렸는지 판정."""
    root = plistlib.load(open(PLIST, "rb"))
    entries = plistlib.loads(bytes(root["trackedApplications"]))
    claimers = []
    for e in entries:
        if not isinstance(e, dict) or "location" not in e:
            continue
        owner = loc_name(e["location"])
        if any(loc_name(l) == APP_ID for l in (e.get("menuItemLocations") or [])) or BUILD_MARK in owner:
            claimers.append((owner, e.get("isAllowed")))
    print(f"{APP_ID} 를 주장하는 소유자 {len(claimers)}건:")
    for o, a in claimers:
        print(f"  {o[:70]}  허용={a}")
    foreign = [c for c in claimers if c[0] != APP_ID]
    if foreign:
        print("\n→ 아직 남의 주장이 있다. 수리가 되돌려졌거나 아직 적용되지 않았다.")
        return 1
    print("\n→ 자기 주장만 남았다. 이제 Finder에서 dist/MacFG.app 을 더블클릭하라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
