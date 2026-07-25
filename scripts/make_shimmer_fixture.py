#!/usr/bin/env python3
"""텍스트 흔들림 지표(staticDev) 검증용 합성 삼중항 생성기.

왜 합성인가: staticDev가 **실제로 흔들림에 반응하는지** 먼저 확인해야 한다. 실프레임으로
바로 가면 지표가 잘못돼도 "콘텐츠 탓"과 구분이 안 된다. 여기서는 정답을 알고 있다 —
왼쪽 절반은 정지(텍스트 모사), 오른쪽 절반은 등속 이동. 정지 영역을 건드리는 설정이면
staticDev가 떨어져야 한다.

압축 노이즈 모사가 핵심: 실제 스트리밍 영상의 정적 텍스트는 완전히 동일하지 않고 ±1~3 LSB로
흔들린다. 바로 그 미세한 차이 때문에 정적 판정 문턱을 조이면 텍스트가 "움직이는 픽셀"로
분류돼 워프를 먹는다. 노이즈가 없으면 이 현상 자체가 재현되지 않는다.

사용: python3 scripts/make_shimmer_fixture.py <출력디렉터리> [프레임수] [폭] [높이]
"""
import os
import struct
import sys
import zlib

def write_png(path, w, h, rgba):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4:(y + 1) * w * 4]) for y in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)


def lcg(seed):
    """결정론적 의사난수 — 같은 픽스처가 매번 재현돼야 A/B가 의미를 갖는다."""
    s = seed & 0xFFFFFFFF
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s


def frame(idx, w, h):
    """움직이는 배경 **위에 얹힌** 정지 텍스트 — 실제 흔들림이 나오는 배치.

    첫 픽스처는 정지 영역과 이동 영역을 좌우로 갈라놨는데 지표가 반응하지 않았다(±0.065dB).
    당연했다 — 정지 영역 주변에 움직임이 없으면 그곳 flow는 깨끗하게 0이고, 0으로 워프하면
    원본과 같아서 흔들릴 이유가 없다. 흔들림은 **주변 움직임이 정지 요소의 flow를 오염시킬 때**
    생긴다(사용자 표현: "정적 UI나 패턴이 빠르게 이동할 때"). 그래서 자막/UI처럼 겹쳐 놓는다.
    """
    px = bytearray(w * h * 4)
    rnd = lcg(0xC0FFEE + idx * 7919)
    # 배경 스크롤 10px/프레임. **주기적 텍스처 + 주기의 배수인 이동량은 금물** — 첫 시도에서
    # 텍스처 주기 32px에 A→B 이동량이 정확히 32px이라 배경이 A와 B에서 픽셀 단위로 동일해졌고,
    # 정적비율이 99%로 떠서 지표가 아무것도 못 갈랐다. 셀 24px / 이동 10px(A→B=20px)로 어긋낸다.
    shift = idx * 10
    band_y0, band_y1 = h * 3 // 5, h * 3 // 5 + 96      # 자막 띠
    panel_x0, panel_x1 = w - 300, w - 40               # 우측 UI 패널
    panel_y0, panel_y1 = 40, 40 + 240
    for y in range(h):
        rowbase = y * w * 4
        for x in range(w):
            i = rowbase + x * 4
            # ── 배경: 24px 셀의 **비주기** 블록 노이즈가 통째로 스크롤.
            # 블록이라 flow가 추적할 수 있고(고주파 난수면 폴백으로 빠진다), 비주기라
            # 어떤 이동량에서도 A와 B가 같아지지 않는다.
            sx = x + shift
            cell = ((sx // 24) * 2654435761 ^ (y // 24) * 40503) & 0x7FFFFFFF
            v = 30 + (cell % 200)
            # ── 그 위에 정지 요소: 자막 띠 + UI 패널 (배경과 무관하게 절대 안 움직인다)
            in_band = band_y0 <= y < band_y1 and w // 6 <= x < w - w // 6
            in_panel = panel_y0 <= y < panel_y1 and panel_x0 <= x < panel_x1
            if in_band or in_panel:
                stroke = (x % 9 < 3) and (6 <= (y % 22) <= 15)   # 글자 획 모사
                v = 30 if stroke else 240
                # 압축 노이즈 ±2 LSB — 실제 스트리밍 정적 텍스트가 이렇게 흔들린다.
                # 정적 판정 문턱이 0.004(≈1 LSB)/0.008(≈2 LSB)이라 바로 이 대역에서 판정이 갈린다.
                v += (next(rnd) % 5) - 2
            px[i] = px[i + 1] = px[i + 2] = max(0, min(255, v))
            px[i + 3] = 255
    return px


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 720
    os.makedirs(out, exist_ok=True)
    for i in range(n):
        write_png(os.path.join(out, f"frame_{i:03d}.png"), w, h, frame(i, w, h))
    print(f"{n}장 생성 → {out}  ({w}x{h})")
    print("  좌측 절반 = 정지(텍스트 모사, 압축노이즈 ±2 LSB) / 우측 절반 = 등속 이동")
    return 0


if __name__ == "__main__":
    sys.exit(main())
