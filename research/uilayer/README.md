# 정적 UI 층 — 합성 자막 오라클 실험

목적: 층의 채움 품질(P4)이 자막 주변 번짐의 원인인지 가린다. 자막 없는 실제 움직이는 영상에 정지 자막을 얹으면
자막 밑 진짜 배경을 알기 때문에, "완벽한 채움"을 넣었을 때의 상한을 잴 수 있다.

1. `swiftc -O -o mksub mksub.swift && ./mksub synth_sub.png` — 1920×1080 투명 캔버스에 빨강+검은 외곽선 자막.
2. 자막 없는 연속 프레임(예: `만약 이 영상이…webm` 1787.5~1789.6초 = seq_stairs 150~275)에 ffmpeg overlay로 얹어
   `seq_synth/`, 원본은 `seq_synth_bg/`.
3. 연속쌍(MACFG_TRIPSEQ=1, DUMPALL)으로 세 팔: UI 처리 없음(MACFG_UILAYER=0, seq_synth) / 층(MACFG_TRIPMASK=1, seq_synth) /
   오라클(MACFG_UILAYER=0, seq_synth_bg → 출력 위에 synth_sub.png overlay).
4. 지표: frozenband(자막 24px 이내 움직이는 배경 중 A·B·블렌드로 얼어붙은 비율), 자막 영역 PSNR.

결과(2026-09-30): 띠 MetalFlow 47.2 / 37.8 / 39.2%, RIFE 27.4 / 19.8 / 25.4%, AppleFI 23.4 / 19.0 / 22.7%
(없음 / 층 / 오라클). 자막 영역 PSNR MetalFlow 15.39 / 15.59 / 15.57dB.
→ **층은 이미 완벽한 채움과 같은 수준이다. 남은 번짐은 엔진의 빠른 움직임 보간 품질**(MetalFlow가 가장 약함)이지 UI 처리가 아니다.
P4(모션 보상 채움)는 이 지표로는 이득이 없다.
