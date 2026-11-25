# Findings

Dated results, newest last. Each says what was measured and how to rerun it.

## 2026-11-24: emulator throughput (unit: Ryzen 9 9900X, 12 cores / 24 threads)

Measured with `python benchmark.py $QUICKNES_CORE 4` (random buttons, 4 frames per action, in-game).

| setup | 1 process | best total |
|---|---|---|
| FCEUmm core, Python callbacks | 5.5K frames/s | not measured |
| QuickNES core, Python callbacks | 26K | 349K (18 processes) |
| QuickNES, callbacks answered in C (`shim/shim.c`) | 28K | 378K |
| QuickNES, C callbacks, core told video/audio are off | 56K | **752K (18 processes)** |

- Python callbacks were only ~6% of the time; the core's own per-frame pixel conversion was ~35%.
- Telling the core video is off (`RETRO_ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE`) does not change game
  behavior: RAM identical at 334 checkpoints over 20K frames of random play (`check_video_off.py`).
  Only checked near the start of the game so far.
- Remaining profile: 75% 6502 CPU emulation, ~12% audio synthesis the core still does.
- Save/load state: ~1-5 us, ~21 KB per state.
