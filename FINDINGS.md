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

## 2026-11-24: Metroid memory and progress measures

- QuickNES exposes only the 2 KB system RAM, but Metroid keeps items, missiles, tanks and the
  Kraid/Ridley statue flags in cartridge RAM ($6000-$7FFF). That RAM is in the save state's `SRAM`
  block; `Emulator.memory()` returns system + cartridge RAM (10 KB) from a save state in ~2 us.
- Verified by experiment: $001E goes 8 -> 3 when play starts; $0050 (map x) changes 3 -> 2 walking
  into the room to the left; $6878 becomes $10 (Morph Ball) when Samus touches the pedestal
  (`find_morph_ball.py`, found by random search in 53 attempts). Other addresses in `progress.py`
  are from the Data Crystal RAM map and the nmikstas disassembly, not yet verified.
- "One-way bits" is too noisy to be a headline measure: 300 random runs from one start state flag
  ~9,800 bits, because everything the game writes once after starting looks one-way. It's kept as a
  diagnostic; watch whether it shrinks as exploration revisits places. This is also an early warning
  that the toy's sticky-bit trick may not transfer directly.
