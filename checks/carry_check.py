"""
Positions wider than a byte, found without knowing the game: for each low byte (a byte that holding a direction
moves), the high byte is the one that steps by +-1 at the steps where the low byte wraps (jumps by more than
half its range) and almost never otherwise. Shows the pairs found on replayed recordings.

Usage: python -m checks.carry_check <run>[+<run>...]
"""

import os
import random
import sys
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y

LOWS = (0x51, 0x30E, 0x30D, 0x52, 0x8, 0x9, 0x6, 0x7)


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in sys.argv[1].split("+") for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    replayed = build(emulator, recordings[:1500], np.arange(len(emulator.memory())))
    diffs = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0)
                            for r in range(len(replayed.start_states))])
    print(f"{len(diffs):,} steps; known: map y {MAP_Y:#x}, map x {MAP_X:#x}, fine y {ROOM_Y:#x}, fine x {ROOM_X:#x}")
    for low in LOWS:
        wraps = np.abs(diffs[:, low]) > 128
        if wraps.sum() < 5:
            print(f"  low {low:#06x}: only {wraps.sum()} wraps")
            continue
        steps_one = np.abs(diffs) == 1
        at_wrap = steps_one[wraps].mean(0)
        elsewhere = (diffs[~wraps] != 0).mean(0)
        score = at_wrap - 20 * elsewhere
        score[low] = -1
        best = np.argsort(-score)[:3]
        print(f"  low {low:#06x} ({wraps.sum()} wraps): best high bytes " +
              ", ".join(f"{b:#06x} (steps at {at_wrap[b]:.0%} of wraps, changes at {elsewhere[b]:.3%} of other steps)"
                        for b in best), flush=True)


if __name__ == "__main__":
    main()
