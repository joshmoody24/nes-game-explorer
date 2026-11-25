"""
Can lasting acquisitions (items, capacity upgrades) be found without knowing the game? Over recorded
segments, lists memory bits that went 0 -> 1 between consecutive steps but never 1 -> 0, and shows where
the known Metroid item and missile-capacity bytes fall.

Usage: python -m checks.one_way_check <run name> [recordings]
"""

import os
import sys
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import ENERGY_TANKS, ITEMS, MAX_MISSILES, at

NAMED = {at(ITEMS): "items", at(MAX_MISSILES): "max missiles", at(ENERGY_TANKS): "energy tanks"}


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    np.random.default_rng(0).shuffle(recordings)
    replayed = build(emulator, recordings[:count], np.arange(len(emulator.memory())))
    up = np.zeros(replayed.observations.shape[1] * 8, bool)
    down = np.zeros_like(up)
    for r in range(len(replayed.start_states)):
        bits = np.unpackbits(replayed.recording(r)[0], axis=1)
        step_up, step_down = (bits[1:] > bits[:-1]).any(0), (bits[1:] < bits[:-1]).any(0)
        up |= step_up
        down |= step_down
    one_way = np.flatnonzero(up & ~down)
    print(f"{len(one_way)} bits only ever went 0 -> 1 (of {up.sum()} that ever went up):")
    print("  " + ", ".join(f"{b // 8:#06x}.{7 - b % 8}{'=' + NAMED[b // 8] if b // 8 in NAMED else ''}" for b in one_way[:60]))
    for index, name in NAMED.items():
        bits = range(index * 8, index * 8 + 8)
        print(f"  {name} ({index:#06x}): bits that went up {[7 - b % 8 for b in bits if up[b]]}, "
              f"of which one-way {[7 - b % 8 for b in bits if up[b] and not down[b]]}")


if __name__ == "__main__":
    main()
