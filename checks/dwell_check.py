"""
How often does a place change during ordinary play? Steps between changes ("dwell") of hand-made places and of
candidate bytes bucketed at several sizes, over replayed recordings.

Usage: python -m checks.dwell_check <run>[+<run>...]
"""

import os
import random
import sys
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM, Settings, location

CANDIDATES = (0x4, 0x6, 0x7, 0x8, 0x9, 0x4D, 0x4F, 0x50, 0x51, 0x52, 0x1EE, 0x223, 0x300, 0x301, 0x30D, 0x30E)


def dwell(keys):
    """Mean steps between changes of a per-step key, over all recordings."""
    changes = sum(int(np.count_nonzero(np.any(np.diff(k, axis=0) != 0, axis=tuple(range(1, k.ndim))))) for k in keys)
    return sum(len(k) - 1 for k in keys) / max(changes, 1)


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in sys.argv[1].split("+") for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    replayed = build(emulator, recordings[:1500], np.arange(len(emulator.memory())))
    memories = [replayed.recording(r)[0] for r in range(len(replayed.start_states))]
    hand = [np.array([location(m, Settings()) for m in rec]) for rec in memories]
    print(f"hand-made place: dwell {dwell(hand):.1f} steps; room only {dwell([h[:, :3] for h in hand]):.1f}", flush=True)
    for b in CANDIDATES:
        cells = ", ".join(f"/{k}: {dwell([rec[:, b] // k for rec in memories]):.1f}" for k in (1, 8, 32, 64))
        print(f"  {b:#06x} dwell {cells}", flush=True)


if __name__ == "__main__":
    main()
