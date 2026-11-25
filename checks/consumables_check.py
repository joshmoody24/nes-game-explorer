"""
Can "consumables" (ammo, health; in Zelda bombs, keys, hearts) be found without knowing the game? Ranks
memory bytes by how much more often random play decreases them than increases them, and shows where the
known Metroid resource bytes rank.

Usage: python -m checks.consumables_check <run name>
"""

import os
import sys
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, at

NAMED = {"missiles": at(CURRENT_MISSILES), "energy (tens)": 0x107, "energy (units)": 0x106}


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    np.random.default_rng(0).shuffle(recordings)
    replayed = build(emulator, recordings[:3000], np.arange(len(emulator.memory())))
    changes = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0)
                              for r in range(len(replayed.start_states))])
    decreases, increases = (changes < 0).mean(0), (changes > 0).mean(0)
    # A consumable drops in small steps far more often than it rises; counters and timers rise and wrap.
    small = (np.abs(changes) <= 16).mean(0)
    score = np.where((decreases > 1e-4) & (small > 0.999), decreases / (increases + 1e-4), 0)
    ranked = np.argsort(-score)
    names = {index: name for name, index in NAMED.items()}
    print("top 20 (index: decreases/increases per step):")
    print("  " + ", ".join(f"{b:#06x}{'=' + names[b] if b in names else ''}: {decreases[b]:.4f}/{increases[b]:.4f}"
                           for b in ranked[:20]))
    for name, index in NAMED.items():
        print(f"  {name:>15} ({index:#06x}): rank {int(np.flatnonzero(ranked == index)[0])}, "
              f"decreases {decreases[index]:.4f}, increases {increases[index]:.4f} per step")


if __name__ == "__main__":
    main()
