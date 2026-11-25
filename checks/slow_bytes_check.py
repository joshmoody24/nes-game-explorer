"""
Can "places" be chosen automatically instead of by hand? Ranks memory bytes by how rarely they change
between steps (while not being constant) on recordings, and shows where the hand-picked place bytes rank.
Game-agnostic: the same ranking would run on any game's RAM.

Usage: python -m checks.slow_bytes_check <run name>
"""

import os
import sys
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import AREA, CURRENT_MISSILES, ENERGY_TANKS, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, ROOM_Y, at

NAMED = {"area": at(AREA), "map y": at(MAP_Y), "map x": at(MAP_X), "room x": at(ROOM_X), "room y": at(ROOM_Y),
         "items": at(ITEMS), "energy tanks": at(ENERGY_TANKS), "max missiles": at(MAX_MISSILES),
         "missiles": at(CURRENT_MISSILES)}


def _entropy(values):
    counts = np.bincount(values, minlength=256)
    p = counts[counts > 0] / len(values)
    return float(-(p * np.log2(p)).sum())


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    np.random.default_rng(0).shuffle(recordings)
    replayed = build(emulator, recordings[:3000], np.arange(len(emulator.memory())))
    steps = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0) != 0
                            for r in range(len(replayed.start_states))])
    change_rate = steps.mean(0)
    distinct = np.array([len(np.unique(replayed.observations[:, b])) for b in range(steps.shape[1])])
    candidates = np.flatnonzero((change_rate > 0) & (distinct > 1))
    # Slowness alone favors junk that flips once; a state variable is slow *and* spread across many values.
    entropy = np.array([_entropy(replayed.observations[:, b]) for b in range(steps.shape[1])])
    slowness_score = entropy / (change_rate + 1e-3)
    ranked = candidates[np.argsort(-slowness_score[candidates])]
    print(f"{len(candidates)} bytes ever change; top 30 by entropy / change rate (index: changes per step, distinct values):")
    names = {index: name for name, index in NAMED.items()}
    print("  " + ", ".join(f"{b:#06x}{'=' + names[b] if b in names else ''}: {change_rate[b]:.4f}/{distinct[b]}"
                           for b in ranked[:30]))
    for name, index in NAMED.items():
        rank = int(np.flatnonzero(ranked == index)[0]) if index in ranked else None
        print(f"  {name:>13} ({index:#06x}): rank {rank}, changes per step {change_rate[index]:.4f}, "
              f"distinct values {distinct[index]}")


if __name__ == "__main__":
    main()
