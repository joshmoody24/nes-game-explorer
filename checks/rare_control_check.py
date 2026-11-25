"""
Can a comparison of *what my actions can change* pick the state with ammo, when the two states also
differ in position, enemies and scroll? Pairs come from two runs' archives (different states kept for
the same place); one of each pair gets 5 missiles by editing its save state. Compares:
  - total: number of bits my actions change (plain empowerment)
  - rare: bits only one state can change, weighted by how rarely bits are changeable across states
Metroid addresses are used only to build the test.

Usage: python -m checks.rare_control_check <run A> <run B> [pairs]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from checks.empowerment_check import with_missiles
from emulator import Emulator
from empowerment import controllable_bits
from metroid import MAX_MISSILES, at
from explorers import Settings, hand_made_place


def load(run):
    return pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))


def main():
    run_a, run_b = sys.argv[1:3]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    settings = Settings()
    by_place = {}
    for run in (run_a, run_b):
        for state in load(run)["states"]:
            by_place.setdefault(hand_made_place(emulator.view(state), settings), []).append(state)
    pairs = [states[:2] for states in by_place.values()
             if len(states) >= 2 and emulator.view(states[0])[at(MAX_MISSILES)] >= 5 and states[0] != states[1]]
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(count, len(pairs)), replace=False)]

    # How often each bit is changeable, estimated from a sample of states (rare = precious).
    sample = [s for states in by_place.values() for s in states]
    sample = [sample[i] for i in rng.choice(len(sample), 300, replace=False)]
    frequency = np.mean([controllable_bits(emulator, s, np.random.default_rng(i)) for i, s in enumerate(sample)], 0)
    rarity = -np.log(np.maximum(frequency, 1 / len(sample)))

    total_right = rare_right = weighted_right = ties = 0
    for i, (first, second) in enumerate(pairs):
        armed, unarmed = (first, second) if i % 2 else (second, first)  # alternate which state gets the ammo
        armed = with_missiles(emulator, armed, 5)
        armed_bits = controllable_bits(emulator, armed, np.random.default_rng(i))
        unarmed_bits = controllable_bits(emulator, unarmed, np.random.default_rng(i))
        total_right += armed_bits.sum() > unarmed_bits.sum()
        armed_only = (rarity * (armed_bits & ~unarmed_bits)).sum()
        unarmed_only = (rarity * (unarmed_bits & ~armed_bits)).sum()
        rare_right += armed_only > unarmed_only
        weighted_right += (rarity * armed_bits).sum() > (rarity * unarmed_bits).sum()
        ties += armed_only == unarmed_only
    print(f"{len(pairs)} pairs of different states at the same place, one given 5 missiles")
    print(f"plain total picks the armed state: {total_right / len(pairs):.0%}")
    print(f"rarity-weighted 'what only I can change' picks it: {rare_right / len(pairs):.0%} (ties {ties / len(pairs):.0%})")
    print(f"rarity-weighted total, one number per state: {weighted_right / len(pairs):.0%}")


if __name__ == "__main__":
    main()
