"""
Does action diversity (empowerment) rate a state higher when it has missiles? Takes archived states
with missile capacity but 0 missiles, makes an exact copy with the missile byte set to 5 by editing the
save state, and scores both with the same random button sequences. Metroid addresses are used only to
build the test, not by the score.

Usage: python -m checks.empowerment_check <run name> [state count]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from emulator import Emulator
from empowerment import action_diversity
from metroid import CURRENT_MISSILES, MAX_MISSILES, at


def with_missiles(emulator, state, count):
    """The same save state with the current-missile byte changed."""
    edited = bytearray(state)
    edited[emulator._cartridge_slice.start + CURRENT_MISSILES - 0x6000] = count
    return bytes(edited)


def main():
    run_dir = Path("runs") / sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    archive = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    empty = [s for s in archive["states"]
             if emulator.view(s)[at(MAX_MISSILES)] >= 5 and emulator.view(s)[at(CURRENT_MISSILES)] == 0]
    picks = np.random.default_rng(0).choice(len(empty), min(count, len(empty)), replace=False)
    emulator.load(with_missiles(emulator, empty[picks[0]], 5))
    emulator.run(0, 1)
    assert emulator.memory()[at(CURRENT_MISSILES)] == 5, "the emulator didn't take the edited missile count"

    without, with_ = [], []
    for i, pick in enumerate(picks):
        state = empty[pick]
        without.append(action_diversity(emulator, state, np.random.default_rng(i)))
        with_.append(action_diversity(emulator, with_missiles(emulator, state, 5), np.random.default_rng(i)))
    without, with_ = np.array(without), np.array(with_)
    print(f"{len(picks)} states. Bits changed by my actions, median: without missiles {np.median(without):.0f}, "
          f"with 5 missiles {np.median(with_):.0f}")
    print(f"with missiles scored higher in {np.mean(with_ > without):.0%} of states, equal in "
          f"{np.mean(with_ == without):.0%}, lower in {np.mean(with_ < without):.0%}")

    # Control: the same paired test, editing a byte that never varies across saved states (likely unused).
    memories = np.stack([emulator.memory(s) for s in archive["states"][:2000]])
    constant = np.flatnonzero((memories == memories[0]).all(0))
    rng = np.random.default_rng(1)
    higher, lower = 0, 0
    for i, pick in enumerate(picks):
        state = bytearray(empty[pick])
        byte = int(rng.choice(constant))
        offset = (emulator._system_slice.start + byte if byte < 0x800
                  else emulator._cartridge_slice.start + byte - 0x800)
        state[offset] ^= 0x55
        edited = action_diversity(emulator, bytes(state), np.random.default_rng(i))
        higher += edited > without[i]
        lower += edited < without[i]
    print(f"control (flip bits of a never-varying byte): scored higher in {higher / len(picks):.0%}, "
          f"lower in {lower / len(picks):.0%}")

if __name__ == "__main__":
    main()
