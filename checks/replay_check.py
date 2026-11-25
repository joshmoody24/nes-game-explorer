"""
Do recorded button sequences replay exactly? For sampled recordings, checks the start state still has the
recorded checksum, then replays the buttons twice and confirms both replays end in identical memory.

Usage: python -m checks.replay_check <run name> [recordings]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator


def replay(emulator, state, actions):
    emulator.load(state)
    for action in actions:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    return emulator.memory()


def main():
    run_dir = Path("runs") / sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    states = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))["states"]
    recordings = np.load(run_dir / "trajectories.npz")
    total = len(recordings["start"])
    picks = np.random.default_rng(0).choice(total, min(count, total), replace=False)
    matching = identical = 0
    for i in picks:
        state = states[recordings["start"][i]]
        if zlib.crc32(state) != recordings["start_checksum"][i]:
            continue
        matching += 1
        actions = recordings["actions"][recordings["offsets"][i]:recordings["offsets"][i + 1]]
        identical += np.array_equal(replay(emulator, state, actions), replay(emulator, state, actions))
    steps = len(recordings["actions"])
    print(f"{total} recordings, {steps} recorded steps ({steps / total:.0f} per recording)")
    print(f"of {len(picks)} checked: start state checksum matches {matching}, replays identical {identical}")


if __name__ == "__main__":
    main()
