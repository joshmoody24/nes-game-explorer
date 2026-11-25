"""
Causal test at the Long Beam red door: copies of archive states near room (5,8)'s left edge, with 5
missiles edited into memory (current and max), then random Go-Explore-style segments. How often does random
play open the door and reach room (5,7), versus the unedited copies? Hand-made Metroid knowledge,
diagnostic only.

Usage: python -m checks.red_door_check <run name> [segments per state]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import CURRENT_MISSILES, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, area_of, at

DOOR_ROOM, BEYOND = (0, 5, 8), (0, 5, 7)


def _opened(task):
    state, seeds = task
    emulator = workers.emulator()
    reached = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        emulator.load(state)
        action, hit = int(rng.integers(len(ACTIONS))), False
        for _ in range(100):
            if rng.random() > 0.9:
                action = int(rng.integers(len(ACTIONS)))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            m = emulator.memory()
            if (area_of(m), int(m[MAP_Y]), int(m[MAP_X])) == BEYOND:
                hit = True
                break
        reached.append(hit)
    return reached


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / sys.argv[1] / "checkpoint.pkl.z").read_bytes()))
    per_state = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    near = [s for s in saved["states"]
            if (lambda m: (area_of(m), int(m[MAP_Y]), int(m[MAP_X])) == DOOR_ROOM and m[ROOM_X] < 64)(emulator.memory(s))]
    offsets = emulator.state_offsets([at(CURRENT_MISSILES), at(MAX_MISSILES)])
    armed = []
    for state in near:
        edited = bytearray(state)
        for offset in offsets:
            edited[offset] = 5
        armed.append(bytes(edited))
    for label, states in [("as found (0 missiles)", near), ("with 5 missiles edited in", armed)]:
        tasks = [(s, [k * 1000 + j for j in range(per_state)]) for k, s in enumerate(states)]
        reached = [r for result in workers.map(_opened, tasks) for r in result]
        print(f"{label}: {len(states)} states x {per_state} segments: entered the room beyond the red door in "
              f"{np.mean(reached):.2%} ({sum(reached)})", flush=True)


if __name__ == "__main__":
    main()
