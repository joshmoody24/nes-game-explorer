"""Runs random long-hold tries from a save until some reach a target map cell; writes the first arrival's save."""
import os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import GAME_MODE, GAME_OVER, MAP_X, MAP_Y

def tries(task):
    state, target, seeds, steps = task
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for seed in seeds:
        rng = np.random.default_rng(seed); keep = 0.99 - rng.random() * 0.09; a = rng.integers(len(ACTIONS))
        emulator.load(state)
        for _ in range(steps):
            if rng.random() > keep: a = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[a], FRAMES_PER_STEP)
            m = emulator.memory()
            if m[GAME_MODE] == GAME_OVER: break
            if (int(m[MAP_Y]), int(m[MAP_X])) == target:
                for _ in range(10): emulator.run(0, FRAMES_PER_STEP)
                return emulator.save()
    return None

if __name__ == "__main__":
    source, target, out = sys.argv[1], tuple(int(v) for v in sys.argv[2].split(",")), sys.argv[3]
    state = Path(source).read_bytes()
    with get_context("spawn").Pool(20) as pool:
        for found in pool.imap_unordered(tries, [(state, target, range(k, 20000, 20), 500) for k in range(20)]):
            if found:
                Path(out).write_bytes(found); print("arrived; wrote", out); pool.terminate(); break
        else:
            print("no arrival")
