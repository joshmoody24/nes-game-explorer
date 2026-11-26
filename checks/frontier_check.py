"""
Which unexplored parts of the world are blocked, and how hard? Reads Metroid's 32x32 world map from the ROM (file
offset 0x254E; 0xFF = no room; a diagnostic that uses game knowledge), finds map cells next to cells a run explored,
and for each, runs quick tries (long holds, refilled energy and missiles) from saves in the explored neighbour,
counting how often a try enters the unexplored cell. A rate of 0 marks a real wall.

Usage: python -m checks.frontier_check <run> [tries per save]
"""

import os
import pickle
import sys
import zlib
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM, refilled
from metroid import GAME_MODE, GAME_OVER, MAP_X, MAP_Y

MAP_OFFSET, STEPS, SAVES = 0x254E, 200, 3
REFILL = "4218:99,4217:99,262:0x99,263:0x03"


def world_map():
    grid = np.frombuffer(Path(ROM).read_bytes()[MAP_OFFSET:MAP_OFFSET + 1024], np.uint8).reshape(32, 32)
    return {(y, x) for y in range(32) for x in range(32) if grid[y, x] != 0xFF}


def _tries(task):
    state, target, seeds = task
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    hits = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        keep = 0.99 - rng.random() * 0.09
        action = rng.integers(len(ACTIONS))
        emulator.load(state)
        for _ in range(STEPS):
            if rng.random() > keep:
                action = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            memory = emulator.memory()
            if memory[GAME_MODE] == GAME_OVER:
                break
            if (int(memory[MAP_Y]), int(memory[MAP_X])) == target:
                hits += 1
                break
    return hits


def main():
    run, tries = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    saves_at = defaultdict(list)
    for state in saved["states"]:
        view = emulator.view(state)
        saves_at[(int(view[MAP_Y]), int(view[MAP_X]))].append(state)
    cells = world_map()
    frontier = [(cell, near) for cell in sorted(cells - set(saves_at))
                for near in [(cell[0] + dy, cell[1] + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                if near in saves_at]
    print(f"{len(cells)} map cells, {len(saves_at)} explored, {len({c for c, _ in frontier})} unexplored cells next to "
          f"explored ones ({len(frontier)} edges)", flush=True)
    rng = np.random.default_rng(0)
    with get_context("spawn").Pool(20) as pool:
        for cell, near in frontier:
            states = [saves_at[near][i] for i in rng.choice(len(saves_at[near]), min(SAVES, len(saves_at[near])), replace=False)]
            tasks = [(refilled(emulator, s, REFILL), cell, range(k, tries, 20)) for s in states for k in range(20)]
            hits = sum(pool.map(_tries, tasks))
            print(f"  {cell} from {near}: {hits}/{tries * len(states)}", flush=True)


if __name__ == "__main__":
    main()
