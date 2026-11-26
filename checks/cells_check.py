"""
World-map coverage of runs (a diagnostic using Metroid's map from the ROM): cells each run reached that a reference
run never did, and cells where Kraid or Ridley is present ($6987 = 1).

Usage: python -m checks.cells_check <reference run> <run> [<run> ...]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

from checks.frontier_check import world_map
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at

BOSS_PRESENT = 0x6987


def cells(emulator, run):
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    views = [emulator.view(s) for s in saved["states"]]
    reached = {(int(v[MAP_Y]), int(v[MAP_X])) for v in views}
    boss = {(int(v[MAP_Y]), int(v[MAP_X])) for v in views if v[at(BOSS_PRESENT)] == 1}
    return reached, boss


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    reference, _ = cells(emulator, sys.argv[1])
    total = set(reference)
    for run in sys.argv[2:]:
        reached, boss = cells(emulator, run)
        total |= reached
        print(f"{run}: {len(reached)} cells, new {sorted(reached - reference)}; boss present in {sorted(boss) or 'none'}")
    print(f"together: {len(total)} of {len(world_map())} map cells")


if __name__ == "__main__":
    main()
