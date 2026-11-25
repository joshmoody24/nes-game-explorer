"""
What blocks exploration? Finds archive states at a room's left or right edge whose neighboring map cell
was never visited (doors to unexplored territory), and saves one screenshot per such door, with the most
missiles any state there held. Hand-made Metroid knowledge, diagnostic only.

Usage: python -m checks.frontier_check <run name>
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emulator import Emulator
from explorers import ROM
from metroid import CURRENT_MISSILES, MAP_X, MAP_Y, ROOM_X, area_of, at


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / sys.argv[1] / "checkpoint.pkl.z").read_bytes()))
    memories = [emulator.memory(s) for s in saved["states"]]
    rooms = {(area_of(m), int(m[MAP_Y]), int(m[MAP_X])) for m in memories}
    doors = {}
    for i, m in enumerate(memories):
        area, y, x = area_of(m), int(m[MAP_Y]), int(m[MAP_X])
        side = -1 if m[ROOM_X] < 24 else 1 if m[ROOM_X] > 232 else 0
        if side and (area, y, x + side) not in rooms:
            door = (area, y, x, side)
            if door not in doors or m[at(CURRENT_MISSILES)] > memories[doors[door]][at(CURRENT_MISSILES)]:
                doors[door] = i
    print(f"{len(rooms)} rooms visited; {len(doors)} room edges facing unvisited map cells:")
    shots = []
    for (area, y, x, side), i in sorted(doors.items()):
        print(f"  area {area} room ({y},{x}) {'left' if side < 0 else 'right'} edge: most missiles held "
              f"{memories[i][at(CURRENT_MISSILES)]}")
        emulator.load(saved["states"][i])
        shots.append(emulator.frame(0))
    columns = 6
    while len(shots) % columns:
        shots.append(np.zeros_like(shots[0]))
    grid = np.concatenate([np.concatenate(shots[r:r + columns], 1) for r in range(0, len(shots), columns)], 0)
    plt.imsave("/tmp/frontier.png", grid)


if __name__ == "__main__":
    main()
