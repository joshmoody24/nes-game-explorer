"""
Start states a fixed number of steps before real acquisitions, for milestone tests: for each kind of gain a run
made over its start (an item bit, an energy tank, missile capacity), the kept visit with the shortest path to it
is replayed from the run's start file, minus the last BEFORE steps, and saved.

Usage: python near_starts.py <run> [steps before, ...]    (writes cache/near-<run>-<gain>-<steps>.state)
"""

import json
import os
import pickle
import sys
import zlib
from pathlib import Path

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import ENERGY_TANKS, ITEMS, MAX_MISSILES, at
from progress import ITEM_BITS


def gains(view, start):
    found = [name for bit, name in ITEM_BITS.items() if view[at(ITEMS)] & bit and not start[at(ITEMS)] & bit]
    if view[at(ENERGY_TANKS)] > start[at(ENERGY_TANKS)]:
        found.append("tank")
    if view[at(MAX_MISSILES)] > start[at(MAX_MISSILES)]:
        found.append("missile pack")
    return found


def replay(emulator, start, actions):
    emulator.load(start)
    for a in actions:
        emulator.run(ACTIONS[a], FRAMES_PER_STEP)
    return emulator.save()


def main():
    run, befores = sys.argv[1], [int(a) for a in sys.argv[2:]] or [200, 500]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    run_dir = Path("runs") / run
    start = Path(json.loads((run_dir / "config.json").read_text())["start_state"]).read_bytes()
    start_view = emulator.view(start)
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    shortest = {}
    for i, (state, path) in enumerate(zip(saved["states"], saved["paths"])):
        for gain in gains(emulator.view(state), start_view):
            if path and (gain not in shortest or len(path) < len(saved["paths"][shortest[gain]])):
                shortest[gain] = i
    for gain, i in sorted(shortest.items()):
        path = saved["paths"][i]
        reached = gain in gains(emulator.view(replay(emulator, start, path)), start_view)
        for before in befores:
            if before >= len(path):
                continue
            out = Path(f"cache/near-{run.replace('/', '-')}-{gain.replace(' ', '-')}-{before}.state")
            out.write_bytes(replay(emulator, start, path[:-before]))
            print(f"{gain}: path {len(path)} steps (replay reaches it: {reached}); {before} steps before -> {out}")


if __name__ == "__main__":
    main()
