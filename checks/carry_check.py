"""
Composition at the Long Beam door: carry an armed state across known territory (graph.travel, keeping the
arrival that holds the most of the hoarded byte at each hop) to a high spot in the door room, then count how
often random segments from there open the door. Hand-made Metroid knowledge picks the target and judges the
result: diagnostic only.

Usage: python -m checks.carry_check <start state file> <run glob under runs/> [segments]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

import graph
import workers
from checks.red_door_check import _opened
from emulator import Emulator
from experience import valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, ROOM_Y, at, energy_of

DOOR_ROOM = (0, 5, 8)


def missiles_held(memory):
    return int(memory[at(CURRENT_MISSILES)]), energy_of(memory)


def main():
    start = Path(sys.argv[1]).read_bytes()
    runs = sorted(Path("runs").glob(sys.argv[2]))
    segments = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    journeys = []
    for run in runs:
        saved = pickle.loads(zlib.decompress((run / "checkpoint.pkl.z").read_bytes()))
        journeys += [(start, path) for path in saved["paths"] if path]
        journeys += [(state, bytes(actions)) for state, actions in valid_recordings(run)]
    world = graph.build(journeys)
    high = {loc for u in world for loc in [u, *world[u]] if loc[:3] == DOOR_ROOM and loc[4] <= 2}
    print(f"map from {len(runs)} runs: {len(world)} locations; {len(high)} high spots in the door room", flush=True)
    rng = np.random.default_rng(0)
    arrived, state, steps, hops, failed = graph.travel(emulator, start, world, high, rng, keep=missiles_held)
    m = emulator.memory(state)
    print(f"carried: {'arrived' if arrived else 'stuck at ' + str(graph.location(m))} after {hops} hops "
          f"({steps} steps tried), holding {m[at(CURRENT_MISSILES)]} missiles, {energy_of(m)} energy, room y {m[ROOM_Y]}",
          flush=True)
    if arrived:
        reached = [r for result in workers.map(_opened, [(state, list(range(segments)))]) for r in result]
        print(f"random segments from there entered the room beyond the red door in {np.mean(reached):.1%} "
              f"({sum(reached)} of {len(reached)})")


if __name__ == "__main__":
    main()
