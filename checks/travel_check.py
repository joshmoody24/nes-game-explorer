"""
Composition test: carry armed states (5+ missiles) across known territory to the Long Beam red door with
graph.travel, then run random segments from where they arrive and count entries to the room beyond.
Hand-made Metroid knowledge for choosing states and judging, diagnostic only.

Usage: python -m checks.travel_check <run name> [armed states] [segments each] [paths for the map]
"""

import os
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import graph
import workers
from checks.red_door_check import DOOR_ROOM, _opened
from emulator import Emulator
from experience import valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, MAP_X, MAP_Y, area_of, at, energy_of


def holdings(memory):
    """What a journey should keep: missiles, then energy. Hand-given, diagnostic."""
    return int(memory[at(CURRENT_MISSILES)]), energy_of(memory)


def main():
    run = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    per_state = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    map_paths = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    states, paths = saved["states"], saved["paths"]
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    sample = rng.choice(len(paths), min(map_paths, len(paths)), replace=False)
    # Stored paths only lead outward from the start; recorded random segments also wander back.
    recordings = valid_recordings(Path("runs") / run)
    picked = rng.choice(len(recordings), min(20000, len(recordings)), replace=False)
    world = graph.build([(states[0], paths[i]) for i in sample if paths[i]]
                        + [(recordings[i][0], bytes(recordings[i][1])) for i in picked])
    print(f"map from {len(sample)} paths and {len(picked)} recordings: {len(world)} locations, {sum(len(v) for v in world.values())} edges "
          f"[{time.perf_counter() - started:.0f}s]", flush=True)
    memories = [emulator.memory(s) for s in states]
    targets = {loc for u in world for loc in [u, *world[u]] if loc[:3] == DOOR_ROOM and loc[3] == 0}
    armed = [i for i, m in enumerate(memories) if m[at(CURRENT_MISSILES)] >= 5 and energy_of(m) >= 20]
    picks = rng.choice(armed, min(count, len(armed)), replace=False)
    arrivals = []
    journeys = graph.travel_many([states[i] for i in picks], world, targets, keep=holdings)
    for i, (arrived, state, steps, hops, failed) in zip(picks, journeys):
        m, a = memories[i], emulator.memory(state)
        print(f"  from {area_of(m)},{m[MAP_Y]},{m[MAP_X]} ({m[at(CURRENT_MISSILES)]} missiles, {energy_of(m)} energy): "
              f"{'arrived' if arrived else 'stuck at ' + str(graph.location(a))} after {hops} hops, {steps} steps tried, "
              f"holding {a[at(CURRENT_MISSILES)]} missiles, {energy_of(a)} energy; last failed hop {failed}", flush=True)
        if arrived and a[at(CURRENT_MISSILES)] >= 5:
            arrivals.append(state)
    print(f"{len(arrivals)} of {len(picks)} armed states arrived at the door with 5+ missiles", flush=True)
    if arrivals:
        tasks = [(s, [k * 1000 + j for j in range(per_state)]) for k, s in enumerate(arrivals)]
        reached = [r for result in workers.map(_opened, tasks) for r in result]
        print(f"random segments from those arrivals entered the room beyond the red door in {np.mean(reached):.2%} "
              f"({sum(reached)} of {len(reached)})", flush=True)


if __name__ == "__main__":
    main()
