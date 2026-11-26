"""
Where is a resource needed? For candidate dead ends (poor, tried often), compares random probes from a visit
there as found with the same visit holding the richest amount of the spendable (byte edited from the saved
state): spots where holding more lets probes reach locations the archive has never seen are where the
resource is needed. Shows where the Long Beam door room ranks (diagnostic only).

Usage: python -m checks.need_check <run name> <spendable byte> [candidates] [probes]
"""

import os
import pickle
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from graph import location


def _new_reached(task):
    """In a worker: how many of the probes from state reach a location outside known."""
    state, known, probes = task
    emulator, hits = workers.emulator(), 0
    for seed in range(probes):
        rng = np.random.default_rng(seed)
        emulator.load(state)
        action = int(rng.integers(len(ACTIONS)))
        for _ in range(100):
            if rng.random() > 0.9:
                action = int(rng.integers(len(ACTIONS)))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            if location(emulator.view(emulator.save())) not in known:
                hits += 1
                break
    return hits


def needs(emulator, run_dir, byte, candidates=60, probes=300):
    """
    (richest visit's state, candidate dead-end locations ranked by need, need per location). Need: extra probes
    reaching locations the archive never saw when the visit there holds the richest amount of byte.
    """
    saved = pickle.loads(zlib.decompress((Path(run_dir) / "checkpoint.pkl.z").read_bytes()))
    views = [emulator.view(s) for s in saved["states"]]
    locs = [location(v) for v in views]
    known = frozenset(locs)
    held, tried, children, visit_at = defaultdict(int), defaultdict(int), set(), {}
    for i, (loc, v, t) in enumerate(zip(locs, views, saved["times_chosen"])):
        if v[byte] >= held[loc]:
            visit_at[loc] = i
        held[loc] = max(held[loc], int(v[byte]))
        tried[loc] += t
    for p in saved["parents"]:
        if p >= 0:
            children.add(locs[p])
    rich = max(held.values())
    richest = max(range(len(views)), key=lambda i: int(views[i][byte]))
    dead = sorted((loc for loc in held if loc not in children and held[loc] < rich),
                  key=lambda loc: -(rich - held[loc]) * tried[loc])[:candidates]
    offset = emulator.state_offsets([byte])[0]
    tasks = []
    for loc in dead:
        state = saved["states"][visit_at[loc]]
        edited = bytearray(state)
        edited[offset] = rich
        tasks += [(state, known, probes), (bytes(edited), known, probes)]
    results = list(workers.map(_new_reached, tasks))
    need = {loc: results[2 * k + 1] - results[2 * k] for k, loc in enumerate(dead)}
    return saved["states"][richest], sorted(dead, key=lambda loc: -need[loc]), need


def main():
    run, byte = sys.argv[1], int(sys.argv[2])
    candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    probes = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    _, ranked, need = needs(emulator, Path("runs") / run, byte, candidates, probes)
    door = [k for k, loc in enumerate(ranked) if loc[:3] == (0, 5, 8)]
    print(f"{len(ranked)} candidates x {probes} probes each way; door-room ranks {door[:4]} "
          f"(need {[need[ranked[k]] for k in door[:4]]}); top 6: {[(loc[:3], need[loc]) for loc in ranked[:6]]}")


if __name__ == "__main__":
    main()
