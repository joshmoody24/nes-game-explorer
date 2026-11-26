"""
Where to bring resources: barriers. A barrier is a location exploration keeps trying and never gets past
(no kept visit there has descendants), where tries mostly survive (a spot you die at is a death trap, not a
barrier), and that has only ever been reached poorer than the richest state in the archive:
the kind of spot where something you could carry there might be needed (a red door reached many times
without missiles). Ranked by how often exploration tried it.
"""

import pickle
import zlib
from collections import defaultdict
from pathlib import Path

from graph import location

MOST_DEATHS = 0.5  # share of tries from a spot that may end in game over for it to count as a barrier


def find(emulator, run_dir, wealth):
    """(richest visit's state, barrier locations, most tried first). wealth(memory) -> comparable."""
    saved = pickle.loads(zlib.decompress((Path(run_dir) / "checkpoint.pkl.z").read_bytes()))
    memories = [emulator.memory(s) for s in saved["states"]]
    richest = max(range(len(memories)), key=lambda i: wealth(memories[i]))
    locations = [location(m) for m in memories]
    best_held, tried, died, has_children = {}, defaultdict(int), defaultdict(int), set()
    deaths = saved.get("deaths") or [0] * len(memories)
    for loc, m, times, lost in zip(locations, memories, saved["times_chosen"], deaths):
        best_held[loc] = max(best_held.get(loc, wealth(m)), wealth(m))
        tried[loc] += times
        died[loc] += lost
    for parent in saved["parents"]:
        if parent >= 0:
            has_children.add(locations[parent])
    rich = wealth(memories[richest])
    barriers = [loc for loc in best_held if loc not in has_children and best_held[loc] < rich
                and died[loc] <= MOST_DEATHS * tried[loc]]
    return saved["states"][richest], sorted(barriers, key=lambda loc: -tried[loc])
