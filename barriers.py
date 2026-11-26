"""
Where to bring resources: barriers. A barrier is a location exploration keeps trying and never gets past
(no kept visit there has descendants) and has only ever reached poorer than the richest state in the archive:
the kind of spot where something you could carry there might be needed (a red door reached many times
without missiles). Ranked by how often exploration tried it.
"""

import pickle
import zlib
from collections import defaultdict
from pathlib import Path

from graph import location


def find(emulator, run_dir, wealth):
    """(richest visit's state, barrier locations, most tried first). wealth(memory) -> comparable."""
    saved = pickle.loads(zlib.decompress((Path(run_dir) / "checkpoint.pkl.z").read_bytes()))
    memories = [emulator.memory(s) for s in saved["states"]]
    richest = max(range(len(memories)), key=lambda i: wealth(memories[i]))
    locations = [location(m) for m in memories]
    best_held, tried, has_children = {}, defaultdict(int), set()
    for loc, m, times in zip(locations, memories, saved["times_chosen"]):
        best_held[loc] = max(best_held.get(loc, wealth(m)), wealth(m))
        tried[loc] += times
    for parent in saved["parents"]:
        if parent >= 0:
            has_children.add(locations[parent])
    rich = wealth(memories[richest])
    barriers = [loc for loc in best_held if loc not in has_children and best_held[loc] < rich]
    return saved["states"][richest], sorted(barriers, key=lambda loc: -tried[loc])
