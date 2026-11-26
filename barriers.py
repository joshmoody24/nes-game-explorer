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

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from graph import location
from metroid import GAME_MODE, GAME_OVER

MOST_DEATHS = 0.5  # share of tries from a spot that may end in game over for it to count as a barrier


def _death_rate(task):
    """In a worker: share of random sticky segments from state that end in game over (diagnostic check)."""
    state, seeds, steps = task
    emulator, deaths = workers.emulator(), 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        emulator.load(state)
        action = int(rng.integers(len(ACTIONS)))
        for _ in range(steps):
            if rng.random() > 0.9:
                action = int(rng.integers(len(ACTIONS)))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            if emulator.view(emulator.save())[GAME_MODE] == GAME_OVER:
                deaths += 1
                break
    return deaths / len(seeds)


def find(emulator, run_dir, wealth, candidates=60, probes=20, steps=100):
    """
    (richest visit's state, barrier locations, most tried first). wealth(memory) -> comparable. Archives
    without death counts get them estimated: probes random segments from the best visit at each of the most
    tried candidates.
    """
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
    poor_dead_ends = sorted((loc for loc in best_held if loc not in has_children and best_held[loc] < rich),
                            key=lambda loc: -tried[loc])
    if saved.get("deaths"):
        barriers = [loc for loc in poor_dead_ends if died[loc] <= MOST_DEATHS * tried[loc]]
    else:
        visit_at = {loc: i for i, loc in enumerate(locations)}
        top = poor_dead_ends[:candidates]
        rates = list(workers.map(_death_rate, [(saved["states"][visit_at[loc]], list(range(probes)), steps)
                                                for loc in top]))
        barriers = [loc for loc, rate in zip(top, rates) if rate <= MOST_DEATHS]
    return saved["states"][richest], barriers
