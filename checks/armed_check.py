"""
Is the exploration wall a budget problem or a coordination problem? From saved states holding 5+ missiles,
runs Go-Explore-style random segments (sticky, 100 steps) and counts segments that reach a room, item,
area or missile capacity the archive has never seen. Hand-made Metroid knowledge, diagnostic only.

Usage: python -m checks.armed_check <run name> [states] [segments per state]
"""

import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from metroid import CURRENT_MISSILES, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, area_of, at


def progress_marks(memory):
    """Rooms, items, areas and missile capacity a state shows."""
    return {("room", area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X])), ("items", int(memory[at(ITEMS)])),
            ("area", area_of(memory)), ("max missiles", int(memory[at(MAX_MISSILES)]))}


def _segments(task):
    state, seeds, known = task
    emulator = workers.emulator()
    found = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        emulator.load(state)
        action, new = int(rng.integers(len(ACTIONS))), set()
        for _ in range(100):
            if rng.random() > 0.9:
                action = int(rng.integers(len(ACTIONS)))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            new |= progress_marks(emulator.memory()) - known
        found.append(new)
    return found


def main():
    saved = pickle.loads(zlib.decompress((Path("runs") / sys.argv[1] / "checkpoint.pkl.z").read_bytes()))
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    per_state = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    from emulator import Emulator
    import os
    from explorers import ROM
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    memories = [emulator.memory(s) for s in saved["states"]]
    known = set().union(*(progress_marks(m) for m in memories))
    armed = [i for i, m in enumerate(memories) if m[at(CURRENT_MISSILES)] >= 5]
    picks = np.random.default_rng(0).choice(armed, min(count, len(armed)), replace=False)
    tasks = [(saved["states"][i], [int(i) * 1000 + k for k in range(per_state)], known) for i in picks]
    results = [segment for state_result in workers.map(_segments, tasks) for segment in state_result]
    hits = [r for r in results if r]
    print(f"{len(armed)} armed states of {len(memories)}; {len(results)} segments from {len(picks)} of them: "
          f"{len(hits)} ({len(hits) / len(results):.2%}) reached something new")
    discovered = {}
    for r in hits:
        for mark in r:
            discovered[mark] = discovered.get(mark, 0) + 1
    for mark, times in sorted(discovered.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {mark}: {times} segments")


if __name__ == "__main__":
    main()
