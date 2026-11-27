"""
Stepping stones: in a fresh run that broke a wall, the chain of parents from the first successful save back to the
start. How much of the run's effort went to them, and how they differ from other saves.
Usage: python heuristic/stones/stones.py runs/bench/<name>/s<seed>/<wall> ...
"""
import os, pickle, sys, zlib
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from bench import WALLS

if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run in sys.argv[1:]:
        wall = Path(run).name
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        states, parents, chosen = saved["states"], saved["parents"], np.array(saved["times_chosen"], float)
        test = WALLS[wall][1]
        first = next((i for i, s in enumerate(states) if test(emulator.view(s))), None)
        if first is None:
            continue
        chain, i = [], first
        while i is not None and i >= 0 and i not in chain:
            chain.append(i)
            i = parents[i]
        stones = sorted(set(chain) - {first})
        n = len(states)
        tries = chosen.sum()
        stone_tries = chosen[stones].sum()
        index_frac = np.array(stones) / max(first, 1)
        rank = lambda arr, i: (arr[:first + 1] > arr[i]).mean()
        print(f"{run}: {n} saves, success at {first}; {len(stones)} stepping stones got {stone_tries / tries:.1%} of {int(tries)} tries "
              f"(fair share {len(stones) / (first + 1):.1%}); stones' tries rank (share of saves chosen more): "
              f"{np.mean([rank(chosen, i) for i in stones]):.2f}; created at {np.round(index_frac, 2).tolist()[:12]}")
