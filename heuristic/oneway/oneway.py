"""
One-way moves: for sampled saves of successful runs, can quick random tries get back to the parent save's place?
A save nobody can get back from is "one-way". AUC of one-way-ness for "is a stepping stone".
Usage: TURBO=1 python heuristic/oneway/oneway.py <run dir> ...
"""
import os, pickle, sys, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM, Settings, hand_made_place
from bench import WALLS
from heuristic.stones.features import auc

TRIES, STEPS, SAMPLE = 10, 150, 60
SETTINGS = Settings(keep_more="4217,263")

def location(memory):
    return tuple(hand_made_place(memory, SETTINGS)[:5])

def returns(args):
    state, target, seed = args
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng(seed)
    back = 0
    for _ in range(TRIES):
        emulator.load(state); a = rng.integers(len(ACTIONS)); keep = rng.uniform(0.9, 0.99)
        for _ in range(STEPS):
            if rng.random() > keep: a = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[a], FRAMES_PER_STEP)
            if location(emulator.memory()) == target:
                back += 1; break
    return back / TRIES

if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    aucs = []
    with ProcessPoolExecutor(5) as pool:
        for run in sys.argv[1:]:
            saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
            states, parents = saved["states"], saved["parents"]
            test = WALLS[Path(run).name][1]
            first = next((i for i, s in enumerate(states) if test(emulator.view(s))), None)
            if first is None: continue
            stones, i = set(), parents[first]
            while i is not None and i >= 0 and i not in stones:
                stones.add(i); i = parents[i]
            rng = np.random.default_rng(0)
            others = [i for i in range(1, first) if i not in stones and parents[i] is not None and parents[i] >= 0]
            sample = sorted(s for s in stones if parents[s] is not None and parents[s] >= 0) + list(rng.choice(others, min(SAMPLE, len(others)), replace=False))
            targets = [location(emulator.view(states[parents[i]])) for i in sample]
            back = np.array(list(pool.map(returns, [(states[i], t, k) for k, (i, t) in enumerate(zip(sample, targets))])))
            label = np.array([i in stones for i in sample])
            a = auc(1 - back, label); aucs.append(a)
            print(f"{run}: {label.sum()} stones vs {(~label).sum()} others; return rate stones {back[label].mean():.2f}, others {back[~label].mean():.2f}; one-way AUC {a:.2f}", flush=True)
    print(f"mean one-way AUC over {len(aucs)} runs: {np.nanmean(aucs):.2f}")
