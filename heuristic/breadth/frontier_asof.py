"""Hindsight check for graph scores: at time T (a share of the way to the first success), score only saves made before T,
counting only children/descendants made before T. Stone AUC averaged over runs."""
import glob
from pathlib import Path
import numpy as np
from emulator import state_blocks
from bench import WALLS
from heuristic.breadth.common import load_run, auc, ancestors

out = {}
for run in sorted(glob.glob("runs/bench/*/s*/*/")):
    run = Path(run)
    if not (run / "checkpoint.pkl.z").exists():
        continue
    saved = load_run(run)
    states, parents = saved["states"], saved["parents"]
    b = state_blocks(states[0])
    mem = lambda s: np.concatenate([np.frombuffer(s[b["LRAM"][0]:sum(b["LRAM"])], np.uint8), np.frombuffer(s[b["SRAM"][0]:sum(b["SRAM"])], np.uint8)])
    first = next((i for i, s in enumerate(states) if WALLS[run.name][1](mem(s))), None)
    if first is None or first < 10 or first > 3000:
        continue
    chain = set(ancestors(parents, first))
    for share in [0.25, 0.5, 0.75]:
        T = int(first * share)
        children, subtree = np.zeros(T), np.zeros(T)
        for i in range(T):
            p = parents[i]
            if p is not None and 0 <= p < T:
                children[p] += 1
            for a in ancestors(parents, i):
                if a < T:
                    subtree[a] += 1
        label = np.array([i in chain for i in range(T)])
        out.setdefault(("children", share), []).append(auc(children, label))
        out.setdefault(("descendants", share), []).append(auc(subtree, label))
        out.setdefault(("found-new rate (end of run)", share), []).append(auc(np.array(saved["productive"][:T]) / np.maximum(np.array(saved["times_chosen"][:T], float), 1), label))
for k, v in out.items():
    print(f"{k[0]:30s} as of {k[1]:.0%} of the way: stone AUC {np.nanmean(v):.2f} over {np.sum(~np.isnan(v))} runs")
