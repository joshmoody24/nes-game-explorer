"""Which general per-save features separate stepping stones from other saves (AUC per run, averaged)."""
import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from bench import WALLS

def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    results = {}
    for run in sys.argv[1:]:
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        states, parents, places = saved["states"], saved["parents"], saved["places"]
        test = WALLS[Path(run).name][1]
        first = next((i for i, s in enumerate(states) if test(emulator.view(s))), None)
        if first is None: continue
        chain, i = set(), parents[first]
        while i is not None and i >= 0 and i not in chain:
            chain.add(i); i = parents[i]
        n = first
        label = np.array([i in chain for i in range(n)])
        chosen = np.array(saved["times_chosen"][:n], float)
        productive = np.array(saved["productive"][:n], float)
        deaths = np.array(saved["deaths"][:n], float)
        paths = saved["paths"]
        room = lambda p: tuple(p[1]) if p[0] == "record" else tuple(p[:3])
        rooms = Counter(room(p) for p in places[:n])
        same_place = Counter(tuple(p) for p in places[:n])
        children = Counter(p for p in parents[:n] if p is not None and p >= 0)
        holdings = np.array([sum(h) if h else 0 for h in saved["holdings"][:n]], float)
        feats = {
            "found-new rate (productive/tries)": productive / np.maximum(chosen, 1),
            "times chosen": chosen,
            "productive tries": productive,
            "children (saves it created)": np.array([children[i] for i in range(n)], float),
            "path length": np.array([len(paths[i]) if paths[i] is not None else 0 for i in range(n)], float),
            "creation order (newer = higher)": np.arange(n, dtype=float),
            "room crowding (fewer = higher)": -np.array([rooms[room(places[i])] for i in range(n)], float),
            "place crowding (fewer = higher)": -np.array([same_place[tuple(places[i])] for i in range(n)], float),
            "deaths per try (fewer = higher)": -deaths / np.maximum(chosen, 1),
            "holdings": holdings,
            "is record place": np.array([places[i][0] == "record" for i in range(n)], float),
        }
        for k, v in feats.items():
            results.setdefault(k, []).append(auc(v, label))
    print(f"{len(results['holdings'])} runs; AUC (0.5 = chance, 1 = stones always ranked first):")
    for k, v in sorted(results.items(), key=lambda kv: -np.nanmean(kv[1])):
        v = np.array(v); print(f"  {np.nanmean(v):.2f}  (min {np.nanmin(v):.2f}, max {np.nanmax(v):.2f})  {k}")
