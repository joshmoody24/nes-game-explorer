"""Frontier ideas: per-save scores, AUC for 'is a stepping stone' and 'is the first successful save', per run, averaged.
Usage: python -m heuristic.breadth.frontier [max saves per run]"""
import glob, sys, time, zlib
from collections import Counter
from pathlib import Path
import numpy as np
from emulator import state_blocks
from bench import WALLS
from heuristic.breadth.common import load_run, auc

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
rng = np.random.default_rng(0)
RECENT = 50


def arrays(states):
    b = state_blocks(states[0])
    sl = lambda t: slice(b[t][0], b[t][0] + b[t][1])
    buf = lambda t: np.stack([np.frombuffer(s[sl(t)], np.uint8) for s in states])
    return np.concatenate([buf("LRAM"), buf("SRAM")], 1), buf("NTAB")


def zlib_novelty(rows, dict_size, sample):
    """Bytes zlib needs for row i, primed with `sample` random earlier rows as the dictionary (newest last)."""
    out = np.zeros(len(rows))
    for i in range(len(rows)):
        earlier = rng.choice(i, min(sample, i), replace=False) if i else []
        zdict = b"".join(rows[j].tobytes() for j in sorted(earlier))[-dict_size:]
        c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, zdict) if zdict else zlib.compressobj(9)
        out[i] = len(c.compress(rows[i].tobytes()) + c.flush())
    return out


def nearest_ncd(rows, sample):
    """Smallest compressed size of row i given any single earlier row (from a random sample) as dictionary."""
    out = np.zeros(len(rows))
    for i in range(len(rows)):
        x = rows[i].tobytes()
        best = len(zlib.compress(x, 9))
        for j in (rng.choice(i, min(sample, i), replace=False) if i else []):
            c = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, rows[j].tobytes())
            best = min(best, len(c.compress(x) + c.flush()))
        out[i] = best
    return out


def value_surprise(rows):
    """Sum over bytes of -log P(value) under per-byte value counts of earlier saves (add-one smoothing)."""
    counts = np.zeros((rows.shape[1], 256), np.int32)
    cols = np.arange(rows.shape[1])
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        out[i] = -np.log((counts[cols, r] + 1) / (i + 256)).sum()
        counts[cols, r] += 1
    return out


def knn_hamming(rows, sample):
    out = np.zeros(len(rows))
    for i in range(1, len(rows)):
        earlier = rows[rng.choice(i, min(sample, i), replace=False)]
        out[i] = (earlier != rows[i]).sum(1).min()
    return out


def graph_scores(parents, chosen, tiles):
    n = len(parents)
    par = np.array([p if p is not None and p >= 0 else -1 for p in parents])
    children = Counter(par[par >= 0])
    unexpanded = Counter(par[(par >= 0) & (chosen == 0)])
    gen = np.zeros(n, int); since_cut = np.zeros(n, int); subtree = np.ones(n)
    def depth(i, memo):
        seen = []
        while i >= 0 and i not in memo and len(seen) < n:
            seen.append(i); i = par[i]
        base = memo.get(i, (-1, -1)) if i >= 0 else (-1, -1)
        for j in reversed(seen):
            p = par[j]
            g = base[0] + 1
            s = 0 if p < 0 or tiles[p] != tiles[j] else base[1] + 1
            memo[j] = base = (g, s)
        return memo
    memo = {}
    for i in range(n):
        if i not in memo:
            depth(i, memo)
    gen = np.array([memo[i][0] for i in range(n)], float)
    since_cut = np.array([memo[i][1] for i in range(n)], float)
    for i in sorted(range(n), key=lambda i: -gen[i]):
        if par[i] >= 0:
            subtree[par[i]] += subtree[i]
    siblings = np.array([children[par[i]] if par[i] >= 0 else 0 for i in range(n)], float)
    ch = np.array([children[i] for i in range(n)], float)
    return {
        "7 children": ch,
        "7 unexpanded children (never chosen)": np.array([unexpanded[i] for i in range(n)], float),
        "7 share of children unexpanded": np.array([unexpanded[i] for i in range(n)], float) / np.maximum(ch, 1),
        "7 generation depth": gen,
        "7 generations since tile bag changed (fewer=higher)": -since_cut,
        "7 subtree size (descendants)": subtree,
        "7 leaf (no children)": (ch == 0).astype(float),
        "7 siblings (fewer=higher)": -siblings,
    }


results = {}
started = time.time()
for run in sorted(glob.glob("runs/bench/*/s*/*/")):
    run = Path(run)
    if not (run / "checkpoint.pkl.z").exists():
        continue
    saved = load_run(run)
    states, parents = saved["states"], saved["parents"]
    ram, ntab = arrays(states)
    test = WALLS[run.name][1]
    first = next((i for i in range(len(states)) if test(ram[i])), None)
    if first is None or first < 10 or first > LIMIT:
        print(f"skip {run} (first={first})", flush=True); continue
    chain, i = set(), parents[first]
    while i is not None and i >= 0 and i not in chain:
        chain.add(i); i = parents[i]
    n = first + 1
    ram, ntab = ram[:n], ntab[:n]
    stone = np.array([i in chain for i in range(n)])
    chosen = np.array(saved["times_chosen"][:n], float)
    tiles = [frozenset(np.flatnonzero(np.bincount(t[np.r_[0:960, 1024:1984]], minlength=256) >= 32)) for t in ntab]
    tile_count = Counter(tiles)
    seen_tiles, tile_rarity = Counter(), np.zeros(n)
    for i, t in enumerate(tiles):
        tile_rarity[i] = -seen_tiles[t]; seen_tiles[t] += 1
    scores = {
        "base times chosen": chosen,
        "base found-new rate": np.array(saved["productive"][:n], float) / np.maximum(chosen, 1),
        "base path length": np.array([len(p) for p in saved["paths"][:n]], float),
        "base creation order": np.arange(n, dtype=float),
        "5 zlib system RAM | 15 earlier": zlib_novelty(ram[:, :0x800], 32768, 15),
        "5 zlib full RAM | 3 earlier": zlib_novelty(ram, 32768, 3),
        "5 zlib nametable | 15 earlier": zlib_novelty(ntab, 32768, 15),
        "5 zlib full RAM, nearest of 20 earlier": nearest_ncd(ram, 20),
        "8 byte-value surprise (full RAM)": value_surprise(ram),
        "ref kNN Hamming RAM (200 earlier)": knn_hamming(ram, 200),
        "1+ tile-bag rarity among earlier saves": tile_rarity,
        **graph_scores(parents[:n], chosen, tiles),
    }
    for k, v in scores.items():
        inner = np.array([c > 0 for c in scores["7 children"][:first]])
        recent = slice(max(0, first - RECENT), n)
        results.setdefault(k, []).append((auc(v[:first], stone[:first]), auc(v[:first][inner], stone[:first][inner]),
                                          auc(v, np.arange(n) == first), auc(v[recent], np.arange(n)[recent] == first)))
    print(f"{run}: {n} saves, {stone.sum()} stones, {time.time() - started:.0f}s", flush=True)

print(f"\n{len(results['base times chosen'])} runs; mean AUC: stones (min..max) | stones among saves with children | "
      f"first success among all saves | among the last {RECENT}")
for k, v in sorted(results.items(), key=lambda kv: -np.nanmean([a[0] for a in kv[1]])):
    a = np.array(v)
    print(f"  {k:52s} {np.nanmean(a[:, 0]):.2f} ({np.nanmin(a[:, 0]):.2f}..{np.nanmax(a[:, 0]):.2f}) | {np.nanmean(a[:, 1]):.2f} | "
          f"{np.nanmean(a[:, 2]):.2f} | {np.nanmean(a[:, 3]):.2f}")
