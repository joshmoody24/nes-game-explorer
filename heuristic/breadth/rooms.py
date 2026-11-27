"""Room ideas scored on the shared room test (cache/rooms.npz): purity / completeness / ids per room."""
import time
import numpy as np
T0 = time.time()
from collections import defaultdict
from heuristic.breadth.common import RUNS, CACHE, load_run, ancestors, room_scores, nametable_slice

d = np.load(CACHE / "rooms.npz")
labels = [tuple(map(int, x)) for x in d["label"]]
areas = [l[0] for l in labels]
traj, ntab, ram, pal = d["traj"], d["ntab"], d["ram"], d["pal"]
TILES = np.r_[0:960, 1024:1984]
hist = np.stack([np.bincount(n[TILES], minlength=256) for n in ntab])
saves, steps = traj < 0, traj >= 0
rows = []


def granularity(ids, runs):
    """Per run, averaged: distinct ids, effective number of ids (exp of entropy), share of states in the biggest id."""
    stats = []
    for run in np.unique(runs):
        _, counts = np.unique(ids[runs == run], return_counts=True)
        q = counts / counts.sum()
        stats.append((len(counts), np.exp(-(q * np.log(q)).sum()), q.max()))
    return np.mean(stats, 0)


def report(name, ids, where=None, area_level=False):
    ids = np.asarray([hash(x) if not isinstance(x, (int, np.integer)) else x for x in ids])
    for subset, mask in [("saves", saves), ("steps", steps)]:
        m = mask if where is None else mask & where
        if not m.any():
            continue
        truth = areas if area_level else labels
        p, c, k = room_scores(ids[m], [truth[i] for i in np.flatnonzero(m)])
        g = granularity(ids[m], d["run"][m])
        rows.append((name, subset, p, c, k, len(set(ids[m])), *g))
        print(f"{name:58s} {subset:5s} {p:6.1%} {c:6.1%} {k:8.1f} {len(set(ids[m])):6d} {g[0]:7.0f} {g[1]:7.0f} {g[2]:6.0%}  ({time.time() - T0:.0f}s)", flush=True)


report("one id per run", d["run"])

# Baseline from checks/scene_bytes_check.py, refit on these paths: bytes that change at many scene changes, rarely otherwise.
changes = np.concatenate([np.diff(ram[traj == t].astype(np.int16), axis=0) != 0 for t in np.unique(traj[steps])])
share = changes.mean(1)
scene = share >= np.quantile(share, 0.995)
at_scene, elsewhere = changes[scene].mean(0), changes[~scene].mean(0)
for min_scene, max_else in [(0.02, 0.001), (0.05, 0.0005), (0.1, 0.0002)]:
    where = np.flatnonzero((at_scene >= min_scene) & (elsewhere <= max_else))
    report(f"base scene-change bytes ({min_scene:.0%}, {max_else:.2%}): {len(where)} B", [m[where].tobytes() for m in ram])
report("true area (reference)", areas)
report("true area, area level", areas, area_level=True)

for k in [8, 32, 96]:
    report(f"1 tile bag: tiles with count>={k}", [frozenset(np.flatnonzero(h >= k)) for h in hist])
for n in [4, 8, 16]:
    report(f"1 tile bag: top-{n} tiles", [frozenset(np.argsort(-h, kind="stable")[:n]) for h in hist])
for q in [16, 64]:
    report(f"1 tile bag: counts rounded to /{q} (log)", [tuple(np.round(np.log2(1 + h / q)).astype(int)) for h in hist])



def leader_clusters(present, threshold):
    """Greedy: each state joins the first earlier leader whose tile set overlaps it by Jaccard >= threshold."""
    leaders, ids = [], np.zeros(len(present), np.int64)
    for i, x in enumerate(present):
        if leaders:
            L = np.array(leaders)
            jac = (L & x).sum(1) / np.maximum((L | x).sum(1), 1)
            best = int(np.argmax(jac))
            if jac[best] >= threshold:
                ids[i] = best
                continue
        leaders.append(x)
        ids[i] = len(leaders) - 1
    return ids


for k in [4, 16]:
    for threshold in [0.7, 0.85]:
        report(f"1 tile bag: leader clusters, tiles>={k}, Jaccard>={threshold}", leader_clusters(hist >= k, threshold))
ATTRIBUTES = np.r_[960:1024, 1984:2048]
report("1b attribute-byte bag (count>=8)", [frozenset(np.flatnonzero(np.bincount(n[ATTRIBUTES], minlength=256) >= 8)) for n in ntab])

report("2 palette (32 B)", [p.tobytes() for p in pal])
report("2 palette (bg 16 B)", [p[:16].tobytes() for p in pal])
report("2 palette, area level", [p.tobytes() for p in pal], area_level=True)
report("2 palette (bg 16 B), area level", [p[:16].tobytes() for p in pal], area_level=True)
report("2 bg palette + CHR crc, area level", [(p[:16].tobytes(), c) for p, c in zip(pal, d["chr"])], area_level=True)
report("2b CHR-RAM crc", d["chr"])
report("2b CHR-RAM crc, area level", d["chr"], area_level=True)
report("2+1 palette + tiles>=32", [(p.tobytes(), frozenset(np.flatnonzero(h >= 32))) for p, h in zip(pal, hist)])


def scene_cut_ids(threshold, delay, bag):
    """Along each replayed path: a cut is a step where >= threshold RAM bytes change; every state after a cut gets
    the tile bag seen `delay` steps after it (or just before the next cut)."""
    ids = np.full(len(labels), -1, dtype=np.int64)
    for t in np.unique(traj[steps]):
        idx = np.flatnonzero(traj == t)
        changed = (np.diff(ram[idx].astype(np.int16), axis=0) != 0).sum(1)
        cuts = [0] + list(np.flatnonzero(changed >= threshold) + 1) + [len(idx)]
        for a, b in zip(cuts, cuts[1:]):
            if b > a:
                ids[idx[a:b]] = hash(bag(hist[idx[min(a + delay, b - 1)]]))
    return ids


for threshold in [300, 500]:
    for delay in [0, 20, 60]:
        report(f"3 scene cut (>= {threshold} RAM bytes) + tile bag>=32 {delay} steps later",
               scene_cut_ids(threshold, delay, lambda h: frozenset(np.flatnonzero(h >= 32))))
report("3 scene cut (>=500) + tile bag>=8 20 steps later", scene_cut_ids(500, 20, lambda h: frozenset(np.flatnonzero(h >= 8))))

# Deliberately not room-based places.
step_changes = changes.mean(0)
for lo, hi in [(0.00005, 0.0002), (0.0001, 0.0005), (0.0005, 0.005)]:
    where = np.flatnonzero((step_changes >= lo) & (step_changes <= hi))
    report(f"N slow bytes (change on {lo:.2%}..{hi:.1%} of steps): {len(where)} B", [m[where].tobytes() for m in ram])
    report(f"N slow bytes {lo:.2%}..{hi:.1%}, low 2 bits dropped", [(m[where] >> 2).tobytes() for m in ram])
holdings = np.full(len(labels), -1)
for r_, run in enumerate(RUNS):
    held = load_run(run)["holdings"]
    rows_here = np.flatnonzero((d["run"] == r_) & saves)
    holdings[rows_here] = [hash(tuple(held[s]) if held[s] else ()) for s in d["slot"][rows_here]]
top4 = [frozenset(np.argsort(-h, kind="stable")[:4]) for h in hist]
report("N situation: holdings x bg palette x top-4 tiles", [(h, p[:16].tobytes(), t) for h, p, t in zip(holdings, pal, top4)], where=saves)
report("N situation: holdings x bg palette", [(h, p[:16].tobytes()) for h, p in zip(holdings, pal)], where=saves)

# Idea 4: clusters of each run's save tree; scored on the sampled saves.
graph_ids = defaultdict(lambda: np.full(len(labels), -1, dtype=np.int64))
for r, run in enumerate(RUNS):
    saved = load_run(run)
    parents, paths = saved["parents"], saved["paths"]
    n = len(parents)
    length = np.array([len(p) for p in paths])

    def root_of(i, cut):
        seen = {i}
        while (p := parents[i]) is not None and p >= 0 and p not in seen and not cut(p, i):
            seen.add(p)
            i = p
        return i

    def generation(i):
        return sum(1 for _ in ancestors(parents, i))

    gens = np.array([generation(i) for i in range(n)])
    variants = {f"4 tree: cut edges with path jump > {L} steps": (lambda L: lambda p, c: length[c] - length[p] > L)(L)
                for L in [10, 30, 60]}
    for g in [2, 4]:
        variants[f"4 tree: cut every {g}th generation"] = (lambda g: lambda p, c: gens[c] % g == 0)(g)
    bag = [None] * n
    def tiles(i):
        if bag[i] is None:
            nt = np.frombuffer(saved["states"][i][nametable_slice(saved["states"][i])], np.uint8)
            bag[i] = frozenset(np.flatnonzero(np.bincount(nt[TILES], minlength=256) >= 32))
        return bag[i]
    variants["4+1 tree: cut edges whose tile bag>=32 changes"] = lambda p, c: tiles(p) != tiles(c)
    rows_here = np.flatnonzero((d["run"] == r) & saves)
    for name, cut in variants.items():
        for i in rows_here:
            graph_ids[name][i] = r * 100000 + root_of(int(d["slot"][i]), cut)
for name, ids in graph_ids.items():
    report(name, ids, where=saves)

print(f"{'method':58s} {'set':5s} {'purity':>6s} {'compl':>6s} {'ids/room':>8s} {'ids':>6s} {'ids/run':>7s} {'eff/run':>7s} {'top':>6s}")
for name, subset, p, c, k, nids, a, b, top in rows:
    print(f"{name:58s} {subset:5s} {p:6.1%} {c:6.1%} {k:8.1f} {nids:6d} {a:7.0f} {b:7.0f} {top:6.0%}")
