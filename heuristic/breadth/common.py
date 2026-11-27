"""Shared loading, labels and yardsticks for the breadth experiments."""
import os, pickle, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from emulator import state_blocks

HERE = Path(__file__).parent
CACHE = HERE / "cache"


def load_run(run):
    return pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))


RUNS = ["runs/bench/fert1/s1/lava-corridor", "runs/bench/fert1/s1/ridley-door", "runs/bench/fert1/s1/ridley-exit",
        "runs/bench/q2-plain/s2/lava-corridor", "runs/bench/q2-plain/s2/ridley-exit",
        *[f"runs/bench/prefix-laptop/s2/{w}" for w in ["escape", "ice-beam", "missile-door", "mother-brain", "tourian-entry", "zebetites"]]]


def nametable_slice(state):
    start, size = state_blocks(state)["NTAB"]
    return slice(start, start + size)


def palette_slice(state):
    start, _ = state_blocks(state)["PPUR"]
    return slice(start + 12, start + 44)


def chr_slice(state):
    start, size = state_blocks(state)["CHRR"]
    return slice(start, start + size)


def room_label(memory):
    """Ground truth for evaluation only."""
    return (int(memory[0x74]) & 15, int(memory[0x4F]), int(memory[0x50]))


def room_scores(ids, labels):
    """purity: share of states whose id's majority room is theirs; completeness: share whose room's majority id is
    theirs; ids per room: mean distinct ids per true room."""
    by_id, by_room = defaultdict(Counter), defaultdict(Counter)
    for i, l in zip(ids, labels):
        by_id[i][l] += 1
        by_room[l][i] += 1
    n = len(ids)
    purity = sum(c.most_common(1)[0][1] for c in by_id.values()) / n
    completeness = sum(c.most_common(1)[0][1] for c in by_room.values()) / n
    return purity, completeness, np.mean([len(c) for c in by_room.values()])


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ancestors(parents, i):
    """Parent, grandparent, ... of save i. Slot reuse can make parent links loop, so each save is visited once."""
    seen = {i}
    while (p := parents[i]) is not None and p >= 0 and p not in seen:
        seen.add(p)
        yield p
        i = p
