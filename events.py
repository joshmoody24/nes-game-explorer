"""
Lasting acquisitions found without knowing the game. An event is a set of one-way bits (seen turning on but
never off, over every step of a run; see progress.see_change) that the layer's start state lacks and that
exactly the same kept visits hold: picking something up flips several such bits at once (the item, the
game's own record of it), and every later visit in that lineage keeps them. Each event's best holder is a
natural start for the next layer.
"""

import json
import pickle
import zlib
from pathlib import Path

import numpy as np

SMALLEST_SHARE, LARGEST_SHARE = 0.02, 0.9  # held by too few: junk; by nearly all: part of the layer itself


def find(emulator, run_dir):
    """[(bits, share of visits holding them, holder visit indices)], largest events first."""
    run_dir = Path(run_dir)
    flips = np.load(run_dir / "bit_flips.npz")
    one_way = np.flatnonzero(np.unpackbits(flips["ever_up"] & ~flips["ever_down"]))
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    start_file = json.loads((run_dir / "config.json").read_text()).get("start_state")
    start = Path(start_file).read_bytes() if start_file else saved["states"][0]
    bits = np.unpackbits(np.stack([emulator.memory(s) for s in saved["states"]]), axis=1)[:, one_way].astype(bool)
    gained = ~np.unpackbits(emulator.memory(start))[one_way].astype(bool)
    share = bits.mean(0)
    candidates = np.flatnonzero(gained & (share >= SMALLEST_SHARE) & (share <= LARGEST_SHARE))
    groups = {}
    for k in candidates:
        groups.setdefault(np.packbits(bits[:, k]).tobytes(), []).append(int(one_way[k]))
    found = [(group, float(share[np.searchsorted(one_way, group[0])]),
              np.flatnonzero(bits[:, np.searchsorted(one_way, group[0])])) for group in groups.values()]
    return sorted(found, key=lambda event: -len(event[0]))
