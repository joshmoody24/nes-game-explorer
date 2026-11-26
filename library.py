"""
Every state any run has kept, in one index, so experiments can start next to any wall instead of re-finding
it. The index stores a small summary per state (where it lives, what it holds) and fetches the state itself
on demand. The summary reads named Metroid bytes: a catalog for us, not something the method sees.

Usage: python library.py build          (scans runs/, writes cache/library.pkl)
       python library.py show           (counts by items held)
"""

import os
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path

from emulator import Emulator
from explorers import ROM
from metroid import CURRENT_MISSILES, ENERGY_TANKS, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, area_of, at, energy_of
from progress import ITEM_BITS

INDEX = Path("cache/library.pkl")


def summary(view):
    return {"items": int(view[at(ITEMS)]), "tanks": int(view[at(ENERGY_TANKS)]),
            "max missiles": int(view[at(MAX_MISSILES)]), "missiles": int(view[at(CURRENT_MISSILES)]),
            "energy": energy_of(view), "room": (area_of(view), int(view[MAP_Y]), int(view[MAP_X]))}


def build():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    entries = []
    for checkpoint in sorted(Path("runs").rglob("checkpoint.pkl.z")):
        try:
            saved = pickle.loads(zlib.decompress(checkpoint.read_bytes()))
        except Exception as error:
            print(f"skipping {checkpoint}: {error}")
            continue
        paths = saved.get("paths") or [None] * len(saved["states"])
        for slot, (state, path) in enumerate(zip(saved["states"], paths)):
            entries.append({"run": str(checkpoint.parent), "slot": slot,
                            "path length": len(path) if path else None, **summary(emulator.view(state))})
        print(f"{checkpoint.parent}: {len(saved['states'])} states", flush=True)
    INDEX.write_bytes(pickle.dumps(entries))
    print(f"{len(entries):,} states from {len({e['run'] for e in entries})} runs -> {INDEX}")


def entries():
    return pickle.loads(INDEX.read_bytes())


def fetch(entry):
    """The saved state an index entry points to."""
    saved = pickle.loads(zlib.decompress((Path(entry["run"]) / "checkpoint.pkl.z").read_bytes()))
    return saved["states"][entry["slot"]]


def names(items):
    return tuple(name for bit, name in ITEM_BITS.items() if items & bit)


def show():
    counts = Counter((names(e["items"]), e["tanks"], e["max missiles"]) for e in entries())
    for (items, tanks, capacity), n in sorted(counts.items(), key=lambda kv: (-len(kv[0][0]), -kv[0][1], -kv[1])):
        print(f"{n:>8} states: items {list(items)}, tanks {tanks}, max missiles {capacity}")


if __name__ == "__main__":
    {"build": build, "show": show}[sys.argv[1]]()
