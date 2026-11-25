"""
Can a layer's lasting acquisitions be found without knowing the game? Uses the run's every-step record of
bit flips (bit_flips.npz): one-way bits are bits seen turning on but never off. Scores each kept visit by the
one-way bits it gained since the layer's start state, each scored by its rarest gained bit
(a fresh acquisition is held by few), and shows what the top visits actually hold (Metroid items:
diagnostic only).

Usage: python -m checks.trigger_check <run name>
"""

import json
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from emulator import Emulator
from explorers import ROM
from metroid import ITEMS, MAX_MISSILES, at
from progress import ITEM_BITS


def main():
    run_dir = Path("runs") / sys.argv[1]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    flips = np.load(run_dir / "bit_flips.npz")
    one_way = np.flatnonzero(np.unpackbits(flips["ever_up"] & ~flips["ever_down"]))
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    start_file = json.loads((run_dir / "config.json").read_text()).get("start_state")
    start = Path(start_file).read_bytes() if start_file else saved["states"][0]
    memories = np.stack([emulator.memory(s) for s in saved["states"]])
    bits = np.unpackbits(memories, axis=1)[:, one_way].astype(bool)
    gained = bits & ~np.unpackbits(emulator.memory(start))[one_way].astype(bool)
    held_by = bits.mean(0)
    # A lasting acquisition is one bit few visits hold yet; summing over many gained bits lets junk drown it.
    score = np.where(gained, -np.log(held_by + 1e-3), 0).max(1)
    print("rarest one-way bits (byte.bit: share of visits holding it): " + ", ".join(
        f"{one_way[k] // 8:#06x}.{7 - one_way[k] % 8}: {held_by[k]:.2%}" for k in np.argsort(held_by)[:8] if held_by[k] > 0))
    names = lambda m: [n for b, n in ITEM_BITS.items() if m[at(ITEMS)] & b] + [f"max missiles {m[at(MAX_MISSILES)]}"]
    print(f"{len(one_way)} one-way bits over the whole run; start holds {names(emulator.memory(start))}")
    for i in np.argsort(-score)[:5]:
        print(f"  score {score[i]:.1f}: {names(memories[i])}")
    item_bits = [at(ITEMS) * 8 + (7 - b) for b in range(8)]
    print(f"  item bits among the one-way bits: {[7 - (b % 8) for b in one_way if b in item_bits]}")


if __name__ == "__main__":
    main()
