"""
Position bytes found without knowing the game: from many saved states, hold one direction for a few steps in one
copy and the opposite direction in another; bytes that end up different with a consistent sign (say, higher
after RIGHT than after LEFT) carry position along that axis. Shows where Metroid's known position bytes rank.

Usage: python -m checks.position_bytes_check [states]
"""

import os
import pickle
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import library
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM
from metroid import ROOM_X, ROOM_Y

HOLD = 8


def after(emulator, state, buttons):
    emulator.load(state)
    for _ in range(HOLD):
        emulator.run(buttons, FRAMES_PER_STEP)
    return emulator.memory().astype(int)


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    random.seed(0)
    picks = random.sample(library.entries(), count)
    by_run = defaultdict(list)
    for e in picks:
        by_run[e["run"]].append(e)
    states = []
    for run, group in by_run.items():
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        states += [saved["states"][e["slot"]] for e in group]
    for axis, (plus, minus), known in [("horizontal", ("RIGHT", "LEFT"), ROOM_X), ("vertical", ("DOWN", "A"), ROOM_Y)]:
        signs = np.stack([np.sign(after(emulator, s, button_mask(plus)) - after(emulator, s, button_mask(minus)))
                          for s in states])
        consistency = np.abs(signs.mean(0))  # 1: always the same direction of difference; 0: no relation
        moved = (signs != 0).mean(0)
        score = consistency * moved
        ranked = list(np.argsort(-score))
        print(f"{axis} ({plus} vs {minus}): top bytes {[(int(b), round(float(score[b]), 2)) for b in ranked[:6]]}; "
              f"known position byte {known:#06x} ranks {ranked.index(known)} (score {score[known]:.2f})", flush=True)


if __name__ == "__main__":
    main()
