"""
Which bytes does holding a direction move, at short and long horizons? Holding one direction vs the opposite,
then releasing (so button copies drop out); score = how consistently the byte ends up higher on one side x how
often it differs. Shows where Metroid's fine and map position bytes rank, and their typical moves.

Usage: python -m checks.movement_bytes_check [states]
"""

import os
import random
import sys

import numpy as np

import generic_places
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y

RELEASE = 4


def after(emulator, state, buttons, hold):
    emulator.load(state)
    for _ in range(hold):
        emulator.run(buttons, FRAMES_PER_STEP)
    for _ in range(RELEASE):
        emulator.run(0, FRAMES_PER_STEP)
    return emulator.memory().astype(int)


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    random.seed(0)
    states = generic_places.library_states(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
    known = {MAP_X: "map x", MAP_Y: "map y", ROOM_X: "fine x", ROOM_Y: "fine y"}
    for plus, minus in generic_places.AXES:
        for hold in (8, 48):
            diffs = np.stack([after(emulator, s, button_mask(plus), hold) - after(emulator, s, button_mask(minus), hold)
                              for s in states])
            signs = np.sign(diffs)
            score = np.abs(signs.mean(0)) * (signs != 0).mean(0)
            ranked = list(np.argsort(-score))
            top = [(hex(int(b)), round(float(score[b]), 2), int(np.median(np.abs(diffs[:, b][diffs[:, b] != 0]))))
                   for b in ranked[:8]]
            print(f"{plus} vs {minus}, hold {hold}: top (byte, score, median move) {top}", flush=True)
            print("    known: " + ", ".join(f"{name} {b:#x} rank {ranked.index(b)} score {score[b]:.2f}"
                                          for b, name in known.items()), flush=True)


if __name__ == "__main__":
    main()
