"""
Finds "damage dealt" bytes (an enemy's or obstacle's health) without knowing the game: paired tries from one save
with identical random buttons, once as pressed and once with the fire button masked out. A damage byte drops in
the firing tries, never rises within a try, and doesn't move in the masked tries. The player's own ammo also drops
when firing, wherever the shot goes; spendables.py already finds those, so they can be set aside.

Usage: python -m checks.damage_bytes <state file> [pairs] [steps] [refill edits]
"""

import os
import sys
from multiprocessing import get_context

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM, refilled
from metroid import GAME_MODE, GAME_OVER

FIRE = button_mask("B")
UNFIRED = [ACTIONS.index(a & ~FIRE) for a in ACTIONS]  # each action with the fire button let go


def _pair(task):
    state, seeds, steps = task
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    size = len(emulator.memory())
    drops = {True: np.zeros(size), False: np.zeros(size)}
    rises = np.zeros(size)
    for seed in seeds:
        for fire in (True, False):
            rng = np.random.default_rng(seed)
            keep = 0.99 - rng.random() * 0.09
            action = rng.integers(len(ACTIONS))
            emulator.load(state)
            first = previous = emulator.memory().astype(int)
            for _ in range(steps):
                if rng.random() > keep:
                    action = rng.integers(len(ACTIONS))
                emulator.run(ACTIONS[action if fire else UNFIRED[action]], FRAMES_PER_STEP)
                memory = emulator.memory().astype(int)
                if memory[GAME_MODE] == GAME_OVER:
                    break
                if fire:
                    rises += memory > previous
                previous = memory
            drops[fire] += previous < first
    return drops[True], drops[False], rises


def main():
    state = open(sys.argv[1], "rb").read()
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    if len(sys.argv) > 4:
        state = refilled(emulator, state, sys.argv[4])
    with get_context("spawn").Pool(20) as pool:
        results = pool.map(_pair, [(state, range(k, pairs, 20), steps) for k in range(20)])
    fired, masked, rises = (sum(r[i] for r in results) / pairs for i in range(3))
    score = (fired - masked) * (rises < 0.05)  # drops only with firing, and (almost) never rises within a try
    for i in np.argsort(-score)[:8]:
        print(f"  {i:#06x}: drops in {fired[i]:.0%} of firing tries vs {masked[i]:.0%} masked; rises per try {rises[i]:.2f}")


if __name__ == "__main__":
    main()
