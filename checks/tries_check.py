"""
How often does one explorer try get through a wall? From a save, runs many single tries (STEPS steps of random
held buttons, as explorers.explore presses them) under several hold styles, and counts tries that reach a target
room, tries that die and tries that take any damage. Seconds of CPU instead of whole exploration runs.

Usage: python -m checks.tries_check <state file> <area,map y,map x of the target room> [tries] [last button]
"""

import os
import sys
from multiprocessing import get_context

import numpy as np

import graph
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import GAME_MODE, GAME_OVER, energy_of

STEPS = int(os.environ.get("STEPS", 100))
STYLES = {"normal holds (keep 0.9)": (0.9, 0.0, 0.0), "long holds (keep 0.90-0.99)": (0.99, 0.09, 0.0),
          "long holds + momentum 0.5": (0.99, 0.09, 0.5), "very long holds (keep 0.99)": (0.99, 0.0, 0.0)}


def _tries(task):
    state, target, keep, spread, momentum, last, seeds = task
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    reached = died = damaged = clean = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        p = keep - rng.random() * spread
        action = last if last is not None and rng.random() < momentum else rng.integers(len(ACTIONS))
        emulator.load(state)
        energy, hurt = energy_of(emulator.memory()), False
        for _ in range(STEPS):
            if rng.random() > p:
                action = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            memory = emulator.memory()
            hurt = hurt or energy_of(memory) < energy
            energy = energy_of(memory)
            if memory[GAME_MODE] == GAME_OVER:
                died += 1
                break
            if graph.location(memory)[:3] == target:
                reached += 1
                clean += not hurt
                break
        damaged += hurt
    return reached, died, damaged, clean


def main():
    state = open(sys.argv[1], "rb").read()
    target = tuple(int(v) for v in sys.argv[2].split(","))
    tries = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    last = int(sys.argv[4]) if len(sys.argv) > 4 else None
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    print(f"{sys.argv[1]}: at {graph.location(emulator.view(state))}, target room {target}, {tries} tries of {STEPS} steps"
          + (f", last button {last}" if last is not None else ""), flush=True)
    chunks = np.array_split(np.arange(tries), 16)
    with get_context("spawn").Pool(16) as pool:
        for name, (keep, spread, momentum) in STYLES.items():
            if momentum and last is None:
                continue
            results = pool.map(_tries, [(state, target, keep, spread, momentum, last, list(c)) for c in chunks])
            reached, died, damaged, clean = map(sum, zip(*results))
            print(f"  {name:>28}: reached {reached / tries:.1%} ({clean / tries:.1%} without damage), died "
                  f"{died / tries:.1%}, took damage {damaged / tries:.1%}", flush=True)


if __name__ == "__main__":
    main()
