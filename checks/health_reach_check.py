"""
Does more health causally extend how far random play reaches? Takes recently discovered (frontier)
states with low energy, makes an exact copy with energy raised to 29 by editing the save state, and runs
the same random segments (same seeds) from both. Counts places reached that the archive doesn't have yet.
Metroid's health addresses are used only to build the copies.

Usage: python -m checks.health_reach_check <run name> [states] [segments per state]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import Settings, hand_made_place
from metroid import GAME_MODE, GAME_OVER

STEPS = 100


def energy(memory):
    return 10 * (memory[0x107] & 0x0F) + (memory[0x106] >> 4)


def healed(emulator, state, value=29):
    copy = bytearray(state)
    base = emulator._system_slice.start
    copy[base + 0x107] = (copy[base + 0x107] & 0xF0) | (value // 10)
    copy[base + 0x106] = (copy[base + 0x106] & 0x0F) | ((value % 10) << 4)
    return bytes(copy)


def reach(emulator, state, seeds, known, settings):
    """Places reached by random segments from state that aren't in known, and how many segments died."""
    new, deaths = set(), 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        emulator.load(state)
        action = rng.integers(len(ACTIONS))
        for _ in range(STEPS):
            if rng.random() > 0.9:
                action = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            memory = emulator.view(emulator.save())
            if memory[GAME_MODE] == GAME_OVER:
                deaths += 1
                break
            place = hand_made_place(memory, settings)
            if place not in known:
                new.add(place)
    return new, deaths


def main():
    run_dir = Path("runs") / sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    segments = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    settings = Settings()
    archive = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    known = set(map(tuple, archive["places"]))
    frontier = [s for s in archive["states"][-400:] if energy(emulator.view(s)) <= 9][:count]

    sample_state = healed(emulator, frontier[0])
    emulator.load(sample_state)
    emulator.run(0, 1)
    assert energy(emulator.memory()) >= 28, f"edit didn't take: energy {energy(emulator.memory())}"

    low_new, high_new, low_deaths, high_deaths = [], [], 0, 0
    for i, state in enumerate(frontier):
        seeds = [[i, s] for s in range(segments)]
        new, deaths = reach(emulator, state, seeds, known, settings)
        low_new.append(len(new))
        low_deaths += deaths
        new, deaths = reach(emulator, healed(emulator, state), seeds, known, settings)
        high_new.append(len(new))
        high_deaths += deaths
    low_new, high_new = np.array(low_new), np.array(high_new)
    total = len(frontier) * segments
    print(f"{len(frontier)} frontier states with energy <= 9, {segments} random segments of {STEPS} steps each")
    print(f"new places reached per state: low energy mean {low_new.mean():.2f}, healed to 29 mean {high_new.mean():.2f}")
    print(f"healed reached more in {np.mean(high_new > low_new):.0%} of states, fewer in {np.mean(high_new < low_new):.0%}, "
          f"same in {np.mean(high_new == low_new):.0%}")
    print(f"segments ending in game over: low {low_deaths / total:.0%}, healed {high_deaths / total:.0%}")


if __name__ == "__main__":
    main()
