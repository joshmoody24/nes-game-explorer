"""
Two generic ideas, graded against Metroid's known addresses (used only for grading):
  resources: bytes my buttons can change, that hold still when I do nothing
  vitals:    a state is dangerous if random play from it often ends with my buttons changing almost
             nothing (dead / game over), measured without knowing which byte is health

Usage: python -m checks.resource_vital_check <run name> [states]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from empowerment import controllable_bits
from metroid import CURRENT_MISSILES, ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAX_MISSILES, at

STEPS = 30
REPLAYS = 8
KNOWN = {"current missiles": at(CURRENT_MISSILES), "weapon": 0x10E, "max missiles": at(MAX_MISSILES),
         "energy tanks": at(ENERGY_TANKS), "items": at(ITEMS), "health ones": 0x106, "health tens": 0x107}


def name(byte):
    return f"${byte:04X}" if byte < 0x800 else f"${byte - 0x800 + 0x6000:04X}"


def random_play(emulator, state, rng, steps):
    emulator.load(state)
    action = rng.integers(len(ACTIONS))
    for _ in range(steps):
        if rng.random() > 0.9:  # mostly repeat the previous button, like the explorer
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    return emulator.save()


def energy(memory):
    return 10 * (memory[0x107] & 0x0F) + (memory[0x106] >> 4)


def main():
    run_dir = Path("runs") / sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    archive = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    rng = np.random.default_rng(0)
    states = [archive["states"][i] for i in rng.choice(len(archive["states"]), count, replace=False)]

    changeable = np.zeros((count, 10240), bool)
    drifts = np.zeros((count, 10240), bool)
    for i, state in enumerate(states):
        changeable[i] = controllable_bits(emulator, state, np.random.default_rng(i), REPLAYS, STEPS).reshape(-1, 8).any(1)
        emulator.load(state)
        start = emulator.memory()
        emulator.run(0, STEPS * FRAMES_PER_STEP)
        drifts[i] = start != emulator.memory()

    print("== resources: changeable by my buttons, still when idle ==")
    changeable_count = changeable.sum(0)
    still_when_changeable = (changeable & ~drifts).sum(0)
    sometimes = (changeable_count >= 3) & (changeable_count < 0.3 * count)
    stillness = np.where(changeable_count > 0, still_when_changeable / np.maximum(changeable_count, 1), 0)
    candidates = np.flatnonzero(sometimes & (stillness >= 0.95))
    candidates = candidates[np.argsort(-changeable_count[candidates])]
    print(f"{sometimes.sum()} bytes sometimes changeable; {len(candidates)} of them hold still when idle (>=95%):")
    print("  " + ", ".join(f"{name(b)} ({changeable_count[b]})" for b in candidates[:40]))
    for label, byte in KNOWN.items():
        print(f"  {label:>16} {name(byte)}: changeable in {changeable_count[byte]} states, "
              f"still-when-idle {stillness[byte]:.0%}, in candidates: {byte in set(candidates)}")

    print("== vitals: how often random play ends with my buttons changing almost nothing ==")
    lost, game_over, energies = [], [], []
    for i, state in enumerate(states):
        replay_rng = np.random.default_rng(10_000 + i)
        endings = [random_play(emulator, state, replay_rng, 60) for _ in range(4)]
        responsiveness = [controllable_bits(emulator, end, replay_rng, 3, 8).sum() for end in endings]
        lost.append(np.mean(np.array(responsiveness) < 20))
        game_over.append(np.mean([emulator.view(end)[GAME_MODE] == GAME_OVER for end in endings]))
        energies.append(energy(emulator.view(state)))
    lost, game_over, energies = map(np.array, (lost, game_over, energies))
    print(f"agreement of 'lost control' with real game over: {np.mean((lost > 0) == (game_over > 0)):.0%} of states")
    for low, high in [(0, 10), (10, 20), (20, 30), (30, 100)]:
        mask = (energies >= low) & (energies < high)
        if mask.any():
            print(f"  energy {low:>2}-{high - 1:<2}: {mask.sum():>3} states, danger (lost control) {lost[mask].mean():.2f}, "
                  f"real game over {game_over[mask].mean():.2f}")


if __name__ == "__main__":
    main()
