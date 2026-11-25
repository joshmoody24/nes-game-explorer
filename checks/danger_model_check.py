"""
Can a model learn from random play alone that health matters? Labels states by whether short random
play loses control, fits vitals.DangerModel on raw memory, then grades what it learned using Metroid's
health bytes (used only here): does adding health lower predicted danger, more than editing other bytes?

Usage: python -m checks.danger_model_check <run name> [<run name> ...]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import jax
import numpy as np

from emulator import Emulator
from vitals import fit, lost_control

STATES = 4000
REPLAYS_PER_STATE = 2


def energy(memory):
    return 10 * (memory[0x107] & 0x0F) + (memory[0x106] >> 4)


def with_energy(memory, value):
    edited = memory.copy()
    edited[0x107] = (edited[0x107] & 0xF0) | (value // 10)
    edited[0x106] = (edited[0x106] & 0x0F) | ((value % 10) << 4)
    return edited


def auc(scores, labels):
    """Probability that a random positive outranks a random negative."""
    positives, negatives = scores[labels], scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positives, negatives]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(positives)].sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    states = [s for run in sys.argv[1:]
              for s in pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))["states"]]
    rng = np.random.default_rng(0)
    states = [states[i] for i in rng.choice(len(states), min(STATES, len(states)), replace=False)]
    memories = np.stack([emulator.memory(s) for s in states])
    labels = np.array([np.mean([lost_control(emulator, s, np.random.default_rng([i, r])) for r in range(REPLAYS_PER_STATE)])
                       for i, s in enumerate(states)])
    print(f"{len(states)} states; random play lost control in {labels.mean():.0%} of replays")

    held_out = rng.random(len(states)) < 0.2
    model, losses = fit(memories[~held_out], labels[~held_out], jax.random.key(0))
    predicted = model.danger(memories[held_out])
    print(f"training loss {losses[0]:.3f} -> {losses[-1]:.3f}; held-out AUC (any replay lost control): "
          f"{auc(predicted, labels[held_out] > 0):.2f} (0.5 = guessing)")

    low = memories[(np.array([energy(m) for m in memories]) <= 5)]
    healed = np.stack([with_energy(m, 29) for m in low])
    change = model.danger(healed) - model.danger(low)
    print(f"{len(low)} states with energy <= 5: setting energy to 29 changes predicted danger by "
          f"{np.median(change):+.2f} (median); lowered it in {np.mean(change < 0):.0%}")
    control_changes = []
    for byte in rng.choice(model.byte_indices, 200, replace=False):
        if byte in (0x106, 0x107):
            continue
        edited = low.copy()
        edited[:, byte] = rng.choice(memories[:, byte], len(low))
        control_changes.append(np.median(model.danger(edited) - model.danger(low)))
    control_changes = np.abs(control_changes)
    print(f"control: editing 200 random other bytes changes it by {np.median(control_changes):.2f} (median absolute); "
          f"{np.mean(control_changes >= abs(np.median(change))):.0%} of them by as much as healing")


if __name__ == "__main__":
    main()
