"""
Does a promise model trained on raw memory learn that ammo matters? Uses Metroid's known addresses
only to score the model, never to train it.

Usage: python -m checks.promise_check <run name>
"""

import os
import sys
from pathlib import Path

import jax
import numpy as np

from emulator import Emulator
from metroid import CURRENT_MISSILES, MAX_MISSILES, at
from promise import deviance_explained, fit, subset, trip_record

WEAPON = 0x10E  # 0 beam, 1 missiles


def address_name(index):
    return f"${index:04X}" if index < 0x800 else f"${index - 0x800 + 0x6000:04X}"


def main():
    run_dir = Path("runs") / sys.argv[1]
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    record = trip_record(run_dir, emulator)
    rng = np.random.default_rng(0)
    held_out = rng.random(len(record.trips)) < 0.2
    train, test = subset(record, ~held_out), subset(record, held_out)
    print(f"{len(record.trips)} places explored from, {int(record.trips.sum())} trips, "
          f"{int(record.discoveries.sum())} discoveries; model reads {(record.memories != record.memories[0]).any(0).sum()} "
          f"varying bytes")

    model, losses = fit(train, jax.random.key(0))
    print(f"training loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"deviance explained: train {deviance_explained(model, train):.1%}, held out {deviance_explained(model, test):.1%}")

    readable = {int(i) for i in model.byte_indices}
    missile_byte = at(CURRENT_MISSILES)
    print(f"model can see current missiles: {missile_byte in readable}, weapon byte: {WEAPON in readable}")
    empty = (record.memories[:, at(MAX_MISSILES)] >= 5) & (record.memories[:, missile_byte] == 0)
    if empty.any() and missile_byte in readable:
        before = record.memories[empty]
        refilled = before.copy()
        refilled[:, missile_byte] = before[:, at(MAX_MISSILES)]
        missile_effect = np.median(model.rate(refilled) / model.rate(before))
        control_effects = []
        others = [i for i in model.byte_indices if i != missile_byte]
        for byte in rng.choice(others, 200, replace=False):
            edited = before.copy()
            edited[:, byte] = rng.choice(record.memories[:, byte], len(before))
            control_effects.append(np.median(model.rate(edited) / model.rate(before)))
        rank = np.mean(np.abs(np.log(control_effects)) >= abs(np.log(missile_effect)))
        print(f"{empty.sum()} states with missile capacity but 0 missiles. Refilling missiles changes predicted "
              f"promise x{missile_effect:.2f}; random-byte edits change it x{np.median(control_effects):.2f} "
              f"(median); {rank:.0%} of random edits have a bigger effect")

    test_rate = model.rate(test.memories)
    baseline_loss = np.mean(test.trips * test_rate - test.discoveries * np.log(test_rate))
    importance = []
    for byte in model.byte_indices:
        shuffled = test.memories.copy()
        shuffled[:, byte] = rng.permutation(shuffled[:, byte])
        rate = model.rate(shuffled)
        importance.append(np.mean(test.trips * rate - test.discoveries * np.log(rate)) - baseline_loss)
    order = np.argsort(importance)[::-1][:15]
    print("bytes whose shuffling hurts predictions most:",
          ", ".join(f"{address_name(model.byte_indices[i])} (+{importance[i]:.4f})" for i in order))
    print(f"(for reference: current missiles = {address_name(missile_byte)}, weapon = {address_name(WEAPON)})")


if __name__ == "__main__":
    main()
