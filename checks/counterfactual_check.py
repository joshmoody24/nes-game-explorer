"""
Proxy vs the real thing. Proxy: a resource is a byte my buttons can change that holds still when I do
nothing. Real thing: a byte matters if giving it a different value (one it really takes elsewhere)
changes how much my actions can affect the world. The real test edits the byte in save states and
measures empowerment with paired random buttons. Metroid addresses are used only to grade.

Usage: python -m checks.counterfactual_check <run name> [states for the proxy] [states for edits]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from controls import FRAMES_PER_STEP
from emulator import Emulator
from empowerment import controllable_bits
from metroid import CURRENT_MISSILES, ENERGY_TANKS, ITEMS, MAX_MISSILES, at

KNOWN = {at(CURRENT_MISSILES): "current missiles", 0x10E: "weapon", at(MAX_MISSILES): "max missiles",
         at(ENERGY_TANKS): "energy tanks", at(ITEMS): "items", 0x106: "health ones", 0x107: "health tens"}


def name(byte):
    label = f"${byte:04X}" if byte < 0x800 else f"${byte - 0x800 + 0x6000:04X}"
    return f"{label} ({KNOWN[byte]})" if byte in KNOWN else label


def edited(emulator, state, byte, value):
    offset = (emulator._system_slice.start + byte if byte < 0x800
              else emulator._cartridge_slice.start + byte - 0x800)
    copy = bytearray(state)
    copy[offset] = value
    return bytes(copy)


def main():
    run_dir = Path("runs") / sys.argv[1]
    proxy_count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    edit_count = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    states = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))["states"]
    rng = np.random.default_rng(0)

    sample = [states[i] for i in rng.choice(len(states), proxy_count, replace=False)]
    memories = np.stack([emulator.memory(s) for s in sample])
    changeable = np.zeros((proxy_count, memories.shape[1]), bool)
    drifts = np.zeros_like(changeable)
    for i, state in enumerate(sample):
        changeable[i] = controllable_bits(emulator, state, np.random.default_rng(i), 8, 30).reshape(-1, 8).any(1)
        emulator.load(state)
        emulator.run(0, 30 * FRAMES_PER_STEP)
        drifts[i] = memories[i] != emulator.memory()
    count = changeable.sum(0)
    sometimes = (count >= 3) & (count < 0.3 * proxy_count)
    stillness = (changeable & ~drifts).sum(0) / np.maximum(count, 1)
    proxy = set(np.flatnonzero(sometimes & (stillness >= 0.95)).tolist())
    rejected = sorted(set(np.flatnonzero(sometimes).tolist()) - proxy)
    tested = sorted(proxy | set(KNOWN) | set(rng.choice(rejected, min(100, len(rejected)), replace=False).tolist()))

    edit_states = sample[:edit_count]
    effects = {}
    for byte in tested:
        values = np.unique(memories[:, byte])
        changes = []
        for j, state in enumerate(edit_states):
            current = emulator.memory(state)[byte]
            others = values[values != current]
            if len(others) == 0:
                continue
            seed = 1_000 + j
            before = controllable_bits(emulator, state, np.random.default_rng(seed), 4, 20).sum()
            after = controllable_bits(emulator, edited(emulator, state, byte, int(rng.choice(others))),
                                      np.random.default_rng(seed), 4, 20).sum()
            changes.append(abs(int(after) - int(before)) / max(int(before), 1))
        effects[byte] = float(np.median(changes)) if changes else 0.0

    def report(label, group):
        scores = np.array([effects[b] for b in group])
        print(f"{label}: {len(group)} bytes, median effect {np.median(scores):.1%}, "
              f"share with effect over 5%: {np.mean(scores > 0.05):.0%}")

    report("bytes the proxy flags as resources", sorted(proxy))
    report("sometimes-changeable bytes the proxy rejects (sample)", [b for b in tested if b not in proxy and b not in KNOWN])
    print("known bytes:")
    for byte in KNOWN:
        print(f"  {name(byte)}: proxy {'yes' if byte in proxy else 'no'}, editing changes empowerment by {effects[byte]:.1%}")
    ranked = sorted(tested, key=lambda b: -effects[b])[:25]
    print("biggest effects:", ", ".join(f"{name(b)} {effects[b]:.0%}{'*' if b in proxy else ''}" for b in ranked))
    print("(* = flagged by the proxy)")


if __name__ == "__main__":
    main()
