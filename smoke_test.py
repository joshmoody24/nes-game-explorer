"""
Checks a core can run Metroid headless: speed, save/load round trip, and that RAM reacts to input.
Usage: smoke_test.py <core path> <rom path>
"""

import sys
import time

import numpy as np

from emulator import Emulator, button_mask


def main():
    core_path, rom_path = sys.argv[1:3]
    emulator = Emulator(core_path, rom_path)
    print(f"save state size: {emulator.state_size} bytes")

    started = time.perf_counter()
    emulator.run(frames=3000)
    print(f"idle frames per second: {3000 / (time.perf_counter() - started):,.0f}")

    for _ in range(4):
        emulator.run(button_mask("START"), frames=5)
        emulator.run(frames=120)
    in_game = emulator.save()

    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 1 << 9, size=600)
    started = time.perf_counter()
    for buttons in inputs:
        emulator.run(int(buttons))
    print(f"random-input frames per second: {600 / (time.perf_counter() - started):,.0f}")
    after_random = emulator.ram.copy()

    emulator.load(in_game)
    for buttons in inputs:
        emulator.run(int(buttons))
    print(f"save/load replay reproduces RAM exactly: {np.array_equal(after_random, emulator.ram)}")

    started = time.perf_counter()
    for _ in range(1000):
        emulator.load(in_game)
    load_time = (time.perf_counter() - started) / 1000
    started = time.perf_counter()
    for _ in range(1000):
        emulator.save()
    print(f"load: {load_time * 1e6:.0f} us, save: {(time.perf_counter() - started) / 1000 * 1e6:.0f} us")

    emulator.load(in_game)
    before = emulator.ram.copy()
    emulator.run(button_mask("RIGHT"), frames=60)
    changed = np.flatnonzero(before != emulator.ram)
    print(f"bytes changed by holding RIGHT for 60 frames: {len(changed)}; "
          f"first few: {[(hex(i), int(before[i]), int(emulator.ram[i])) for i in changed[:12]]}")


if __name__ == "__main__":
    main()
