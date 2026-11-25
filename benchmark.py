"""
How fast is emulation? Per-process callback overhead, and total frames/sec as processes are added.
Usage: benchmark.py <core path> [frames per action]
"""

import multiprocessing
import sys
import time

import numpy as np

from emulator import Emulator, button_mask

FRAMES = 20_000
FRAME_SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def in_game(core_path):
    emulator = Emulator(core_path, "roms/metroid.nes")
    emulator.run(frames=3000)
    for _ in range(4):
        emulator.run(button_mask("START"), frames=5)
        emulator.run(frames=120)
    return emulator


def frames_per_second(core_path):
    emulator = in_game(core_path)
    inputs = np.random.default_rng(0).integers(0, 1 << 9, size=FRAMES).tolist()
    started = time.perf_counter()
    for buttons in inputs[::FRAME_SKIP]:
        emulator.run(buttons, FRAME_SKIP)
    return FRAMES / (time.perf_counter() - started)


def main():
    core_path = sys.argv[1]
    for processes in [1, 6, 12, 18, 24]:
        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            rates = pool.map(frames_per_second, [core_path] * processes)
        print(f"{processes:>2} processes: {sum(rates):>9,.0f} frames/sec total, "
              f"{np.mean(rates):>7,.0f} per process", flush=True)


if __name__ == "__main__":
    main()
