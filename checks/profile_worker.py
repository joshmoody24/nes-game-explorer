"""
How much of a worker's time is emulation vs our per-step bookkeeping? Times the explore loop against
bare emulation in one process, checks MemoryView matches memory(), and shows the top profile entries.
"""
import cProfile
import os
import pstats
import time

import numpy as np

from emulator import Emulator
from explorers import ACTIONS, FRAMES_PER_STEP, Settings, explore
from metroid import start_game

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
start = start_game(emulator)
rng = np.random.default_rng(0)

for _ in range(200):
    emulator.run(ACTIONS[rng.integers(len(ACTIONS))], FRAMES_PER_STEP)
    state = emulator.save()
    full, view = emulator.memory(state), emulator.view(state)
    assert all(full[i] == view[i] for i in range(len(full))), "MemoryView disagrees with memory()"
print("MemoryView matches memory() on 200 states")

steps = 20_000
started = time.perf_counter()
for _ in range(steps):
    emulator.run(ACTIONS[rng.integers(len(ACTIONS))], FRAMES_PER_STEP)
bare = steps * FRAMES_PER_STEP / (time.perf_counter() - started)

segments = 200
settings = Settings()
profile = cProfile.Profile()
started = time.perf_counter()
profile.enable()
frames = sum(explore(emulator, rng, settings, set(), 0, start).frames for _ in range(segments))
profile.disable()
explored = frames / (time.perf_counter() - started)
print(f"bare emulation: {bare:,.0f} frames/s; explore loop: {explored:,.0f} frames/s "
      f"({1 - explored / bare:.0%} overhead)")
pstats.Stats(profile).sort_stats("tottime").print_stats(8)
