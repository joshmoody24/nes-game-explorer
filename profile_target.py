"""Runs the emulator in-game for a while, for a profiler to watch."""
import os

import numpy as np

from emulator import Emulator, button_mask

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
for buttons in np.random.default_rng(0).integers(0, 1 << 9, size=50_000).tolist():
    emulator.run(buttons, 4)
