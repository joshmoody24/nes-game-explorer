"""Runs the emulator in-game for a while, for a profiler to watch."""
import os

import numpy as np

from emulator import Emulator, button_mask
from metroid import start_game

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
start_game(emulator)
for buttons in np.random.default_rng(0).integers(0, 1 << 9, size=50_000).tolist():
    emulator.run(buttons, 4)
