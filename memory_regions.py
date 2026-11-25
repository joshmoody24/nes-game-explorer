"""Which memory regions does the core expose, and does cartridge RAM change during play?"""
import ctypes
import os

import numpy as np

from emulator import Emulator, button_mask

REGIONS = {"save RAM (cartridge $6000-$7FFF)": 0, "RTC": 1, "system RAM": 2, "video RAM": 3}
emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
core = emulator._core
for name, region in REGIONS.items():
    print(f"{name:>34}: {core.retro_get_memory_size(region)} bytes, "
          f"pointer {'yes' if core.retro_get_memory_data(region) else 'no'}")

size = core.retro_get_memory_size(0)
if size:
    cartridge = np.ctypeslib.as_array(ctypes.cast(core.retro_get_memory_data(0), ctypes.POINTER(ctypes.c_uint8)), (size,))
    emulator.run(frames=3000)
    for _ in range(4):
        emulator.run(button_mask("START"), frames=5)
        emulator.run(frames=120)
    before = cartridge.copy()
    rng = np.random.default_rng(0)
    for buttons in rng.integers(0, 1 << 9, size=5000).tolist():
        emulator.run(buttons, 4)
    print(f"cartridge RAM bytes nonzero: {np.count_nonzero(cartridge)}, "
          f"changed during 20K frames of random play: {np.count_nonzero(before != cartridge)}")
