"""Lists the tagged blocks inside a QuickNES save state, and which change during play."""
import os
import re

import numpy as np

from emulator import Emulator, button_mask

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
before = np.frombuffer(emulator.save(), np.uint8).copy()
for buttons in np.random.default_rng(0).integers(0, 1 << 9, size=5000).tolist():
    emulator.run(buttons, 4)
after = np.frombuffer(emulator.save(), np.uint8)
raw = bytes(before)
print("header:", raw[:16])
for match in re.finditer(rb"[A-Z][A-Z0-9 ]{3}", raw):
    offset = match.start()
    size = int.from_bytes(raw[offset + 4:offset + 8], "little")
    if 0 < size < len(raw) and offset + 8 + size <= len(raw):
        changed = np.count_nonzero(before[offset + 8:offset + 8 + size] != after[offset + 8:offset + 8 + size])
        print(f"{match.group().decode()!r} at {offset:>6}, size {size:>6}, bytes changed during play: {changed}")
