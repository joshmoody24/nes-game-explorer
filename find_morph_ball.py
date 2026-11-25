"""Random search from the room left of the start until the documented items byte ($6878) changes."""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emulator import Emulator, button_mask

ITEMS = 0x6878 - 0x6000 + 0x800
emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
for _ in range(40):
    emulator.run(button_mask("LEFT"), 4)
start = emulator.save()
rng = np.random.default_rng(0)
choices = [button_mask("LEFT"), button_mask("LEFT", "A"), button_mask("A"), button_mask("RIGHT"),
           button_mask("RIGHT", "A"), 0]
started = time.perf_counter()
for attempt in range(3000):
    emulator.load(start)
    for step in range(150):
        emulator.run(choices[rng.choice(len(choices), p=[0.35, 0.3, 0.1, 0.1, 0.1, 0.05])], 4)
        memory = emulator.memory()
        if memory[ITEMS]:
            print(f"items byte became {hex(memory[ITEMS])} on attempt {attempt}, step {step} "
                  f"({time.perf_counter() - started:.0f}s); map y,x = {memory[0x4F]},{memory[0x50]}")
            plt.imsave("/tmp/morph.png", emulator.screenshot())
            raise SystemExit
print(f"no item change after 3000 attempts ({time.perf_counter() - started:.0f}s)")
