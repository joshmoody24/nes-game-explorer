"""Runs the Morph Ball random search while feeding Progress, and prints the summary."""
import os

import numpy as np

from emulator import Emulator, button_mask
from metroid import start_game
from progress import Progress

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
start_game(emulator)
start = emulator.save()
progress = Progress()
rng = np.random.default_rng(0)
choices = [button_mask("LEFT"), button_mask("LEFT", "A"), button_mask("A"), button_mask("RIGHT"),
           button_mask("RIGHT", "A"), 0]
for attempt in range(300):
    emulator.load(start)
    previous = emulator.memory()
    for step in range(190):
        emulator.run(choices[rng.choice(len(choices), p=[0.35, 0.3, 0.1, 0.1, 0.1, 0.05])], 4)
        if step % 10 == 0:
            memory = emulator.memory()
            progress.see(memory, emulator.nametable())
            progress.see_change(previous, memory)
            previous = memory
print(progress.summary())
one_way = np.flatnonzero(progress.ever_up ^ progress.ever_down)
print("one-way bits (address, bit):", [(hex(i // 8 if i // 8 < 0x800 else i // 8 - 0x800 + 0x6000), 7 - i % 8)
                                       for i in one_way][:40])
