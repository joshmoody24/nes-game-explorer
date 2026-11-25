"""Walks left from the start toward the Morph Ball and prints the documented addresses as they change."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator import Emulator, button_mask

CARTRIDGE = 0x800 - 0x6000  # memory() puts cartridge RAM ($6000-) right after the 2 KB system RAM


def read(memory):
    return {
        "mode $1E": memory[0x1E],
        "area $74": hex(memory[0x74]),
        "map y,x $4F/$50": (memory[0x4F], memory[0x50]),
        "items $6878": hex(memory[0x6878 + CARTRIDGE]),
        "tanks $6877": memory[0x6877 + CARTRIDGE],
        "max missiles $687A": memory[0x687A + CARTRIDGE],
    }


emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
previous = None
for step in range(400):
    buttons = button_mask("LEFT", "A") if step % 20 < 8 else button_mask("LEFT")
    emulator.run(buttons, 4)
    values = read(emulator.memory())
    if values != previous:
        print(f"frame {step * 4:>5}: {values}")
        previous = values
plt.imsave(os.environ.get("SHOT", "/tmp/left.png"), emulator.screenshot())
