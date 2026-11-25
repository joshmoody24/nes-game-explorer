"""Saves screenshots after the start sequence and after walking right. Debugging aid."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator import Emulator, button_mask

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
shots = [("after START presses", emulator.screenshot())]
emulator.run(button_mask("RIGHT"), frames=90)
shots.append(("after 90 frames of RIGHT", emulator.screenshot()))
emulator.run(button_mask("LEFT"), frames=180)
shots.append(("after 180 frames of LEFT", emulator.screenshot()))
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, image) in zip(axes, shots):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
