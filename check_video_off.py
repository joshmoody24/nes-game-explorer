"""
Does turning video off change how the game plays? Replays the same random inputs from the same
state with video on and off, and compares RAM every 60 frames.
"""
import os

import numpy as np

from emulator import Emulator, button_mask

emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
emulator.run(frames=3000)
for _ in range(4):
    emulator.run(button_mask("START"), frames=5)
    emulator.run(frames=120)
start = emulator.save()
inputs = np.random.default_rng(1).integers(0, 1 << 9, size=20_000).tolist()


def ram_trace(video):
    emulator.load(start)
    emulator._capturing_video = video
    trace = []
    for i, buttons in enumerate(inputs):
        emulator.run(buttons)
        if i % 60 == 0:
            trace.append(emulator.ram.copy())
    emulator._capturing_video = False
    return np.array(trace)


with_video, without_video = ram_trace(True), ram_trace(False)
differing = np.flatnonzero((with_video != without_video).any(1))
print(f"checkpoints compared: {len(with_video)}, checkpoints that differ: {len(differing)}")
if len(differing):
    first = differing[0]
    print(f"first difference at frame {first * 60}, bytes: "
          f"{np.flatnonzero(with_video[first] != without_video[first])[:20].tolist()}")
