import os, subprocess, sys
from pathlib import Path
import numpy as np
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    s = Path(sys.argv[1]).read_bytes()
    frames = []
    for press in (("LEFT", "B"), ("RIGHT", "B"), ("LEFT", "A", "B"), ("RIGHT", "A", "B")):
        emulator.load(s)
        for t in range(120):
            emulator.run(button_mask(*press), FRAMES_PER_STEP)
        m = emulator.memory()
        print("+".join(press), "120 steps -> map", int(m[MAP_Y]), int(m[MAP_X]), "pos", int(m[ROOM_X]), int(m[ROOM_Y]), "door state", int(m[0x56]))
        frames.append(emulator.frame(0))
    image = np.concatenate([np.concatenate(frames[:2], 1), np.concatenate(frames[2:], 1)], 0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{image.shape[1]}x{image.shape[0]}", "-i", "-", "-frames:v", "1", "/tmp/walk.png"], input=image.tobytes(), check=True)
