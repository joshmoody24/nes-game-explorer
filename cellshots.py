"""Screenshots (half size) of one save per requested map cell, from the given runs, tiled 5 wide."""
import os, pickle, subprocess, sys, zlib
from pathlib import Path
import numpy as np
from controls import FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y

if __name__ == "__main__":
    out, runs, cells = sys.argv[1], sys.argv[2].split("+"), [tuple(int(v) for v in c.split(",")) for c in sys.argv[3:]]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    found = {}
    for run in runs:
        for s in pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]:
            v = emulator.view(s); found.setdefault((int(v[MAP_Y]), int(v[MAP_X])), s)
    frames = []
    for c in cells:
        if c in found:
            emulator.load(found[c]); frames.append(emulator.frame(0)[::2, ::2]); print(c, end=" ")
    while len(frames) % 5: frames.append(np.zeros_like(frames[0]))
    image = np.concatenate([np.concatenate(frames[k:k + 5], 1) for k in range(0, len(frames), 5)], 0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{image.shape[1]}x{image.shape[0]}", "-i", "-", "-frames:v", "1", out], input=image.tobytes(), check=True)
