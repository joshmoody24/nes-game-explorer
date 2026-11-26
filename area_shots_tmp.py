import os, pickle, subprocess, sys, zlib
from pathlib import Path
import numpy as np
import graph
from emulator import Emulator
from explorers import ROM

if __name__ == "__main__":
    area, out = int(sys.argv[1]), sys.argv[2]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path("runs/reference2/checkpoint.pkl.z").read_bytes()))
    by_room = {}
    for s in saved["states"]:
        r = graph.location(emulator.view(s))[:3]
        if r[0] == area: by_room.setdefault(r, s)
    rooms = sorted(by_room)
    frames = []
    for r in rooms:
        emulator.load(by_room[r]); f = emulator.frame(0)
        frames.append(f[::2, ::2])
    cols = 10
    while len(frames) % cols: frames.append(np.zeros_like(frames[0]))
    image = np.concatenate([np.concatenate(frames[k:k + cols], 1) for k in range(0, len(frames), cols)], 0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{image.shape[1]}x{image.shape[0]}", "-i", "-", "-frames:v", "1", out], input=image.tobytes(), check=True)
    print(len(rooms), "rooms:", rooms)
