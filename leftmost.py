"""Writes, per given map cell, the save from the given runs nearest the room's left edge, with edits applied."""
import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM, refilled
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y
if __name__ == "__main__":
    runs, edits, cells = sys.argv[1].split("+"), sys.argv[2], [tuple(int(v) for v in c.split(",")) for c in sys.argv[3:]]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    pool = []
    for run in runs: pool += pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
    for c in cells:
        here = [s for s in pool if (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) == c]
        s = min(here, key=lambda s: int(emulator.view(s)[ROOM_X]))
        Path(f"cache/left-{c[0]}-{c[1]}.state").write_bytes(refilled(emulator, s, edits))
        print(c, "x", int(emulator.view(s)[ROOM_X]), "y", int(emulator.view(s)[ROOM_Y]))
