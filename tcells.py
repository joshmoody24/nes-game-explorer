import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run in sys.argv[1:]:
        saved = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))
        ms = [emulator.memory(s) for s in saved["states"]]
        tour = sorted({(int(m[MAP_Y]), int(m[MAP_X])) for m in ms if m[0x74] == 0x13})
        rooms = len({(int(m[0x74]), int(m[MAP_Y]), int(m[MAP_X])) for m in ms})
        print(f"{run}: {len(ms)} saves, {rooms} rooms overall, {len(tour)} Tourian cells {tour}; MB in room {sum(1 for m in ms if m[0x98] == 1)}, MB max status {max(int(m[0x98]) for m in ms)}")
