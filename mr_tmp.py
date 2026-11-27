import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run in ("mbrace/plain", "mbrace/nametable"):
        saved = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))
        vs = [emulator.view(s) for s in saved["states"]]
        print(run, len(vs), "saves | (11,3) saves", sum(1 for v in vs if v[0x74] == 0x13 and (int(v[MAP_Y]), int(v[MAP_X])) == (11, 3)), "| MB status", dict(Counter(int(v[0x98]) for v in vs)), "| max hits", max(int(v[0x99]) for v in vs), "| ending", sum(1 for v in vs if v[at(0x6883)]))
