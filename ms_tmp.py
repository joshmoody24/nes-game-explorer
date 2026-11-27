import os, pickle, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path("runs/mb-stay6/checkpoint.pkl.z").read_bytes()))
    vs = [emulator.view(s) for s in saved["states"]]
    print(len(vs), "saves | cells", sorted(Counter((int(v[MAP_Y]), int(v[MAP_X])) for v in vs).items()), "| MB status", dict(Counter(int(v[0x98]) for v in vs)), "| max hits", max(int(v[0x99]) for v in vs))
