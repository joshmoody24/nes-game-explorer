import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    n = Counter((hex(emulator.view(s)[0x74]), int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) for s in saved["states"])
    route = [(20, 30), (19, 30), (18, 30), (17, 30), (16, 30), (15, 30), (14, 30), (14, 28), (14, 25), (14, 22), (13, 22), (12, 22)]
    print("route up and back (cell, saves):", [(c, n.get(("0x11",) + c, 0)) for c in route], "| Brinstar saves", sum(v for k, v in n.items() if k[0] in ("0x10", "0x0")))
