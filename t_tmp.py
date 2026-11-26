import os, pickle, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path("runs/tourian-approach/checkpoint.pkl.z").read_bytes()))
    vs = [emulator.view(s) for s in saved["states"] if (int(emulator.view(s)[0x74]) or 0x10) == 0x10]
    cells = Counter((int(v[MAP_Y]), int(v[MAP_X])) for v in vs)
    print("Brinstar cells", len(cells), "| nearest to Tourian elevator (2,3):", sorted((abs(c[0] - 2) + abs(c[1] - 3), c) for c in cells)[:3], "| max missiles", max(int(v[at(0x6879)]) for v in vs))
