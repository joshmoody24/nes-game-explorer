import os, pickle, zlib
from collections import Counter
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y
WORLD = np.frombuffer(Path(ROM).read_bytes()[0x254E:0x254E + 1024], np.uint8).reshape(32, 32)
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path("runs/hunt-worlds/checkpoint.pkl.z").read_bytes()))["states"]
    cells = Counter((int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) for s in states)
    empty = {c: n for c, n in cells.items() if c[0] < 32 and c[1] < 32 and WORLD[c] == 0xFF}
    print(len(states), "saves;", len(cells), "cells;", "cells with NO room in the ROM:", sorted(empty.items()), "| $49:", Counter(int(emulator.view(s)[0x49]) for s in states))
