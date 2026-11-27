import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    total = lambda v: (int(v[0x107]) >> 4) * 100 + energy_of(v)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    t = [emulator.view(s) for s in saved["states"] if emulator.view(s)[0x74] == 0x13]
    print(sys.argv[1], len(t), "Tourian saves; cells", sorted(Counter((int(v[MAP_Y]), int(v[MAP_X])) for v in t).items()), "| max missiles", max([int(v[at(0x6879)]) for v in t] or [None]), "| max energy", max([total(v) for v in t if total(v) <= 500] or [None]), "| Mother Brain hits", max([int(v[0x99]) for v in t] or [None]), "| ending", sum(1 for v in t if v[at(0x6883)]))
