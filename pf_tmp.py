import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of, at
from richest import healthy
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    total = lambda v: (int(v[0x107]) >> 4) * 100 + energy_of(v)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    score = lambda v: min(int(v[at(0x6879)]), 60) * 4 + min(total(v), 400)
    lair = sorted((s for s in saved["states"] if emulator.view(s)[0x74] == 0x14 and total(emulator.view(s)) <= 500), key=lambda s: -score(emulator.view(s)))
    s = next(s for s in lair if healthy(emulator, s)); v = emulator.view(s)
    Path(sys.argv[2]).write_bytes(s); print(sys.argv[1], "-> best healthy:", int(v[at(0x6879)]), "missiles", total(v), "energy, cell", int(v[MAP_Y]), int(v[MAP_X]))
