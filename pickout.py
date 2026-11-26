import os, pickle, sys, zlib
from pathlib import Path
from richest import healthy
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at, energy_of
if __name__ == "__main__":
    run, area, out = sys.argv[1], int(sys.argv[2], 16), sys.argv[3]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
    rich = lambda s: (int(emulator.view(s)[at(0x6877)]), int(emulator.view(s)[263]) >> 4, energy_of(emulator.view(s)), int(emulator.view(s)[at(0x6879)]))
    pool = [s for s in states if emulator.view(s)[0x74] == area and emulator.view(s)[at(0x687B)] & 3]
    best = next(s for s in sorted(pool, key=rich, reverse=True) if healthy(emulator, s))
    v = emulator.view(best); Path(out).write_bytes(best)
    print(len(pool), "saves in that area after Kraid; picked (tanks, filled, energy, missiles)", rich(best), "at map", int(v[MAP_Y]), int(v[MAP_X]))
