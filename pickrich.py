import os, pickle, sys, zlib
from pathlib import Path
from richest import healthy
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at, energy_of
if __name__ == "__main__":
    run, out = sys.argv[1], sys.argv[2]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
    rich = lambda s: (int(emulator.view(s)[263]) >> 4, energy_of(emulator.view(s)), int(emulator.view(s)[at(0x6879)]))
    best = next(s for s in sorted(states, key=rich, reverse=True) if healthy(emulator, s))
    v = emulator.view(best); Path(out).write_bytes(best)
    print("richest healthy save:", rich(best), "at map", int(v[MAP_Y]), int(v[MAP_X]), "Kraid present", int(v[at(0x6987)]))
