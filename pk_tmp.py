import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at
from pick_layer import total_energy
from richest import healthy
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))["states"]
    cells = {tuple(map(int, c.split(","))) for c in sys.argv[3:]}
    joint = lambda v: min(int(v[at(0x6879)]) / 150, total_energy(v) / 300)
    c = sorted((s for s in states if (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) in cells and total_energy(emulator.view(s)) <= 600), key=lambda s: -joint(emulator.view(s)))
    s = next(s for s in c if healthy(emulator, s)); v = emulator.view(s)
    Path(sys.argv[2]).write_bytes(s); print("start: cell", int(v[MAP_Y]), int(v[MAP_X]), "missiles", int(v[at(0x6879)]), "energy", total_energy(v), "MB status", int(v[0x98]))
