"""
Picks a layer's start save: among a run's saves in the given map cells, the healthy one with the best balance of
missiles and energy. Usage: python pick_layer.py <run> <out.state> y,x [y,x ...]
"""
import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of, at
from richest import healthy


def total_energy(v):
    return (int(v[0x107]) >> 4) * 100 + energy_of(v)


def balance(v):
    return min(int(v[at(0x6879)]), 30) / 30 + min(total_energy(v), 300) / 300


if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    cells = {tuple(map(int, c.split(","))) for c in sys.argv[3:]}
    states = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))["states"]
    here = [s for s in states if (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) in cells and total_energy(emulator.view(s)) <= 600]
    here.sort(key=lambda s: -balance(emulator.view(s)))
    s = next(s for s in here if healthy(emulator, s)); v = emulator.view(s)
    Path(sys.argv[2]).write_bytes(s)
    print(f"{sys.argv[1]} -> {sys.argv[2]}: cell {int(v[MAP_Y])},{int(v[MAP_X])} missiles {int(v[at(0x6879)])} energy {total_energy(v)}")
