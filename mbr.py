"""Ratchet pick for Mother Brain: healthy save with most hits on her, then best missiles+energy. python mbr.py <run> <out>"""
import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at
from pick_layer import total_energy
from richest import healthy
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))["states"]
    vs = [emulator.view(s) for s in states]
    print(sys.argv[1], "| MB status", dict(Counter(int(v[0x98]) for v in vs)), "| max hits", max(int(v[0x99]) for v in vs), "| ending", sum(1 for v in vs if v[at(0x6883)]))
    c = sorted((s for s in states if total_energy(emulator.view(s)) <= 600 and total_energy(emulator.view(s)) >= 20), key=lambda s: (-int(emulator.view(s)[0x99]), -min(int(emulator.view(s)[at(0x6879)]) / 60, total_energy(emulator.view(s)) / 200)))
    s = next(s for s in c if healthy(emulator, s)); v = emulator.view(s)
    Path(sys.argv[2]).write_bytes(s); print("next start: hits", int(v[0x99]), "missiles", int(v[at(0x6879)]), "energy", total_energy(v), "cell", int(v[MAP_Y]), int(v[MAP_X]))
