import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of
if __name__ == "__main__":
    run, out = sys.argv[1], sys.argv[2]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
    views = [(s, emulator.view(s)) for s in states]
    tour = [(s, v) for s, v in views if v[0x74] == 0x13]
    cells = Counter((int(v[MAP_Y]), int(v[MAP_X])) for _, v in tour)
    mb = Counter((int(v[0x98]), int(v[0x99])) for _, v in tour)
    print(len(tour), "Tourian saves; cells", sorted(cells.items()), "| Mother Brain (status, hits):", sorted(mb.items())[:10])
    best = max(tour, key=lambda sv: (int(sv[1][0x99]), -int(sv[1][MAP_Y]) * 0, energy_of(sv[1])))
    Path(out).write_bytes(best[0])
