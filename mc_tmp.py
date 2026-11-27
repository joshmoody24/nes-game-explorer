import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    total = lambda v: (int(v[0x107]) >> 4) * 100 + energy_of(v)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    best = {}
    for s in saved["states"]:
        v = emulator.view(s)
        if v[0x74] == 0x13:
            c = (int(v[MAP_Y]), int(v[MAP_X])); m = int(v[at(0x6879)]); e = total(v) if total(v) <= 500 else 0
            b = best.get(c, (0, 0)); best[c] = (max(b[0], m), max(b[1], e))
    print(sys.argv[1], "(cell: most missiles, most energy):", sorted(best.items()), "| Mother Brain hits", max(int(emulator.view(s)[0x99]) for s in saved["states"]))
