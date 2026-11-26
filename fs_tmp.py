import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of
from richest import healthy
CELLS = {(21, 26), (21, 27), (21, 28), (21, 29), (21, 30)}
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    total = lambda v: (int(v[0x107]) >> 4) * 100 + energy_of(v)
    near = sorted((s for s in saved["states"] if emulator.view(s)[0x74] == 0x11 and (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) in CELLS), key=lambda s: -total(emulator.view(s)))
    for s in near:
        if healthy(emulator, s):
            v = emulator.view(s); print(sys.argv[1], "richest healthy near the lava corridor: total", total(v), "missiles", int(v[0x6879 - 0x6000 + 0x800]), "cell", int(v[MAP_Y]), int(v[MAP_X]))
            if len(sys.argv) > 2: Path(sys.argv[2]).write_bytes(s)
            break
