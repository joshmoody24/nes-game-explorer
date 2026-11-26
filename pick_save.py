import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, energy_of
if __name__ == "__main__":
    run, y, x, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
    here = [s for s in states if (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) == (y, x)]
    Path(out).write_bytes(max(here, key=lambda s: energy_of(emulator.view(s))))
    print(len(here), "saves at", (y, x), "->", out)
