import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    vs = [emulator.view(s) for s in saved["states"]]
    print(sys.argv[1], "Brinstar cells rows<=13:", sorted({(int(v[MAP_Y]), int(v[MAP_X])) for v in vs if (int(v[0x74]) or 0x10) == 0x10 and int(v[MAP_Y]) <= 13}), "| Norfair saves", sum(1 for v in vs if v[0x74] == 0x11))
