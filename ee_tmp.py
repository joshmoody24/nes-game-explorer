import os, pickle, zlib
from collections import Counter
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress(Path("runs/escape-h/checkpoint.pkl.z").read_bytes()))["states"]
    vs = [emulator.view(s) for s in states]
    print(len(vs), "saves | cells", sorted(Counter((int(v[MAP_Y]), int(v[MAP_X])) for v in vs).items())[:20], "| ENDING saves:", sum(1 for v in vs if v[at(0x6883)]), "| credits", sum(1 for v in vs if v[0x7B]))
