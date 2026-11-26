import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_Y
def height(v):
    return int(v[MAP_Y]) * 240 + int(v[0xFC]) + (int(v[ROOM_Y]) - int(v[0xFC])) % 240
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    vs = [emulator.view(s) for s in saved["states"]]
    brin = [v for v in vs if (int(v[0x74]) or 0x10) == 0x10]
    col = [height(v) for v in brin if int(v[MAP_X]) == 11]
    beyond = sorted({(int(v[MAP_Y]), int(v[MAP_X])) for v in brin if int(v[MAP_Y]) <= 7 and int(v[MAP_X]) != 11})
    print(f"{sys.argv[1]} {len(vs)} saves | lair {sum(1 for v in vs if v[0x74] == 0x12) / len(vs):.0%} | Norfair {sum(1 for v in vs if v[0x74] == 0x11)} | highest col-11 {min(col)} | cells past the door {beyond}")
