import os, pickle, zlib
from pathlib import Path
import numpy as np
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y, energy_of
def absx(v):
    return int(v[MAP_X]) * 256 + int(v[0xFD]) + (int(v[ROOM_X]) - int(v[0xFD])) % 256
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path("runs/lava-layer2c/checkpoint.pkl.z").read_bytes()))
    row = [s for s in saved["states"] if emulator.view(s)[0x74] == 0x14 and int(emulator.view(s)[MAP_Y]) == 29 and 17 <= int(emulator.view(s)[MAP_X]) <= 19]
    row.sort(key=lambda s: absx(emulator.view(s)))
    for s in row[:6]:
        v = emulator.view(s); print("absolute x", absx(v), "| cell", int(v[MAP_Y]), int(v[MAP_X]), "| y", int(v[ROOM_Y]), "| $FD", int(v[0xFD]), "| energy", (int(v[0x107]) >> 4) * 100 + energy_of(v))
    rng = np.random.default_rng(0); hit = 0; n = 0
    for s in row[:20]:
        for _ in range(30):
            emulator.load(s); n += 1; a = rng.integers(len(ACTIONS)); keep = rng.uniform(0.9, 0.99)
            for _ in range(150):
                if rng.random() > keep: a = rng.integers(len(ACTIONS))
                emulator.run(ACTIONS[a], FRAMES_PER_STEP)
                m = emulator.memory()
                if (int(m[MAP_Y]), int(m[MAP_X])) == (29, 17) and absx(m) < 4608: hit += 1; Path("cache/ridley-room.state").write_bytes(emulator.save()); break
    print(f"random tries from the 20 leftmost saves into Ridley's room: {hit}/{n}")
