import os, pickle, zlib
from collections import Counter
from pathlib import Path
import numpy as np
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from kdoors import parse, grid, kinds
from bdoors import room_offset
from metroid import MAP_X, MAP_Y, energy_of, at
if __name__ == "__main__":
    for cell in ((29, 17), (29, 18), (30, 17), (28, 17), (29, 16)):
        n = int(grid[cell])
        if n == 0xFF: print(cell, "no room"); continue
        room = parse(room_offset(0x1618F, 0x3FF0, n, 0x10000))
        print(cell, f"room {n:#x}:", [("right" if d >> 4 == 0xA else "left", kinds.get(d & 0xF, d & 0xF)) for d in room[1]] if room else "unparsed")
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng(0)
    saved = pickle.loads(zlib.decompress(Path("runs/lava-layer2c/checkpoint.pkl.z").read_bytes()))
    for cell in ((29, 18), (30, 17)):
        here = [s for s in saved["states"] if emulator.view(s)[0x74] == 0x14 and (int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) == cell]
        missiles = Counter(int(emulator.view(s)[at(0x6879)]) for s in here)
        print(cell, "missiles held:", sorted(missiles.items())[-6:], "| max total energy", max((int(emulator.view(s)[0x107]) >> 4) * 100 + energy_of(emulator.view(s)) for s in here))
        rich = sorted(here, key=lambda s: -int(emulator.view(s)[at(0x6879)]))[:20]
        hit = 0
        for s in rich:
            for _ in range(25):
                emulator.load(s); a = rng.integers(len(ACTIONS)); keep = rng.uniform(0.9, 0.99)
                for _ in range(200):
                    if rng.random() > keep: a = rng.integers(len(ACTIONS))
                    emulator.run(ACTIONS[a], FRAMES_PER_STEP)
                    m = emulator.memory()
                    if (int(m[MAP_Y]), int(m[MAP_X])) == (29, 17): hit += 1; Path("cache/ridley-room.state").write_bytes(emulator.save()); break
        print(cell, "random tries from the 20 most-armed saves reaching (29,17):", hit, "/ 500")
