import os, subprocess, sys
from pathlib import Path
import numpy as np
from emulator import Emulator, TURBO
from explorers import ROM
from metroid import at
if __name__ == "__main__":
    from controls import ACTIONS, FRAMES_PER_STEP
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng(0)
    base = bytearray(Path(sys.argv[1]).read_bytes())
    off = int(emulator.state_offsets([at(0x6879)])[0]); base[off] = 50; base = bytes(base)
    spent, mode_on = [], []
    for _ in range(400):
        emulator.load(base); a = rng.integers(len(ACTIONS)); keep = rng.uniform(0.9, 0.99); on = 0
        for _ in range(100):
            if rng.random() > keep: a = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[a], FRAMES_PER_STEP); on += int(emulator.memory()[0x010E] != 0)
        spent.append(50 - int(emulator.memory()[at(0x6879)])); mode_on.append(on / 100)
    s = np.array(spent)
    print(f"{len(ACTIONS)} actions: missiles spent per 100-step try: mean {s.mean():.1f}, median {np.median(s):.0f}, 90th pct {np.percentile(s, 90):.0f} | time in missile mode {np.mean(mode_on):.0%}")
