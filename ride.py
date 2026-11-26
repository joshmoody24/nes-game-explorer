import os, sys
from pathlib import Path
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM
from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    base = Path(sys.argv[1]).read_bytes()
    for plan in (["LEFT"] * 30 + ["DOWN"] * 200, ["RIGHT"] * 30 + ["DOWN"] * 200, ["DOWN"] * 230, ["LEFT"] * 15 + ["DOWN"] * 215):
        emulator.load(base)
        for b in plan: emulator.run(button_mask(b), FRAMES_PER_STEP)
        m = emulator.memory()
        print(plan[0], "then DOWN: area byte", hex(m[0x74]), "map", int(m[MAP_Y]), int(m[MAP_X]))
        if m[0x74] == 0x13:
            Path("cache/in-tourian.state").write_bytes(emulator.save()); print("in Tourian; saved cache/in-tourian.state"); break
