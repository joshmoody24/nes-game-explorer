"""
World-map atlas: one screenshot per map cell, from any saved run that reached it.
Border: green = an uncheated run reached it, orange = only other runs, grey = on the ROM map but never reached.
Usage: python atlas.py <out dir>
"""
import os, pickle, subprocess, sys, zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y

AREAS = {0x10: "brinstar", 0x11: "norfair", 0x12: "kraid", 0x13: "tourian", 0x14: "ridley"}
UNCHEATED = {"honest1", "honest2", "honest3", "honest4", "honest5", "honest6", "kh-honest", "kh-layer", "honest-kraid"}
SCALE = 2
WORLD_MAP = 0x254E

def cells_of(run_dir):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    try:
        states = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))["states"]
    except Exception:
        return run_dir.name, {}
    found = {}
    for s in states:
        v = emulator.view(s)
        area = int(v[0x74]) or 0x10
        cell = (area, int(v[MAP_Y]), int(v[MAP_X]))
        if area in AREAS and cell not in found and int(v[0x1E]) == 3:
            found[cell] = s
    return run_dir.name, found

def write_png(image, path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{image.shape[1]}x{image.shape[0]}",
                    "-i", "-", "-frames:v", "1", str(path)], input=image.tobytes(), check=True)

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rom = Path(ROM).read_bytes()
    world = np.frombuffer(rom[WORLD_MAP:WORLD_MAP + 1024], np.uint8).reshape(32, 32)
    cache = out / "shots.pkl"
    if cache.exists():
        shots, uncheated = pickle.loads(cache.read_bytes())
    else:
        shots, uncheated = {}, set()
        with ProcessPoolExecutor(6) as pool:
            for name, found in pool.map(cells_of, sorted(p for p in Path("runs").iterdir() if (p / "checkpoint.pkl.z").exists())):
                for cell, s in found.items():
                    shots.setdefault(cell, s)
                    if name in UNCHEATED: uncheated.add(cell)
        cache.write_bytes(pickle.dumps((shots, uncheated)))
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    h, w = 224 // SCALE, 240 // SCALE
    for area, name in AREAS.items():
        cells = [c[1:] for c in shots if c[0] == area]
        if not cells: continue
        ys, xs = [c[0] for c in cells], [c[1] for c in cells]
        y0, y1, x0, x1 = max(min(ys) - 1, 0), min(max(ys) + 1, 31), max(min(xs) - 1, 0), min(max(xs) + 1, 31)
        image = np.full(((y1 - y0 + 1) * h, (x1 - x0 + 1) * w, 3), 20, np.uint8)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                top, left = (y - y0) * h, (x - x0) * w
                cell = (area, y, x)
                if cell in shots:
                    emulator.load(shots[cell])
                    image[top:top + h, left:left + w] = emulator.frame(0)[:224:SCALE, :240:SCALE][:h, :w]
                    color = (40, 200, 90) if cell in uncheated else (230, 140, 30)
                elif world[y, x] != 0xFF:
                    color = (110, 110, 110)
                else:
                    continue
                image[top:top + 2, left:left + w] = color; image[top + h - 2:top + h, left:left + w] = color
                image[top:top + h, left:left + 2] = color; image[top:top + h, left + w - 2:left + w] = color
        write_png(image, out / f"{name}.png")
        print(f"{name}: {len(cells)} cells reached ({sum(1 for c in uncheated if c[0] == area)} uncheated), rows {y0}-{y1}, cols {x0}-{x1} -> {out / name}.png")
