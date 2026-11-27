"""Saves where Samus plays in a map cell the ROM world map says is empty (0xFF): candidate hidden worlds."""
import os, pickle, zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y

WORLD = np.frombuffer(Path(ROM).read_bytes()[0x254E:0x254E + 1024], np.uint8).reshape(32, 32)

def scan(p):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    try:
        states = pickle.loads(zlib.decompress(p.read_bytes()))["states"]
    except Exception:
        return str(p.parent), Counter(), []
    found, keep = Counter(), []
    for s in states:
        v = emulator.view(s)
        if int(v[0x1E]) != 3: continue
        y, x = int(v[MAP_Y]), int(v[MAP_X])
        if y < 32 and x < 32 and WORLD[y, x] == 0xFF:
            c = (hex(int(v[0x74]) or 0x10), y, x); found[c] += 1
            if found[c] <= 2: keep.append((c, s))
    return str(p.parent), found, keep

if __name__ == "__main__":
    out = Path("/tmp/hidden"); out.mkdir(exist_ok=True)
    total = Counter(); runs = Counter(); kept = {}
    with ProcessPoolExecutor(4) as pool:
        for run, found, keep in pool.map(scan, sorted(Path("runs").glob("**/checkpoint.pkl.z"))):
            if found:
                runs[run] = sum(found.values()); total.update(found)
                for c, s in keep: kept.setdefault(c, (run, s))
    print("runs with saves in empty map cells:", runs.most_common(12))
    print("cells (area, y, x): saves", total.most_common(25))
    pickle.dump(kept, open(out / "kept.pkl", "wb"))
