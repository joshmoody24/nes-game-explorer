"""Per-game lists of bytes that change mostly at camera jumps, and at screen-edge crossings,, from long random wanders from many saves.
Usage: python -m checks.cut_bytes_check <checkpoint.pkl.z> -> cache/cut_bytes.json, cache/edge_bytes.json. Metroid addresses only grade it."""
import json, os, pickle, sys, zlib
from pathlib import Path
import numpy as np
import place_extras
from emulator import Emulator
from explorers import ROM
e = Emulator(os.environ['QUICKNES_CORE'], ROM)
saves = pickle.loads(zlib.decompress(Path(sys.argv[1]).read_bytes()))['states']
rng = np.random.default_rng(0)
walks = []
for i in rng.choice(len(saves), 60, replace=False):
    place_extras.WANDER_WALKS, place_extras.WANDER_STEPS = 1, 400
    walks += place_extras.wander(e, saves[i], rng, True)
cuts = sum(place_extras.jumped(a[1], b[1]) for w in walks for a, b in zip(w, w[1:]))
chosen = place_extras.cut_bytes(walks)
edges = place_extras.edge_bytes(walks)
print(f'{cuts} camera jumps in {sum(len(w) for w in walks)} steps; cut bytes: {chosen}; screen-edge bytes: {edges}')
Path('cache/edge_bytes.json').write_text(json.dumps(edges))
from metroid import MAP_X, MAP_Y
print('map y, x =', MAP_Y, MAP_X, '; area = 116')
Path('cache/cut_bytes.json').write_text(json.dumps(chosen))
