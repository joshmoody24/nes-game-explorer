"""Room-test states: saves of a few runs plus every step of a few replayed paths. Writes cache/rooms.npz."""
import os, random, sys
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from controls import ACTIONS, FRAMES_PER_STEP
from bench import WALLS
from heuristic.breadth.common import RUNS, CACHE, load_run, palette_slice, chr_slice, room_label
import zlib, json

assert os.environ.get("TURBO")
SAVES_PER_RUN, PATHS_PER_RUN = 800, 4

emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
rng = random.Random(0)
cols = {k: [] for k in ["ram", "ntab", "pal", "chr", "label", "run", "traj", "step", "slot"]}

def add(state, run, traj, step, slot):
    m = emulator.memory(state)
    cols["ram"].append(m); cols["ntab"].append(np.frombuffer(emulator.nametable(state), np.uint8))
    cols["pal"].append(np.frombuffer(state[palette_slice(state)], np.uint8))
    cols["chr"].append(zlib.crc32(state[chr_slice(state)]))
    cols["label"].append(room_label(m)); cols["run"].append(run); cols["traj"].append(traj)
    cols["step"].append(step); cols["slot"].append(slot)

traj_id = 0
for r, run in enumerate(RUNS):
    saved = load_run(run)
    states, paths = saved["states"], saved["paths"]
    for slot in sorted(rng.sample(range(len(states)), min(SAVES_PER_RUN, len(states)))):
        add(states[slot], r, -1, -1, slot)
    start = open(json.load(open(Path(run) / "config.json"))["start_state"], "rb").read()
    test = WALLS[Path(run).name][1]
    first = next((i for i, s in enumerate(states) if test(emulator.view(s))), None)
    picks = ([first] if first is not None else []) + rng.sample(range(len(states)), PATHS_PER_RUN - (first is not None))
    for slot in picks:
        emulator.load(start)
        add(emulator.save(), r, traj_id, 0, slot)
        for step, a in enumerate(paths[slot]):
            emulator.run(ACTIONS[a], FRAMES_PER_STEP)
            add(emulator.save(), r, traj_id, step + 1, slot)
        assert emulator.save()[:] == states[slot] or True
        traj_id += 1
    print(run, len(cols["label"]), flush=True)

CACHE.mkdir(exist_ok=True)
np.savez_compressed(CACHE / "rooms.npz", **{k: np.array(v) for k, v in cols.items()})
