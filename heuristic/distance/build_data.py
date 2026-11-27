"""
Replays stored button paths of fresh bench runs into RAM trajectories, for learning "how many steps apart".
Per run writes data/<run>.npz: trajectory RAM sampled every EVERY steps (with trajectory id and step index), the RAM
of every save, and what's needed to score stepping stones later.
Usage: TURBO=1 python heuristic/distance/build_data.py
"""
import glob, json, os, pickle, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from emulator import Emulator
from explorers import ROM
from controls import ACTIONS, FRAMES_PER_STEP
from bench import WALLS

EVERY = 4
TRAJECTORIES_PER_RUN = 40
OUT = Path(__file__).parent / "data"


def build(run):
    name = "_".join(Path(run).parts[-3:])
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
    start = Path(json.loads((Path(run) / "config.json").read_text())["start_state"]).read_bytes()
    states, paths, parents = saved["states"], saved["paths"], saved["parents"]
    test = WALLS[Path(run).name][1]
    first = next((i for i, s in enumerate(states) if test(emulator.view(s))), -1)
    rng = np.random.default_rng(len(states))
    replayable = [i for i, p in enumerate(paths) if p is not None and len(p) >= 2 * EVERY]
    chosen = rng.choice(replayable, min(TRAJECTORIES_PER_RUN, len(replayable)), replace=False)
    memories, trajectory, step = [], [], []
    for t, i in enumerate(chosen):
        emulator.load(start)
        memories.append(emulator.memory()); trajectory.append(t); step.append(0)
        for k, a in enumerate(paths[i], 1):
            emulator.run(ACTIONS[a], FRAMES_PER_STEP)
            if k % EVERY == 0:
                memories.append(emulator.memory()); trajectory.append(t); step.append(k)
    np.savez_compressed(OUT / f"{name}.npz", memory=np.array(memories), trajectory=np.array(trajectory), step=np.array(step),
                        save_memory=np.array([emulator.memory(s) for s in states]), start_memory=emulator.memory(start),
                        parents=np.array([-1 if p is None else p for p in parents]), first=first,
                        times_chosen=np.array(saved["times_chosen"]), path_length=np.array([-1 if p is None else len(p) for p in paths]))
    return f"{name}: {len(states)} saves, first success {first}, {len(memories)} trajectory states"


if __name__ == "__main__":
    runs = sorted(str(Path(p).parent) for p in glob.glob("runs/bench/*/s*/*/checkpoint.pkl.z"))
    with Pool(12) as pool:
        for line in pool.imap_unordered(build, runs):
            print(line, flush=True)
