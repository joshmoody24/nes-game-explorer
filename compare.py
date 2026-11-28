"""
Paired comparison of explorer configs ("arms") against one baseline on the wall benchmark, built for clean stats with
few iterations:
- many 1-worker runs in parallel (each exactly reproducible), so cores buy independent samples, not one big run;
- every arm runs the same seeds as the baseline, so seed s starts from the same random choices in both (pairing);
- each run records frames to break its wall, or None for "more than the cap", not just pass/fail, and how many map
  cells it reached (read from the game's map bytes by the referee only; the explorer never sees it): a continuous
  score that separates configs with far fewer seeds than pass/fail on a hard wall;
- runs go in batches; after each, a bootstrap interval on walls broken per seed (arm minus baseline) settles an arm
  when it excludes 0 (better / worse) or fits inside +-MARGIN (no difference worth having). An arm is only ever
  written off (worse, or no difference) after MIN_SEEDS_TO_REJECT seeds, so luck can't kill an idea.
Usage: python compare.py <name> "baseline=<args>" "arm1=<args>" ["arm2=<args>" ...] -- [wall ...]
Env: COMPARE_PARALLEL (runs at once, default 15), COMPARE_BATCH (seeds per batch, default 15), COMPARE_MAX_BATCHES,
BENCH_MAX_FRAMES (cap per wall run, default 2e6).
"""
import json, os, pickle, subprocess, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bench import WALLS
from emulator import Emulator
from explorers import ROM

PARALLEL = int(os.environ.get("COMPARE_PARALLEL", 15))
BATCH = int(os.environ.get("COMPARE_BATCH", 15))
MAX_BATCHES = int(os.environ.get("COMPARE_MAX_BATCHES", 4))
MAX_FRAMES = int(float(os.environ.get("BENCH_MAX_FRAMES", 2e6)))
MARGIN = 0.25  # walls per seed: a smaller difference isn't worth having
MIN_SEEDS_TO_REJECT = 30
RUN_MEMORY = os.environ.get("COMPARE_RUN_MEMORY", "1500M")  # per run: a scheme whose places explode kills only itself
COMMON = "--turbo --repeat-probability 0.99 --repeat-spread 0.09 --novel-values --keep-more 4217,263 --workers 1"


def map_cells(emulator, states):
    from metroid import MAP_X, MAP_Y
    return len({(int(emulator.view(s)[0x74]), int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) for s in states})


def run_once(name, label, args, wall, seed):
    """(frames to break wall from its start save, None if not within MAX_FRAMES, or "crashed"; map cells reached)."""
    run = f"compare/{name}/{label}/{wall}/s{seed}"
    directory = Path("runs") / run
    cached = directory / "result.json"
    if cached.exists():  # a rerun of the same comparison reuses finished runs (they are exactly reproducible)
        result = json.loads(cached.read_text())
        if "cells" in result:
            return result["frames"], result["cells"]
    start, test = WALLS[wall]
    subprocess.run(["tools/capped", RUN_MEMORY, "python", "-u", "go_explore.py", run, "--minutes", "60", "--seed", str(seed), "--start-state", start,
                    "--stop-when", wall, "--max-frames", str(MAX_FRAMES)] + COMMON.split() + args.split(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "PYTHONHASHSEED": "0", "LONG": "compare run", "LONG_MINUTES": "20"})
    if not (directory / "checkpoint.pkl.z").exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"frames": "crashed", "cells": 0}))  # e.g. killed at RUN_MEMORY: counts as not breaking the wall
        return "crashed", 0
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress((directory / "checkpoint.pkl.z").read_bytes()))["states"]
    frames = int(np.load(directory / "segments.npz")["frames"].sum()) if any(test(emulator.view(s)) for s in states) else None
    cells = map_cells(emulator, states)
    cached.write_text(json.dumps({"frames": frames, "cells": cells}))
    (directory / "checkpoint.pkl.z").unlink()  # the result is all a comparison needs; checkpoints are big
    return frames, cells


def bootstrap(differences, rng, draws=10000):
    means = rng.choice(differences, (draws, len(differences))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def broken(frames):
    return isinstance(frames, int)


if __name__ == "__main__":
    name, rest = sys.argv[1], sys.argv[2:]
    split = rest.index("--") if "--" in rest else len(rest)
    arms = dict(item.split("=", 1) for item in rest[:split])
    walls = rest[split + 1:] or list(WALLS)
    baseline = next(iter(arms))
    print(f"{name}: baseline {baseline} vs {list(arms)[1:]}, walls {walls}, cap {MAX_FRAMES / 1e6:.1f}M frames", flush=True)
    results, cells, verdicts, rng = {}, {}, {}, np.random.default_rng(0)
    for batch in range(MAX_BATCHES):
        open_arms = [label for label in arms if label == baseline or label not in verdicts]
        if open_arms == [baseline]:
            break
        seeds = range(1 + batch * BATCH, 1 + (batch + 1) * BATCH)
        jobs = [(label, wall, seed) for seed in seeds for wall in walls for label in open_arms]
        with ThreadPoolExecutor(PARALLEL) as pool:
            for job, (frames, reached) in zip(jobs, pool.map(lambda j: run_once(name, j[0], arms[j[0]], j[1], j[2]), jobs)):
                results[job], cells[job] = frames, reached
        print(f"--- after {seeds[-1]} seeds", flush=True)
        for label in open_arms:
            seen = sorted({s for l, _, s in results if l == label})
            per_wall = []
            for wall in walls:
                got = [results[(label, wall, s)] for s in seen]
                times = sorted(f for f in got if broken(f))
                per_wall.append(f"{wall} {len(times)}/{len(got)}" + (f" med {np.median(times) / 1e6:.2f}M" if times else ""))
            line = f"  {label}: " + " | ".join(per_wall)
            if label != baseline:
                difference = np.array([sum(broken(results[(label, w, s)]) for w in walls) -
                                       sum(broken(results[(baseline, w, s)]) for w in walls) for s in seen])
                low, high = bootstrap(difference, rng)
                line += f" || vs {baseline}: {difference.mean():+.2f} walls/seed (95% {low:+.2f} to {high:+.2f})"
                more = np.array([cells[(label, w, s)] - cells[(baseline, w, s)] for w in walls for s in seen])
                cell_low, cell_high = bootstrap(more, rng)
                line += f", {more.mean():+.1f} map cells/run (95% {cell_low:+.1f} to {cell_high:+.1f})"
                if low > 0:
                    verdicts[label] = "better"
                elif len(seen) >= MIN_SEEDS_TO_REJECT and high < 0:
                    verdicts[label] = "worse"
                elif len(seen) >= MIN_SEEDS_TO_REJECT and -MARGIN < low and high < MARGIN:
                    verdicts[label] = "no difference"
                if label in verdicts:
                    line += f"  => {verdicts[label]}"
            print(line, flush=True)
        Path(f"runs/compare/{name}/results.json").write_text(json.dumps({f"{l}|{w}|{s}": f for (l, w, s), f in results.items()}))
    for label in arms:
        if label != baseline:
            print(f"verdict {label}: {verdicts.get(label, 'undecided')}", flush=True)
    print("COMPARE DONE", flush=True)
