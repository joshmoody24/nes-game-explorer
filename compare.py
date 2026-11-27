"""
Paired comparison of explorer configs on the wall benchmark, built for clean stats with few iterations:
- many 1-worker runs in parallel (each exactly reproducible), so cores buy independent samples, not one big run;
- every config runs the same seeds, so seed s under A and under B start from the same random choices (pairing);
- each run records frames to break its wall, or "more than the cap" (censored), not just pass/fail;
- runs go in batches; after each, a bootstrap interval on the main number (walls broken, A minus B, per seed) decides
  whether to stop: it excludes 0 (a winner), or fits inside +-MARGIN (no difference worth having), or MAX_BATCHES.
Usage: python compare.py <name> "<config A args>" "<config B args>" [wall ...]
Env: COMPARE_PARALLEL (runs at once, default 15), COMPARE_BATCH (seeds per batch, default 15), BENCH_MAX_FRAMES.
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
COMMON = "--turbo --repeat-probability 0.99 --repeat-spread 0.09 --novel-values --keep-more 4217,263 --workers 1"


def run_once(name, label, args, wall, seed):
    """Frames to break wall from its start save, or None if not within MAX_FRAMES."""
    run = f"compare/{name}/{label}/{wall}/s{seed}"
    start, _ = WALLS[wall]
    subprocess.run(["python", "-u", "go_explore.py", run, "--minutes", "60", "--seed", str(seed), "--start-state", start,
                    "--stop-when", wall, "--max-frames", str(MAX_FRAMES)] + COMMON.split() + args.split(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONHASHSEED": "0"})
    directory = Path("runs") / run
    if not (directory / "checkpoint.pkl.z").exists():
        return "crashed"
    test, emulator = WALLS[wall][1], Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = pickle.loads(zlib.decompress((directory / "checkpoint.pkl.z").read_bytes()))["states"]
    broke = any(test(emulator.view(s)) for s in states)
    return int(np.load(directory / "segments.npz")["frames"].sum()) if broke else None


def bootstrap(differences, rng, draws=10000):
    means = rng.choice(differences, (draws, len(differences))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summary(results, labels, walls):
    for label in labels:
        cells = []
        for wall in walls:
            frames = [f for (l, w, s), f in results.items() if l == label and w == wall]
            broke = sorted(f for f in frames if isinstance(f, int))
            cells.append(f"{wall} {len(broke)}/{len(frames)}" + (f" med {np.median(broke) / 1e6:.2f}M" if broke else ""))
        print(f"  {label}: " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    name, configs, walls = sys.argv[1], {"A": sys.argv[2], "B": sys.argv[3]}, sys.argv[4:] or list(WALLS)
    print(f"{name}: A = '{configs['A']}' vs B = '{configs['B']}', walls {walls}, cap {MAX_FRAMES / 1e6:.1f}M frames", flush=True)
    results, rng = {}, np.random.default_rng(0)
    for batch in range(MAX_BATCHES):
        seeds = range(1 + batch * BATCH, 1 + (batch + 1) * BATCH)
        jobs = [(label, wall, seed) for seed in seeds for wall in walls for label in configs]
        with ThreadPoolExecutor(PARALLEL) as pool:
            for job, frames in zip(jobs, pool.map(lambda j: run_once(name, j[0], configs[j[0]], j[1], j[2]), jobs)):
                results[job] = frames
        all_seeds = sorted({s for _, _, s in results})
        walls_broken = {label: np.array([sum(isinstance(results[(label, w, s)], int) for w in walls) for s in all_seeds])
                        for label in configs}
        difference = walls_broken["A"] - walls_broken["B"]
        low, high = bootstrap(difference, rng)
        print(f"after {len(all_seeds)} seeds: walls broken per seed A {walls_broken['A'].mean():.2f}, B {walls_broken['B'].mean():.2f}, "
              f"A-B {difference.mean():+.2f} (95% {low:+.2f} to {high:+.2f})", flush=True)
        summary(results, configs, walls)
        Path(f"runs/compare/{name}/results.json").write_text(json.dumps({f"{l}|{w}|{s}": f for (l, w, s), f in results.items()}))
        if low > 0 or high < 0:
            print(f"verdict: {'A' if low > 0 else 'B'} breaks more walls", flush=True); break
        if -MARGIN < low and high < MARGIN:
            print(f"verdict: no difference bigger than {MARGIN} walls per seed", flush=True); break
    else:
        print("verdict: undecided after the batch limit", flush=True)
    print("COMPARE DONE", flush=True)
