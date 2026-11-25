"""
Races Go-Explore settings against each other: runs each setting with several seeds side by side
(splitting the machine's workers between them) and prints how long each took to reach each milestone.

Usage: python race.py <race name> <minutes> "<label>=<go_explore flags>" ["<label>=<flags>" ...]
Example: python race.py stale 2 "baseline=" "stale20=--stale-steps 20"
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

SEEDS = 3
WORKERS_PER_RUN = 6  # 3 runs at a time x 6 workers = 18, the throughput sweet spot measured on unit
MILESTONES = {
    "morph ball": lambda line: "morph ball" in line["items"],
    "missiles": lambda line: line["max missiles"] > 0,
    "64 rooms": lambda line: line["rooms"] >= 64,
    "Norfair": lambda line: "Norfair" in line["areas"],
    "128 rooms": lambda line: line["rooms"] >= 128,
    "bombs": lambda line: "bombs" in line["items"],
    "energy tank": lambda line: line["energy tanks"] > 0,
}


def first_time(lines, reached):
    return next((line["seconds"] for line in lines if reached(line)), None)


def run(race, label, flags, seed, minutes):
    name = f"{race}/{label}-seed{seed}"
    command = ["python", "go_explore.py", name, "--minutes", str(minutes), "--workers", str(WORKERS_PER_RUN),
               "--seed", str(seed), *flags.split()]
    return name, subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def main():
    race, minutes, entries = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    contestants = [entry.split("=", 1) for entry in entries]
    results = {}
    for label, flags in contestants:
        processes = [run(race, label, flags, seed, minutes) for seed in range(SEEDS)]
        for name, process in processes:
            _, errors = process.communicate()
            if process.returncode:
                raise SystemExit(f"{name} failed:\n{errors[-2000:]}")
        results[label] = [[json.loads(line) for line in open(Path("runs") / name / "metrics.jsonl")]
                          for name, _ in processes]

    print(f"median seconds to reach each milestone over {SEEDS} seeds ('-' = not reached by all), "
          f"plus final rooms and frames/sec")
    print(f"{'':>14}" + "".join(f"{milestone:>12}" for milestone in MILESTONES) + f"{'rooms':>8}{'frames/s':>10}")
    for label, runs in results.items():
        cells = []
        for reached in MILESTONES.values():
            times = [first_time(lines, reached) for lines in runs]
            cells.append("-" if None in times else f"{statistics.median(times):.0f}")
        rooms = statistics.median(lines[-1]["rooms"] for lines in runs)
        speed = statistics.median(lines[-1]["frames per second"] for lines in runs)
        print(f"{label:>14}" + "".join(f"{cell:>12}" for cell in cells) + f"{rooms:>8.0f}{speed:>10,.0f}")


if __name__ == "__main__":
    main()
