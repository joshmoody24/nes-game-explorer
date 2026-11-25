"""
Go-Explore on Metroid: keep an archive with one save state per distinct place, repeatedly restore a
rarely-chosen one and press random buttons from there, and add any new places found.
"Place" here is hand-made from documented RAM addresses: the sanity baseline the learned versions
must approach.

Usage: python go_explore.py <run name> [--minutes M] [--workers N] [settings flags, see --help]
"""

import argparse
import dataclasses
import json
import os
import pickle
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emulator import Emulator
from explorers import ROM, Explorers, Settings, hand_made_place
from metroid import start_game
from progress import Progress

RUNS = Path("runs")
MAX_RUNS_GIGABYTES = 50
LOG_EVERY_SECONDS = 2
CHECKPOINT_EVERY_SECONDS = 300


@dataclass
class Archive:
    selection: str
    places: list = field(default_factory=list)
    states: list = field(default_factory=list)
    times_chosen: list = field(default_factory=list)
    discoveries: list = field(default_factory=list)  # new places found by segments started here

    def add(self, place, state):
        self.places.append(place)
        self.states.append(state)
        self.times_chosen.append(0)
        self.discoveries.append(0)

    def choose(self, rng):
        chosen = np.array(self.times_chosen)
        weights = 1 / np.sqrt(1 + chosen)
        if self.selection == "productive":
            weights *= 1 + np.array(self.discoveries) / (1 + chosen)
        index = int(rng.choice(len(self.places), p=weights / weights.sum()))
        self.times_chosen[index] += 1
        return index


def directory_gigabytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def checkpoint(run_dir, archive):
    """Plain data only, so a checkpoint loads without this module."""
    temporary = run_dir / "checkpoint.tmp"
    temporary.write_bytes(zlib.compress(pickle.dumps(dataclasses.asdict(archive)), level=1))
    temporary.rename(run_dir / "checkpoint.pkl.z")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--minutes", type=float, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    for setting in dataclasses.fields(Settings):
        kind = int if setting.default is None else type(setting.default)  # the only optional setting is an int
        parser.add_argument("--" + setting.name.replace("_", "-"), type=kind, default=setting.default)
    args = parser.parse_args()
    settings = Settings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)})
    return args, settings


def prepare_run_dir(args, settings):
    RUNS.mkdir(exist_ok=True)
    used = directory_gigabytes(RUNS)
    if used > MAX_RUNS_GIGABYTES:
        raise SystemExit(f"runs/ holds {used:.1f} GB, over the {MAX_RUNS_GIGABYTES} GB cap; clean up old runs first")
    run_dir = RUNS / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"minutes": args.minutes, "workers": args.workers, "seed": args.seed} | dataclasses.asdict(settings),
        indent=2))
    return run_dir


def main():
    args, settings = parse_arguments()
    run_dir = prepare_run_dir(args, settings)
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start = start_game(emulator)
    archive = Archive(settings.selection)
    archive.add(hand_made_place(emulator.memory(start), settings.block_size), start)
    known = set(archive.places)
    progress = Progress()
    rng = np.random.default_rng(args.seed)
    frames = 0
    gaps = []
    explorers = Explorers(args.workers, settings, args.seed)

    def assign(worker_index):
        index = archive.choose(rng)
        explorers.send(worker_index, index, archive.states[index], archive.places)

    for worker_index in range(explorers.count):
        assign(worker_index)
    started = last_log = last_checkpoint = time.perf_counter()
    deadline = started + args.minutes * 60
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        while time.perf_counter() < deadline:
            for worker_index, segment in explorers.finished():
                frames += segment.frames
                progress.merge(segment.progress)
                gaps.extend(segment.gaps)
                for place, state in segment.found.items():
                    if place not in known:
                        known.add(place)
                        archive.add(place, state)
                        archive.discoveries[segment.start] += 1
                assign(worker_index)
            now = time.perf_counter()
            if now - last_log > LOG_EVERY_SECONDS:
                line = {"seconds": round(now - started, 1), "frames": frames, "places": len(archive.places),
                        "frames per second": round(frames / (now - started)),
                        "steps between discoveries (50/90/99%)": np.percentile(gaps, [50, 90, 99]).round().tolist()
                        if gaps else None} | progress.summary()
                metrics.write(json.dumps(line) + "\n")
                metrics.flush()
                last_log = now
            if now - last_checkpoint > CHECKPOINT_EVERY_SECONDS:
                checkpoint(run_dir, archive)
                last_checkpoint = now
    explorers.close()
    checkpoint(run_dir, archive)
    print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
