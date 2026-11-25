"""
Go-Explore on Metroid: keep an archive with one save state per distinct place, repeatedly restore a
rarely-chosen one and press random buttons from there, and add any new places found.
"Place" here is hand-made from documented RAM addresses: the sanity baseline the learned versions
must approach.

Usage: python go_explore.py <run name> [--minutes M] [--workers N]
"""

import argparse
import json
import os
import pickle
import shutil
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from emulator import Emulator, button_mask
from metroid import (AREA, ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, ROOM_Y,
                     at, start_game)
from progress import Progress

ROM = "roms/metroid.nes"
RUNS = Path("runs")
MAX_RUNS_GIGABYTES = 50
FRAMES_PER_STEP = 4
STEPS_PER_SEGMENT = 100
# Go-Explore's random exploration repeats the previous action most of the time, so it makes
# sustained moves (a full jump, a run across a room) instead of jittering in place.
REPEAT_PROBABILITY = 0.9
PROGRESS_SAMPLE_EVERY = 25
LOG_EVERY_SECONDS = 10
CHECKPOINT_EVERY_SECONDS = 300
# START pauses the game and SELECT switches weapons; both are left out.
ACTIONS = [button_mask(*direction, *buttons)
           for direction in [(), ("LEFT",), ("RIGHT",), ("UP",), ("DOWN",)]
           for buttons in [(), ("A",), ("B",), ("A", "B")]]


def hand_made_place(memory):
    """Area, room, Samus's rough position in the room (32-pixel blocks), and progress counters."""
    return (int(memory[AREA]), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // 32, int(memory[ROOM_Y]) // 32,
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5)


def start_worker(seed):
    global emulator, rng
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])


def explore_from(start_state):
    """Random buttons from a saved state. Returns ({place: first state there}, Progress, frames used)."""
    emulator.load(start_state)
    found, progress = {}, Progress()
    action = rng.integers(len(ACTIONS))
    for step in range(STEPS_PER_SEGMENT):
        if rng.random() > REPEAT_PROBABILITY:
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        state = emulator.save()
        memory = emulator.memory(state)
        if memory[GAME_MODE] == GAME_OVER:
            break
        found.setdefault(hand_made_place(memory), state)
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(memory, emulator.nametable(state))
    return found, progress, (step + 1) * FRAMES_PER_STEP


@dataclass
class Archive:
    places: list
    states: list
    times_chosen: np.ndarray

    def add(self, place, state):
        self.places.append(place)
        self.states.append(state)
        self.times_chosen = np.append(self.times_chosen, 0)

    def choose(self, rng, count):
        weights = 1 / np.sqrt(1 + self.times_chosen)
        picks = rng.choice(len(self.places), count, p=weights / weights.sum())
        np.add.at(self.times_chosen, picks, 1)
        return [self.states[i] for i in picks]


def directory_gigabytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def checkpoint(run_dir, archive):
    """Plain data only, so a checkpoint loads without this module: {"places", "states", "times_chosen"}."""
    data = {"places": archive.places, "states": archive.states, "times_chosen": archive.times_chosen.tolist()}
    temporary = run_dir / "checkpoint.tmp"
    temporary.write_bytes(zlib.compress(pickle.dumps(data), level=1))
    temporary.rename(run_dir / "checkpoint.pkl.z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--minutes", type=float, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RUNS.mkdir(exist_ok=True)
    used = directory_gigabytes(RUNS)
    if used > MAX_RUNS_GIGABYTES:
        raise SystemExit(f"runs/ holds {used:.1f} GB, over the {MAX_RUNS_GIGABYTES} GB cap; clean up old runs first")
    run_dir = RUNS / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(vars(args) | {
        "place": "hand-made", "frames_per_step": FRAMES_PER_STEP, "steps_per_segment": STEPS_PER_SEGMENT,
        "repeat_probability": REPEAT_PROBABILITY}, indent=2))

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start = start_game(emulator)
    archive = Archive([hand_made_place(emulator.memory(start))], [start], np.zeros(1))
    progress = Progress()
    rng = np.random.default_rng(args.seed)
    frames = 0
    started = last_log = last_checkpoint = time.perf_counter()
    deadline = started + args.minutes * 60
    known = {archive.places[0]}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=start_worker,
                             initargs=(args.seed,)) as pool:
        pending = {pool.submit(explore_from, state) for state in archive.choose(rng, 2 * args.workers)}
        with open(run_dir / "metrics.jsonl", "a") as metrics:
            while time.perf_counter() < deadline:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    found, seen, used_frames = future.result()
                    frames += used_frames
                    progress.merge(seen)
                    for place, state in found.items():
                        if place not in known:
                            known.add(place)
                            archive.add(place, state)
                pending |= {pool.submit(explore_from, state) for state in archive.choose(rng, len(done))}
                now = time.perf_counter()
                if now - last_log > LOG_EVERY_SECONDS:
                    line = {"seconds": round(now - started), "frames": frames, "places": len(archive.places),
                            "frames per second": round(frames / (now - started))} | progress.summary()
                    metrics.write(json.dumps(line) + "\n")
                    metrics.flush()
                    print(line, flush=True)
                    last_log = now
                if now - last_checkpoint > CHECKPOINT_EVERY_SECONDS:
                    checkpoint(run_dir, archive)
                    last_checkpoint = now
            for future in pending:
                future.cancel()
    checkpoint(run_dir, archive)


if __name__ == "__main__":
    main()
