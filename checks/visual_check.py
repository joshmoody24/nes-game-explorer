"""
Could tiny screenshots replace hand-made places (Go-Explore's original generic cells)? (1) Speed: emulated
steps per second with and without capturing one frame per step. (2) Agreement: renders library states,
downscales each frame to WIDTH x HEIGHT grayscale with LEVELS shades, and compares the resulting "frame places"
with hand-made rooms: purity (how often a frame place stays within one room) and fragmentation (frame places
per room).

Usage: python -m checks.visual_check [states]
"""

import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import library
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM

WIDTH, HEIGHT, LEVELS = 11, 8, 8


def frame_place(frame, width=WIDTH, height=HEIGHT, levels=LEVELS):
    gray = frame.mean(2)
    h, w = gray.shape
    small = gray[:h - h % height, :w - w % width].reshape(height, h // height, width, w // width).mean((1, 3))
    return tuple((small * levels / 256).astype(int).ravel())


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    entries = library.entries()
    random.seed(0)
    picks = random.sample(entries, count)
    by_run = defaultdict(list)
    for e in picks:
        by_run[e["run"]].append(e)
    states, rooms = [], []
    import pickle, zlib
    from pathlib import Path
    for run, group in by_run.items():
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        for e in group:
            states.append(saved["states"][e["slot"]])
            rooms.append(e["room"])
    rng = np.random.default_rng(0)
    emulator.load(states[0])
    started = time.perf_counter()
    for _ in range(2000):
        emulator.run(ACTIONS[int(rng.integers(len(ACTIONS)))], FRAMES_PER_STEP)
    plain = 2000 / (time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(2000):
        emulator.run(ACTIONS[int(rng.integers(len(ACTIONS)))], FRAMES_PER_STEP - 1)
        frame_place(emulator.frame(0))
    with_frames = 2000 / (time.perf_counter() - started)
    if False: print(f"speed: {plain:,.0f} steps/s plain, {with_frames:,.0f} with a frame place every step "
          f"({with_frames / plain:.0%})", flush=True)
    for width, height, levels in [(11, 8, 4), (11, 8, 3), (11, 8, 2), (8, 6, 8), (6, 5, 8), (16, 15, 2), (5, 4, 8)]:
        places = []
        for state in states:
            emulator.load(state)
            places.append(frame_place(emulator.frame(0), width, height, levels))
        rooms_of = defaultdict(Counter)
        places_of_room = defaultdict(set)
        for place, room in zip(places, rooms):
            rooms_of[place][room] += 1
            places_of_room[room].add(place)
        purity = sum(c.most_common(1)[0][1] for c in rooms_of.values()) / len(places)
        print(f"{width}x{height}, {levels} shades: {len(rooms_of)} frame places for {len(places_of_room)} rooms; "
              f"purity {purity:.0%} (states whose frame place's main room is their room); "
              f"median frame places per room {np.median([len(p) for p in places_of_room.values()]):.0f}", flush=True)


if __name__ == "__main__":
    main()
