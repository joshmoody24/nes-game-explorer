"""
"Where am I" bytes found without knowing the game: bytes that change almost only at scene changes (steps where a
large share of memory changes at once, like a room loading) and almost never otherwise. Places built from those
bytes are compared with hand-made rooms on library states: purity and places per room.

Usage: python -m checks.scene_bytes_check <run>[+<run>...] [states]
"""

import os
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import library
from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import MAP_X, MAP_Y, at


def main():
    runs = sys.argv[1].split("+")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in runs for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    replayed = build(emulator, recordings[:3000], np.arange(len(emulator.memory())))
    changes = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0) != 0
                              for r in range(len(replayed.start_states))])
    share = changes.mean(1)
    scene = share >= np.quantile(share, 0.995)
    at_scene, elsewhere = changes[scene].mean(0), changes[~scene].mean(0)
    picks = random.sample(library.entries(), count)
    by_run = defaultdict(list)
    for e in picks:
        by_run[e["run"]].append(e)
    memories, rooms = [], []
    for run, group in by_run.items():
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        for e in group:
            memories.append(emulator.memory(saved["states"][e["slot"]]))
            rooms.append(e["room"])
    memories = np.stack(memories)
    for name, index in [("map y", MAP_Y), ("map x", MAP_X)]:
        print(f"{name} ({index:#06x}): changes at {at_scene[index]:.0%} of scene changes, {elsewhere[index]:.4%} of other steps")
    for min_scene, max_else in [(0.05, 0.0005), (0.1, 0.0002), (0.2, 0.0001), (0.02, 0.001)]:
        where = np.flatnonzero((at_scene >= min_scene) & (elsewhere <= max_else))
        places = [m[where].tobytes() for m in memories]
        rooms_of, places_of_room = defaultdict(Counter), defaultdict(set)
        for place, room in zip(places, rooms):
            rooms_of[place][room] += 1
            places_of_room[room].add(place)
        purity = sum(c.most_common(1)[0][1] for c in rooms_of.values()) / len(places)
        print(f"bytes changing at >= {min_scene:.0%} of scene changes and <= {max_else:.2%} of other steps: "
              f"{len(where)} bytes (map y/x included: {MAP_Y in where}/{MAP_X in where}); {len(rooms_of)} places for "
              f"{len(places_of_room)} rooms; purity {purity:.0%}; median places per room "
              f"{np.median([len(p) for p in places_of_room.values()]):.0f}", flush=True)


if __name__ == "__main__":
    main()
