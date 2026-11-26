"""
Places found without knowing the game.

Scene bytes: bytes that change at scene changes (steps where a large share of memory changes at once) and, as
a set, almost never otherwise.

Position bytes: bytes that consistently differ between holding one direction and holding the opposite one, then
letting go (so copies of the buttons, which reset on release, drop out), over short and long holds. How often a
place changes during play is what decides the rest: each byte gets the smallest bucket at which it changes at
most every BYTE_DWELL steps (bytes that never get there are noise), and bytes are added strongest mover first
while the whole place changes at most every PLACE_DWELL steps.

Usage: python generic_places.py <run>[+<run>...]    (writes cache/generic-places.npz)
"""

import os
import pickle
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import library
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from experience import build, valid_recordings
from explorers import ROM

FOUND = Path("cache/generic-places.npz")
SCENE_SHARE, AT_SCENE, SET_ELSEWHERE = 0.9998, 0.05, 0.001
RELEASE, HOLDS, MOVED = 4, (8, 48), 0.25
AXES = (("RIGHT", "LEFT"), ("DOWN", "A"))
BYTE_DWELL = float(os.environ.get("BYTE_DWELL", 16))
PLACE_DWELL = float(os.environ.get("PLACE_DWELL", 8))
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def dwell(keys):
    """Mean steps between changes of a per-step key ([steps, ...] per recording)."""
    steps = sum(len(k) - 1 for k in keys)
    changes = sum(int(np.count_nonzero(np.any(np.diff(k, axis=0) != 0, axis=1))) for k in keys)
    return steps / max(changes, 1)


def scene_bytes(memories):
    """Bytes that change at scene changes, chosen so that together they rarely change otherwise: candidates
    are added rarest-changing first while the whole set changes outside scene changes at most SET_ELSEWHERE."""
    changes = np.concatenate([np.diff(m.astype(np.int16), axis=0) != 0 for m in memories])
    share = changes.mean(1)
    scene = share >= np.quantile(share, SCENE_SHARE)
    at_scene, elsewhere = changes[scene].mean(0), changes[~scene].mean(0)
    candidates = np.flatnonzero(at_scene >= AT_SCENE)
    candidates = candidates[np.argsort(elsewhere[candidates], kind="stable")]
    set_rate = np.logical_or.accumulate(changes[~scene][:, candidates], axis=1).mean(0)
    return np.sort(candidates[:np.searchsorted(set_rate, SET_ELSEWHERE, side="right")])


def after(emulator, state, buttons, hold):
    emulator.load(state)
    for _ in range(hold):
        emulator.run(buttons, FRAMES_PER_STEP)
    for _ in range(RELEASE):
        emulator.run(0, FRAMES_PER_STEP)
    return emulator.memory().astype(int)


def movers(emulator, states):
    """{byte: strongest score} for bytes that holding a direction moves consistently (score >= MOVED)."""
    best = {}
    for plus, minus in AXES:
        for hold in HOLDS:
            signs = np.sign(np.stack([after(emulator, s, button_mask(plus), hold) - after(emulator, s, button_mask(minus), hold)
                                      for s in states]))
            score = np.abs(signs.mean(0)) * (signs != 0).mean(0)
            for b in np.flatnonzero(score >= MOVED):
                best[int(b)] = max(best.get(int(b), 0.0), float(score[b]))
    return best


def position_bytes(movers, memories, scene):
    """[(byte, bucket)] chosen by dwell (see the module docstring), and how often that place changes."""
    scene_keys = [np.array([[zlib.crc32(row.tobytes())] for row in m[:, scene]], np.int64) for m in memories]

    def keys(chosen):
        return [np.concatenate([s] + [m[:, [b]].astype(np.int64) // k for b, k in chosen], axis=1)
                for s, m in zip(scene_keys, memories)]

    chosen = []
    for b in sorted(movers, key=lambda b: -movers[b]):
        bucket = next((k for k in BUCKETS if dwell([m[:, [b]] // k for m in memories]) >= BYTE_DWELL), None)
        if bucket and dwell(keys(chosen + [(b, bucket)])) >= PLACE_DWELL:
            chosen.append((b, bucket))
    return chosen, dwell(keys(chosen))


def library_states(count):
    by_run = defaultdict(list)
    for e in random.sample(library.entries(), count):
        by_run[e["run"]].append(e)
    states = []
    for run, group in by_run.items():
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        states += [saved["states"][e["slot"]] for e in group]
    return states


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in sys.argv[1].split("+") for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    replayed = build(emulator, recordings[:1500], np.arange(len(emulator.memory())))
    memories = [replayed.recording(r)[0] for r in range(len(replayed.start_states))]
    scene = scene_bytes(memories)
    position, place_dwell = position_bytes(movers(emulator, library_states(800)), memories, scene)
    np.savez(FOUND, scene=scene, position=np.array(position))
    print(f"{len(scene)} scene bytes; position (byte, bucket) {[(hex(b), k) for b, k in position]}; "
          f"place changes every {place_dwell:.1f} steps of play -> {FOUND}")


if __name__ == "__main__":
    main()
