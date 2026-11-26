"""
Places found without knowing the game. Scene bytes: bytes that change at scene changes (steps where a large
share of memory changes at once) and almost never otherwise; their values together say which room this is.
Position bytes: per axis, the byte that most consistently differs between holding one direction and holding the
opposite one, then letting go (so copies of the buttons, which reset on release, drop out).

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
SCENE_SHARE = float(os.environ.get("SCENE_SHARE", 0.995))
AT_SCENE = 0.05
SET_ELSEWHERE = float(os.environ.get("SET_ELSEWHERE", 0.001))
HOLD, RELEASE = 8, 4
AXES = (("RIGHT", "LEFT"), ("DOWN", "A"))


def scene_bytes(emulator, recordings):
    """Bytes that change at scene changes, chosen so that together they rarely change otherwise: candidates
    are added rarest-changing first while the whole set changes outside scene changes at most SET_ELSEWHERE."""
    replayed = build(emulator, recordings, np.arange(len(emulator.memory())))
    changes = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0) != 0
                              for r in range(len(replayed.start_states))])
    share = changes.mean(1)
    scene = share >= np.quantile(share, SCENE_SHARE)
    at_scene, elsewhere = changes[scene].mean(0), changes[~scene].mean(0)
    candidates = np.flatnonzero(at_scene >= AT_SCENE)
    candidates = candidates[np.argsort(elsewhere[candidates], kind="stable")]
    set_rate = np.logical_or.accumulate(changes[~scene][:, candidates], axis=1).mean(0)
    return np.sort(candidates[:np.searchsorted(set_rate, SET_ELSEWHERE, side="right")])


def after(emulator, state, buttons):
    emulator.load(state)
    for _ in range(HOLD):
        emulator.run(buttons, FRAMES_PER_STEP)
    for _ in range(RELEASE):
        emulator.run(0, FRAMES_PER_STEP)
    return emulator.memory().astype(int)


def position_bytes(emulator, states):
    found = []
    for plus, minus in AXES:
        signs = np.stack([np.sign(after(emulator, s, button_mask(plus)) - after(emulator, s, button_mask(minus)))
                          for s in states])
        found.append(int(np.argmax(np.abs(signs.mean(0)) * (signs != 0).mean(0))))
    return np.array(found)


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
    scene = scene_bytes(emulator, recordings[:3000])
    position = position_bytes(emulator, library_states(1500))
    np.savez(FOUND, scene=scene, position=position)
    print(f"{len(scene)} scene bytes; position bytes {list(position)} -> {FOUND}")


if __name__ == "__main__":
    main()
