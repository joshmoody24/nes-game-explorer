"""
Wall time of each stage of a small player experiment, to see what dominates iteration.

Usage: python -m checks.stage_timing_check <run name>
"""

import dataclasses
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

import distance
import policy
import trips
from emulator import Emulator
from experience import build, choose_bytes, valid_recordings
from explorers import ROM


def timed(name, compute):
    started = time.perf_counter()
    result = compute()
    print(f"{name}: {time.perf_counter() - started:.1f}s", flush=True)
    return result


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = timed("load recordings", lambda: valid_recordings(Path("runs") / sys.argv[1]))
    np.random.default_rng(0).shuffle(recordings)
    training, held_out = recordings[:4000], recordings[60000:]
    byte_indices = timed("choose bytes", lambda: choose_bytes(emulator, training))
    evaluation = timed("sample 200 trips", lambda: trips.from_recordings(held_out, np.random.default_rng(1), 200))
    experience = timed("build 4K recordings", lambda: build(emulator, training, byte_indices, on_device=True))
    player = timed("train player 5K updates", lambda: policy.train(experience, jax.random.key(0), 5000, log=lambda _: None))
    judge = timed("train distance 5K updates", lambda: distance.train(experience, jax.random.key(0), 5000, log=lambda _: None))
    timed("18 one-step trips (pool start)", lambda: trips.run([dataclasses.replace(t, budget=1, player="random") for t in evaluation[:18]]))
    timed("18 one-step learned trips (pool start + JAX)", lambda: trips.run([dataclasses.replace(t, budget=1) for t in evaluation[:18]], player))
    timed("200 random trips", lambda: trips.run([dataclasses.replace(t, player="random") for t in evaluation]))
    timed("200 learned trips", lambda: trips.run(evaluation, player))
    timed("200 search-learned trips", lambda: trips.run([dataclasses.replace(t, player="search-learned") for t in evaluation], judge, search_hold=4))


if __name__ == "__main__":
    main()
