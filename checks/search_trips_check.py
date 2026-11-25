"""
How good is a search player that looks ahead in the emulator (see trips._searched) on the same held-out
trips as improve.py? An upper bound for what distilling search into a network could reach.

Usage: python -m checks.search_trips_check <run name> [--train N] [--eval N] [--hold N] [--updates N]
"""

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

import trips
from emulator import Emulator
import distance
import jax
from experience import build, choose_bytes, valid_recordings
from explorers import ROM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--eval", type=int, default=600)
    parser.add_argument("--hold", type=int, default=4)
    parser.add_argument("--updates", type=int, default=20000)
    args = parser.parse_args()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    np.random.default_rng(0).shuffle(recordings)
    training, held_out = recordings[:args.train], recordings[args.train:]
    byte_indices = choose_bytes(emulator, training)
    evaluation = trips.from_recordings(held_out, np.random.default_rng(1), args.eval)
    experience = build(emulator, training, byte_indices, on_device=True)
    learned = distance.train(experience, jax.random.key(0), updates=args.updates, log=lambda _: None)
    experience = None
    for player in ["search-learned", "search", "search-true-distance"]:
        outcomes = trips.run([dataclasses.replace(t, player=player) for t in evaluation], learned,
                             byte_indices=byte_indices, search_hold=args.hold)
        print(f"{player:>20}: {trips.summary(outcomes)}", flush=True)


if __name__ == "__main__":
    main()
