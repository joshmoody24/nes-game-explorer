"""
Distilling search into a player that needs no save states (AlphaZero-style). A learned distance guides a
search teacher that tries every button in the emulator; the teacher's goal-directed trips become
recordings; the no-lookahead player is trained on them (with hindsight relabeling, like random-play
recordings) and scored on held-out trips without any lookahead.

Usage: python distill.py <run name> [--distance-train N] [--distance-updates N] [--trips N]
                         [--policy-train N] [--policy-updates N] [--eval N] [--hold N]
"""

import argparse
import dataclasses
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--distance-train", type=int, default=60000)
    parser.add_argument("--distance-updates", type=int, default=40000)
    parser.add_argument("--trips", type=int, default=20000, help="teacher trips to learn from")
    parser.add_argument("--policy-train", type=int, default=20000, help="random-play recordings mixed in")
    parser.add_argument("--policy-updates", type=int, default=20000)
    parser.add_argument("--eval", type=int, default=600)
    parser.add_argument("--hold", type=int, default=4)
    args = parser.parse_args()

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    np.random.default_rng(0).shuffle(recordings)
    held_out = recordings[args.distance_train:]
    training = recordings[:args.distance_train]
    byte_indices = choose_bytes(emulator, training[:20000])
    evaluation = trips.from_recordings(held_out, np.random.default_rng(1), args.eval)

    started = time.perf_counter()
    experience = build(emulator, training, byte_indices, on_device=True)
    judge = distance.train(experience, jax.random.key(0), updates=args.distance_updates, log=lambda _: None)
    experience = None
    teacher_eval = trips.run([dataclasses.replace(t, player="search-learned") for t in evaluation], judge,
                             search_hold=args.hold)
    print(f"distance from {len(training)} recordings [{time.perf_counter() - started:.0f}s]; "
          f"search teacher: {trips.summary(teacher_eval)}", flush=True)

    started = time.perf_counter()
    lessons = trips.from_recordings(training, np.random.default_rng(2), args.trips, player="search-learned")
    outcomes = trips.run(lessons, judge, search_hold=args.hold)
    taught = [(trip.start, np.frombuffer(outcome.actions, np.uint8)) for trip, outcome in zip(lessons, outcomes)]
    print(f"teacher made {len(taught)} trips [{time.perf_counter() - started:.0f}s]; they {trips.summary(outcomes)}",
          flush=True)

    for name, data in [("random play only", training[:args.policy_train]),
                       ("random play + teacher trips", training[:args.policy_train] + taught),
                       ("teacher trips only", taught)]:
        experience = build(emulator, data, byte_indices, on_device=True)
        player = policy.train(experience, jax.random.key(1), updates=args.policy_updates, log=lambda _: None)
        experience = None
        print(f"player trained on {name}: {trips.summary(trips.run(evaluation, player))}", flush=True)


if __name__ == "__main__":
    main()
