"""
Self-improvement for the goal-reaching player (GCSL-style). Round 0 trains on random-play recordings.
Each later round, the player makes goal-directed trips from training states; its trips become new
recordings (hindsight relabeling treats wherever a trip went as its goal), and it retrains on everything.
Every round is scored on the same held-out trips.

Usage: python improve.py <run name> [--rounds R] [--train N] [--trips T] [--updates U] [--eval E]
"""

import argparse
import dataclasses
import os
import time
from pathlib import Path

import jax
import numpy as np

import trips
from emulator import Emulator
from experience import build, choose_bytes, valid_recordings
from explorers import ROM
import critic
import policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--train", type=int, default=20000, help="random-play recordings to start from")
    parser.add_argument("--trips", type=int, default=20000, help="player trips added per round")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--eval", type=int, default=600, help="held-out trips scored each round")
    parser.add_argument("--player", choices=["policy", "critic"], default="policy")
    args = parser.parse_args()

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    rng = np.random.default_rng(0)
    rng.shuffle(recordings)
    training, held_out = recordings[:args.train], recordings[args.train:]
    byte_indices = choose_bytes(emulator, training)
    evaluation = trips.from_recordings(held_out, np.random.default_rng(1), args.eval)
    baselines = {player: trips.run([dataclasses.replace(t, player=player) for t in evaluation])
                 for player in ["random", "replay"]}
    for player, outcomes in baselines.items():
        print(f"{player:>14}: {trips.summary(outcomes)}", flush=True)

    data = list(training)
    for round_number in range(args.rounds + 1):
        started = time.perf_counter()
        experience = None  # free the previous round's GPU copy before building the next
        experience = build(emulator, data, byte_indices, on_device=True)
        model = {"policy": policy, "critic": critic}[args.player]
        player = model.train(experience, jax.random.key(round_number), updates=args.updates, log=lambda _: None)
        trained = time.perf_counter()
        print(f"round {round_number}: {len(data)} recordings ({len(experience.actions):,} steps); "
              f"learned: {trips.summary(trips.run(evaluation, player))}; "
              f"greedy: {trips.summary(trips.run([dataclasses.replace(t, player='learned-greedy') for t in evaluation], player))} "
              f"[build+train {trained - started:.0f}s]", flush=True)
        if round_number == args.rounds:
            break
        started = time.perf_counter()
        attempts = trips.from_recordings(training, np.random.default_rng(100 + round_number), args.trips)
        outcomes = trips.run(attempts, player)
        data += [(trip.start, np.frombuffer(outcome.actions, np.uint8)) for trip, outcome in zip(attempts, outcomes)]
        print(f"  collected {len(attempts)} player trips in {time.perf_counter() - started:.0f}s; "
              f"they {trips.summary(outcomes)}", flush=True)


if __name__ == "__main__":
    main()
