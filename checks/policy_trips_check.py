"""
Test A: can the hindsight goal-reaching player make short trips without save states? Trains on recordings,
reports training/held-out loss (and with goals shuffled, to see how much it uses them), timing, and
trips to goals 10-30 steps ahead on held-out recordings for several players (see trips.py).

Usage: python -m checks.policy_trips_check <run name> [--train N] [--trials N] [--updates N] [--batch N]
"""

import argparse
import dataclasses
import os
import time
from pathlib import Path

import jax
import numpy as np

import trips
from controls import ACTIONS
from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from policy import mean_loss, train

PLAYERS = ["learned", "learned-greedy", "random", "replay"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--train", type=int, default=4000, help="training recordings")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    rng = np.random.default_rng(0)
    rng.shuffle(recordings)
    training, held_out = recordings[:args.train], recordings[args.train:]

    started = time.perf_counter()
    experience = build(emulator, training, on_device=True)
    built = time.perf_counter()
    print(f"{len(recordings)} usable recordings; training on {len(training)} ({len(experience.actions)} steps, "
          f"{len(experience.byte_indices)} varying bytes)")
    policy = train(experience, jax.random.key(0), updates=args.updates, batch=args.batch)
    trained = time.perf_counter()
    held_out_experience = build(emulator, held_out[:500], experience.byte_indices)
    for name, data in [("training", experience), ("held-out", held_out_experience)]:
        print(f"loss on {name} recordings: {mean_loss(policy.params, data, jax.random.key(1)):.3f} "
              f"(3.22 = guessing among {len(ACTIONS)} buttons)")
        print(f"  same, with each goal swapped for a random other one: "
              f"{mean_loss(policy.params, data, jax.random.key(2), shuffle_goals=True):.3f}")
    print(f"timing: building data {built - started:.0f}s ({len(experience.actions) / (built - started):,.0f} steps/s), "
          f"training {trained - built:.0f}s ({args.updates / (trained - built):.0f} updates/s)")

    evaluation = trips.from_recordings(held_out, np.random.default_rng(1), args.trials)
    for player in PLAYERS:
        outcomes = trips.run([dataclasses.replace(t, player=player) for t in evaluation], policy)
        print(f"{player:>15}: {trips.summary(outcomes)}")


if __name__ == "__main__":
    main()
