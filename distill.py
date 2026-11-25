"""
Distilling search into a player that needs no save states (AlphaZero-style). A learned distance guides a
search teacher that tries every button in the emulator; the teacher's goal-directed trips become lessons
for a no-lookahead player, which is scored on held-out trips without any lookahead. The teacher (distance
and trips) is cached in cache/, so student experiments skip the slow part.

Usage: python distill.py <run name> [--distance-train N] [--distance-updates N] [--trips N]
                         [--policy-updates N] [--eval N] [--hold N]
"""

import argparse
import dataclasses
import os
import pickle
import time
from pathlib import Path

import jax
import numpy as np

import distance
import policy
import trips
from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM


def cached(path, compute):
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = compute()
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def make_teacher(emulator, training, args):
    """(distance class, parameters, byte indices, lesson trips, their outcomes)."""
    started = time.perf_counter()
    experience = build(emulator, training, on_device=True)
    judge = distance.train(experience, jax.random.key(0), updates=args.distance_updates, log=lambda _: None)
    experience = None
    lessons = trips.from_recordings(training, np.random.default_rng(2), args.trips, player="search-learned")
    outcomes = trips.run(lessons, judge, search_hold=args.hold)
    print(f"made the teacher and its {len(lessons)} trips [{time.perf_counter() - started:.0f}s]", flush=True)
    return judge.host_params(), judge.byte_indices, lessons, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--distance-train", type=int, default=20000)
    parser.add_argument("--distance-updates", type=int, default=20000)
    parser.add_argument("--trips", type=int, default=10000, help="teacher trips to learn from")
    parser.add_argument("--policy-updates", type=int, default=10000)
    parser.add_argument("--eval", type=int, default=600)
    parser.add_argument("--hold", type=int, default=4)
    args = parser.parse_args()

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    np.random.default_rng(0).shuffle(recordings)
    training, held_out = recordings[:args.distance_train], recordings[args.distance_train:]
    evaluation = trips.from_recordings(held_out, np.random.default_rng(1), args.eval)
    name = f"teacher-v2-{args.run}-{args.distance_train}-{args.distance_updates}-{args.trips}-{args.hold}.pkl"
    params, byte_indices, lessons, outcomes = cached(Path("cache") / name, lambda: make_teacher(emulator, training, args))
    judge = distance.Distance(params, byte_indices)
    teacher_eval = trips.run([dataclasses.replace(t, player="search-learned") for t in evaluation], judge,
                             search_hold=args.hold)
    print(f"search teacher: {trips.summary(teacher_eval)}; on its own lessons it {trips.summary(outcomes)}", flush=True)

    taught = [(trip.start, np.frombuffer(o.actions, np.uint8)) for trip, o in zip(lessons, outcomes)]
    experience = build(emulator, taught, byte_indices, on_device=True)
    goals = np.stack([trip.goal for trip in lessons])
    button_scores = np.concatenate([o.search_scores for o in outcomes])
    students = {  # name: (training, player that evaluates it)
        "hindsight goals, buttons as pressed": (lambda key: policy.train(
            experience, key, updates=args.policy_updates, log=lambda _: None), "learned"),
        "teacher's goals, scores for every button": (lambda key: policy.train_aimed(
            experience, button_scores, goals, key, updates=args.policy_updates, loss=policy.value_loss,
            log=lambda _: None), "learned-greedy"),
        # Only differences between buttons decide a move; the shared level (how far the goal is) would
        # otherwise dominate the regression.
        "teacher's goals, scores relative to the average button": (lambda key: policy.train_aimed(
            experience, button_scores - button_scores.mean(1, keepdims=True), goals, key,
            updates=args.policy_updates, loss=policy.value_loss, log=lambda _: None), "learned-greedy"),
    }
    for student, (train, player_kind) in students.items():
        player = train(jax.random.key(1))
        outcomes = trips.run([dataclasses.replace(t, player=player_kind) for t in evaluation], player)
        print(f"student ({student}): {trips.summary(outcomes)}", flush=True)


if __name__ == "__main__":
    main()
