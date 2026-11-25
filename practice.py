"""
Practice around a clean route (see route.py): start at random points along it, knocked off course by a few
random buttons, and keep the fastest of several attempts to rejoin the route a few waypoints ahead. Those
recoveries are clean lessons on exactly the stretch the player must master, rare moves included. Rounds
alternate practicing and retraining the player on everything practiced so far (with hindsight goals).
Game-agnostic, except that rejoining a waypoint is judged with hand-made places (see trips).

With --compare, one round of practice trains two players on the same lessons, plain and advantage-weighted
(see distance.train_weighted_player).

Usage: python practice.py <run name> [--rounds N] [--starts N] [--tries N] [--player FILE] [--compare]
"""

import argparse
import os
import pickle
from pathlib import Path

import jax
import numpy as np

import distance
import policy
import route as routes
import trips
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from experience import build
from explorers import ROM


def lessons(emulator, path, player, rng, starts, tries, spacing=10, ahead=3, budget=60, most_random_steps=15):
    """
    (recordings, emulator steps spent): recordings (start state, buttons) of the fastest recovery from each
    knocked-off start that got anywhere.
    """
    memories = [emulator.memory(s) for s in path]
    recorded, attempts = [], []
    starting_points = routes.knocked_off(emulator, path, rng, starts, most_random_steps)
    for start, index in starting_points:
        targets = memories[index + spacing::spacing][:ahead] or [memories[-1]]
        attempts.append([trips.Trip(start, targets[-1], budget, "learned", seed=int(rng.integers(2**31)),
                                    waypoints=tuple(targets[:-1])) for _ in range(tries)])
    outcomes = trips.run([t for group in attempts for t in group], player)
    steps_spent = sum(len(o.actions) for o in outcomes) + starts * most_random_steps // 2
    for g, group in enumerate(attempts):
        useful = [o for o in outcomes[g * tries:(g + 1) * tries] if o.targets_reached]
        if useful:
            best = max(useful, key=lambda o: (o.targets_reached, -o.progress_steps[-1]))
            recorded.append((group[0].start, np.frombuffer(best.actions[:best.progress_steps[-1]], np.uint8)))
    return recorded, steps_spent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--target", default="morph")
    parser.add_argument("--teacher", default="cache/teacher-v2-policy-data-20000-20000-20000-4.pkl")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--starts", type=int, default=1000)
    parser.add_argument("--tries", type=int, default=8)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--player", help="a saved player (parameters, byte indices) to practice with")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start, actions = pickle.loads((Path("cache") / f"route-{args.run}-{args.target}.pkl").read_bytes())
    path = routes.replay(emulator, start, actions)
    params, byte_indices, _ = pickle.loads((Path("cache") / f"student-{Path(args.teacher).stem}.pkl").read_bytes())
    if args.player:
        params, byte_indices = pickle.loads(Path(args.player).read_bytes())
    player = policy.GoalPolicy(params, byte_indices)
    judge_params, _, lesson_trips, lesson_outcomes = pickle.loads(Path(args.teacher).read_bytes())
    judge = distance.Distance(judge_params, byte_indices)
    taught = [(t.start, np.frombuffer(o.actions, np.uint8)) for t, o in zip(lesson_trips, lesson_outcomes)]
    rng = np.random.default_rng(0)
    practiced = []
    if args.compare:
        practiced, steps_spent = lessons(emulator, path, player, rng, args.starts, args.tries)
        print(f"{len(practiced)} of {args.starts} starts recovered, {steps_spent:,} emulator steps", flush=True)
        experience = build(emulator, taught + practiced, byte_indices, on_device=True)
        for name, train in [("plain", lambda key: policy.train(experience, key, updates=args.updates, log=lambda _: None)),
                            ("weighted", lambda key: distance.train_weighted_player(experience, judge, key,
                                                                                    updates=args.updates, log=lambda _: None))]:
            out = Path("cache") / f"compare-{args.run}-{args.target}-{name}.pkl"
            out.write_bytes(pickle.dumps((train(jax.random.key(0)).host_params(), byte_indices)))
            print(f"{name} player saved to {out}", flush=True)
        return
    for round_number in range(args.rounds):
        new, steps_spent = lessons(emulator, path, player, rng, args.starts, args.tries)
        practiced += new
        experience = build(emulator, taught + practiced, byte_indices, on_device=True)
        player = policy.train(experience, jax.random.key(round_number), updates=args.updates, log=lambda _: None)
        experience = None
        out = Path("cache") / f"practiced-{args.run}-{args.target}-round{round_number}.pkl"
        out.write_bytes(pickle.dumps((player.host_params(), byte_indices)))
        print(f"round {round_number}: {len(new)} of {args.starts} starts recovered ({steps_spent:,} emulator steps); "
              f"{len(practiced)} lessons so far; "
              f"player saved to {out}", flush=True)


if __name__ == "__main__":
    main()
