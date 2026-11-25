"""
Test B: from the start of the game, with no save states, follow the exploration archive's chain of places
to a target (e.g. Morph Ball) under sticky-action noise. The chain gives waypoints (full memory snapshots);
the player aims at each in turn. Reaching a waypoint is judged with hand-made places (evaluation-only
Metroid knowledge). Saves videos of the learned player's best and worst attempts.

With --per-hop, each hop is tried on its own instead, starting from the archive's saved state at the
previous waypoint, to find which hops fail.

Usage: python chase.py <run name> [--target morph|missiles] [--attempts N] [--budget N] [--per-hop]
"""

import argparse
import dataclasses
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

import distance
import policy
import trips
import video
from distill import cached
from emulator import Emulator
from experience import build
from explorers import ROM, Settings, hand_made_place
from metroid import ITEMS, MAX_MISSILES, area_of, at

TARGETS = {"morph": lambda memory: memory[at(ITEMS)] & 0x10, "missiles": lambda memory: memory[at(MAX_MISSILES)] > 0,
           "norfair": lambda memory: area_of(memory) == 1}


def route(emulator, run, target):
    """Saved states along the shortest archive chain from the start to a place with the target."""
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    states, parents = saved["states"], saved["parents"]

    def chain(i):
        path = [i]
        while parents[path[-1]] >= 0:
            path.append(parents[path[-1]])
        return path[::-1]

    hits = [i for i, state in enumerate(states) if TARGETS[target](emulator.memory(state))]
    return [states[i] for i in min((chain(i) for i in hits), key=len)]


def archive_path(emulator, run, target):
    """(start state, buttons) of the shortest stored path to a place with the target."""
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    hits = [i for i, state in enumerate(saved["states"]) if TARGETS[target](emulator.memory(state))]
    return saved["states"][0], min((saved["paths"][i] for i in hits), key=len)


def student(emulator, teacher_file):
    """The best no-lookahead player so far: hindsight on the cached search teacher's trips."""
    params, byte_indices, lessons, outcomes = pickle.loads(Path(teacher_file).read_bytes())
    taught = [(t.start, np.frombuffer(o.actions, np.uint8)) for t, o in zip(lessons, outcomes)]
    import jax
    player = policy.train(build(emulator, taught, byte_indices, on_device=True), jax.random.key(1), updates=30000,
                          log=lambda _: None)
    return player.host_params(), byte_indices, distance.Distance(params, byte_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--target", choices=TARGETS, default="morph")
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--budget", type=int, default=300, help="steps allowed per waypoint")
    parser.add_argument("--teacher", default="cache/teacher-v2-policy-data-20000-20000-20000-4.pkl")
    parser.add_argument("--players", default="random,learned,search-learned,search-true-distance")
    parser.add_argument("--per-hop", action="store_true")
    parser.add_argument("--player", help="a saved player (parameters, byte indices) instead of the distilled student")
    parser.add_argument("--follower", action="store_true", help="play the route with route.Follower (no learning)")
    parser.add_argument("--recovery-rounds", type=int, default=0,
                        help="rounds of practice whose best recoveries the follower remembers")
    parser.add_argument("--recovery-starts", type=int, default=1000)
    parser.add_argument("--route", help="a stitched route (see route.py) to take waypoints from, instead of the archive")
    parser.add_argument("--archive-path", action="store_true",
                        help="take waypoints from the archive's stored shortest path to the target")
    parser.add_argument("--spacing", type=int, default=10, help="steps between waypoints on a stitched route")
    args = parser.parse_args()

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    if args.route or args.archive_path:
        import route as routes
        start, actions = (pickle.loads(Path(args.route).read_bytes()) if args.route
                          else archive_path(emulator, args.run, args.target))
        path = routes.replay(emulator, start, actions)
        states = path[::args.spacing] + ([path[-1]] if (len(path) - 1) % args.spacing else [])
    else:
        states = route(emulator, args.run, args.target)
    start, waypoints = states[0], [emulator.memory(s) for s in states[1:]]
    print(f"route to {args.target}: {len(waypoints)} waypoints", flush=True)
    params, byte_indices, judge = cached(Path("cache") / f"student-{Path(args.teacher).stem}.pkl",
                                         lambda: student(emulator, args.teacher))
    if args.player:
        params, byte_indices = pickle.loads(Path(args.player).read_bytes())
    player = policy.GoalPolicy(params, byte_indices)
    if args.follower:
        import route as routes
        memory = routes.route_memory(emulator, path, actions, byte_indices, args.spacing)
        player = routes.Follower(memory, byte_indices)
        rng = np.random.default_rng(1)
        for round_number in range(args.recovery_rounds):
            *entries, steps_spent = routes.recoveries(emulator, path, player, rng, args.spacing,
                                                      starts=args.recovery_starts)
            memory = routes.with_entries(memory, *entries)
            player = routes.Follower(memory, byte_indices)
            print(f"recovery round {round_number}: +{len(entries[1])} memory entries ({len(memory[1])} total), "
                  f"{steps_spent:,} emulator steps", flush=True)
    videos = Path("videos")
    videos.mkdir(exist_ok=True)
    if args.per_hop:
        for kind in args.players.split(","):
            model = judge if kind == "search-learned" else player
            attempts = [trips.Trip(here, there, args.budget, kind, seed=seed)
                        for here, there in zip(states[:-1], waypoints) for seed in range(args.attempts)]
            reached = np.array([o.reached for o in trips.run(attempts, model, byte_indices=byte_indices, search_hold=4)])
            per_hop = reached.reshape(len(waypoints), args.attempts).mean(1)
            print(f"{kind:>20}: success per hop " + " ".join(f"{rate:.0%}" for rate in per_hop), flush=True)
            print(f"{'':>20}  hops under 50%: {list(np.flatnonzero(per_hop < 0.5))}; chance of chaining every "
                  f"hop {np.prod(per_hop):.2%}", flush=True)
        return
    for kind in args.players.split(","):
        attempts = [trips.Trip(start, waypoints[-1], args.budget, kind, seed=seed, waypoints=tuple(waypoints[:-1]))
                    for seed in range(args.attempts)]
        model = judge if kind == "search-learned" else player
        outcomes = trips.run(attempts, model, byte_indices=byte_indices, search_hold=4)
        reached = np.array([o.targets_reached for o in outcomes])
        print(f"{kind:>20}: reached {args.target} in {np.mean(reached == len(waypoints)):.0%} of {len(outcomes)}; "
              f"waypoints reached median {np.median(reached):.0f}/{len(waypoints)} (best {reached.max()}); "
              f"died in {np.mean([o.died for o in outcomes]):.0%}", flush=True)
        if kind != "random":
            for label, i in [("best", int(reached.argmax())), ("worst", int(reached.argmin()))]:
                path = videos / f"{args.target}-{kind}-{label}.mp4"
                video.record(emulator, start, np.frombuffer(outcomes[i].actions, np.uint8), path)
                print(f"  video of the {label} attempt ({reached[i]}/{len(waypoints)} waypoints): {path}", flush=True)


if __name__ == "__main__":
    main()
