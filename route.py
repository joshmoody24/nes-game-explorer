"""
Clean routes: a replayable button path from a start state through a chain of waypoints (full memory
snapshots) to a goal, grown hop by hop from the best of many attempts, then shortened by re-growing it with
its own states as waypoints (attempts may skip ahead, so a re-grown path is never longer). Game-agnostic,
except that reaching a waypoint is judged with hand-made places (see trips).
"""

import numpy as np

import trips
from controls import ACTIONS, FRAMES_PER_STEP


class Follower:
    """
    Plays a route without learning: finds the route state most like the current one (most chosen bytes
    equal) and presses the button the route pressed there. Same interface as the trained players
    (model(memories, goal_memories) -> [rows, buttons] scores; goals are ignored).
    """

    def __init__(self, params, byte_indices):
        self.observations, self.actions = params
        self.byte_indices = byte_indices

    def host_params(self):
        return self.observations, self.actions

    def __call__(self, memories, goal_memories):
        similarity = (memories[:, None, self.byte_indices] == self.observations[None]).sum(-1)
        nearest = np.minimum(similarity.argmax(1), len(self.actions) - 1)
        return np.eye(len(ACTIONS))[self.actions[nearest]]


def replay(emulator, start, actions):
    """Every state along a path: [start, after step 1, ..., after the last step]."""
    emulator.load(start)
    states = [start]
    for action in actions:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        states.append(emulator.save())
    return states


def stitch(emulator, start, waypoints, model, attempts=200, budget=300, player="learned", seed=0, log=print):
    """
    Buttons (one per step) from start through waypoints to the last one, or None if attempts stop making
    progress. Each round tries attempts trips from where the path ends and keeps the prefix of the one that
    got furthest along the waypoints, soonest.
    """
    actions, state, done, round_number = bytearray(), start, 0, 0
    while done < len(waypoints):
        tries = [trips.Trip(state, waypoints[-1], budget, player, seed=seed * 1_000_000 + round_number * attempts + i,
                            waypoints=tuple(waypoints[done:-1])) for i in range(attempts)]
        outcomes = [o for o in trips.run(tries, model) if o.targets_reached]
        if not outcomes:
            log(f"  stuck after {done}/{len(waypoints)} waypoints, {len(actions)} steps")
            return None
        best = max(outcomes, key=lambda o: (o.targets_reached, -o.progress_steps[-1]))
        actions += best.actions[:best.progress_steps[-1]]
        state = replay(emulator, state, best.actions[:best.progress_steps[-1]])[-1]
        done += best.targets_reached
        round_number += 1
    return bytes(actions)


def shorten(emulator, start, actions, model, spacings=(60, 40, 20, 10), rounds=3, log=print, **stitch_options):
    """
    Re-grows the path with its own states as waypoints, sparse first: players head for the next waypoint, so
    with dense ones they retrace every detour, while sparse ones let them cut straight across.
    """
    seed = 0
    for spacing in spacings:
        for _ in range(rounds):
            seed += 1
            states = replay(emulator, start, actions)
            waypoints = [emulator.memory(s) for s in states[spacing::spacing]]
            if (len(states) - 1) % spacing:
                waypoints.append(emulator.memory(states[-1]))
            shorter = stitch(emulator, start, waypoints, model, seed=seed, log=log, **stitch_options)
            if shorter is None or len(shorter) >= len(actions):
                break
            log(f"  spacing {spacing}: shortened {len(actions)} -> {len(shorter)} steps")
            actions = shorter
    return actions


def main():
    import argparse
    import os
    import pickle
    from pathlib import Path

    import policy
    import video
    from chase import TARGETS, route, student
    from distill import cached
    from emulator import Emulator
    from explorers import ROM

    parser = argparse.ArgumentParser(description="Stitch and shorten a route from the archive's start to a target.")
    parser.add_argument("run")
    parser.add_argument("--target", choices=TARGETS, default="morph")
    parser.add_argument("--teacher", default="cache/teacher-v2-policy-data-20000-20000-20000-4.pkl")
    parser.add_argument("--player", help="a saved player (parameters, byte indices) instead of the distilled student")
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--spacings", default="60,40,20,10")
    parser.add_argument("--resume", action="store_true", help="shorten the saved route instead of stitching anew")
    parser.add_argument("--attempt-player", default="learned", help="who makes the attempts: learned or random")
    parser.add_argument("--name", default="", help="suffix for the saved route and video")
    args = parser.parse_args()
    saved_route = Path("cache") / f"route-{args.run}-{args.target}{args.name}.pkl"

    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    states = route(emulator, args.run, args.target)
    params, byte_indices, _ = cached(Path("cache") / f"student-{Path(args.teacher).stem}.pkl",
                                     lambda: student(emulator, args.teacher))
    if args.player:
        params, byte_indices = pickle.loads(Path(args.player).read_bytes())
    player = policy.GoalPolicy(params, byte_indices)
    if args.resume:
        start, actions = pickle.loads(saved_route.read_bytes())
        print(f"shortening the saved {len(actions)}-step route", flush=True)
    else:
        start = states[0]
        actions = stitch(emulator, start, [emulator.memory(s) for s in states[1:]], player, attempts=args.attempts,
                         player=args.attempt_player)
        if actions is None:
            return
        print(f"stitched a path of {len(actions)} steps through {len(states) - 1} archive waypoints", flush=True)
    actions = shorten(emulator, start, actions, player, spacings=[int(s) for s in args.spacings.split(",")],
                      attempts=args.attempts, player=args.attempt_player)
    end = emulator.memory(replay(emulator, start, actions)[-1])
    print(f"final path: {len(actions)} steps; replayed, it ends with the target: {bool(TARGETS[args.target](end))}",
          flush=True)
    Path("cache").mkdir(exist_ok=True)
    saved_route.write_bytes(pickle.dumps((start, actions)))
    Path("videos").mkdir(exist_ok=True)
    video.record(emulator, start, actions, Path("videos") / f"route-{args.target}{args.name}.mp4")


if __name__ == "__main__":
    main()
