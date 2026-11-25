"""
Does carrying resources to the frontier break the exploration wall? For each frontier door (see
frontier_check), finds the strongest archive state (most missiles, then energy) standing somewhere on the
door's stored path, carries it the rest of the way with the route follower (best of many attempts per hop,
using save states, as exploration may), then runs random segments from where it arrives and counts
segments that reach anything the archive never saw. Hand-made Metroid knowledge, diagnostic only.

Usage: python -m checks.transport_check <run name> [segments per door]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

import route as routes
from checks.armed_check import _segments, progress_marks
from emulator import Emulator
from experience import choose_bytes, valid_recordings
from explorers import ROM, Settings, hand_made_place
from metroid import CURRENT_MISSILES, MAP_X, MAP_Y, ROOM_X, area_of, at, energy_of
import workers

SPACING = 20


def main():
    run = sys.argv[1]
    per_door = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    states, paths = saved["states"], saved["paths"]
    memories = [emulator.memory(s) for s in states]
    known = set().union(*(progress_marks(m) for m in memories))
    rooms = {(area_of(m), int(m[MAP_Y]), int(m[MAP_X])) for m in memories}
    byte_indices = choose_bytes(emulator, valid_recordings(Path("runs") / run)[:2000])
    strength = lambda m: (int(m[at(CURRENT_MISSILES)]), energy_of(m))
    place_holders = {}
    for i, m in enumerate(memories):
        place = hand_made_place(m, Settings())
        if place not in place_holders or strength(m) > strength(memories[place_holders[place]]):
            place_holders[place] = i
    doors = {}
    for i, m in enumerate(memories):
        side = -1 if m[ROOM_X] < 24 else 1 if m[ROOM_X] > 232 else 0
        if side and (area_of(m), int(m[MAP_Y]), int(m[MAP_X]) + side) not in rooms and paths[i] is not None:
            doors.setdefault((area_of(m), int(m[MAP_Y]), int(m[MAP_X]), side), i)
    total_new, total_segments = 0, 0
    for door, i in sorted(doors.items()):
        path = routes.replay(emulator, states[0], paths[i])
        along = [hand_made_place(emulator.memory(s), Settings()) for s in path]
        # The strongest state anywhere on the path (by place), latest first among equals.
        candidates = [(strength(memories[place_holders[p]]), k, place_holders[p]) for k, p in enumerate(along)
                      if p in place_holders]
        (missiles, energy), k, holder = max(candidates)
        rest = path[k:]
        memory = routes.route_memory(emulator, rest, paths[i][k:], byte_indices, SPACING)
        follower = routes.Follower(memory, byte_indices)
        waypoints = [emulator.memory(rest[w]) for w in routes.waypoint_indices(len(rest), SPACING)]
        actions = routes.stitch(emulator, states[holder], waypoints, follower, attempts=50, budget=200,
                                player="learned-greedy", log=lambda _: None) if len(rest) > 1 else b""
        if actions is None:
            print(f"door {door}: could not carry the state ({missiles} missiles, {energy} energy) "
                  f"the last {len(rest) - 1} steps", flush=True)
            continue
        arrived = routes.replay(emulator, states[holder], actions)[-1]
        arrived_memory = emulator.memory(arrived)
        tasks = [(arrived, [int(i) * 10_000 + s for s in range(per_door)], known)]
        segments = [segment for result in workers.map(_segments, tasks) for segment in result]
        new = [s for s in segments if s]
        total_new, total_segments = total_new + len(new), total_segments + len(segments)
        print(f"door {door}: carried a state with {missiles} missiles / {energy} energy over {len(rest) - 1} steps, "
              f"arrived with {arrived_memory[at(CURRENT_MISSILES)]} / {energy_of(arrived_memory)}; "
              f"{len(new)} of {len(segments)} segments reached something new "
              f"{sorted(set().union(*new))[:4] if new else ''}", flush=True)
    print(f"total: {total_new} of {total_segments} segments reached something new")


if __name__ == "__main__":
    main()
