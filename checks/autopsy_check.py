"""
Why did a run fail to reach something a known route reaches? Replays the known route from the same start and
asks of the failed run's archive: how far along the route did it get (furthest route location it ever held a
visit at), how often it chose visits there compared with everywhere else, where its choices went instead, and
what happened on tries from the furthest point (deaths, places found). Writes the furthest visit's state so it
can be watched.

Usage: python -m checks.autopsy_check <failed run> <route run> <route visit gain e.g. tank> <steps before>
"""

import json
import os
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

import graph
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import energy_of
from near_starts import gains


def main():
    failed, route_run, gain, before = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    route_dir = Path("runs") / route_run
    route_start = Path(json.loads((route_dir / "config.json").read_text())["start_state"]).read_bytes()
    route_saved = pickle.loads(zlib.decompress((route_dir / "checkpoint.pkl.z").read_bytes()))
    start_view = emulator.view(route_start)
    candidates = [i for i, (s, p) in enumerate(zip(route_saved["states"], route_saved["paths"]))
                  if p and gain in gains(emulator.view(s), start_view)]
    path = route_saved["paths"][min(candidates, key=lambda i: len(route_saved["paths"][i]))]
    emulator.load(route_start)
    for a in path[:-before]:
        emulator.run(ACTIONS[a], FRAMES_PER_STEP)
    route = []
    for a in path[-before:]:
        emulator.run(ACTIONS[a], FRAMES_PER_STEP)
        here = graph.location(emulator.memory())
        if not route or route[-1] != here:
            route.append(here)
    print(f"known route: {before} steps, {len(route)} locations, rooms {list(dict.fromkeys(l[:3] for l in route))}")

    run_dir = Path("runs") / failed
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    views = [emulator.view(s) for s in saved["states"]]
    locations = [graph.location(v) for v in views]
    chosen = np.array(saved["times_chosen"])
    reached = {loc: k for k, loc in enumerate(route)}
    on_route = [(reached[loc], i) for i, loc in enumerate(locations) if loc in reached]
    furthest = max(k for k, _ in on_route)
    print(f"{failed}: {len(views)} visits, {int(chosen.sum()):,} choices; furthest along the route: location "
          f"{furthest + 1}/{len(route)} {route[furthest]}; next on the route: "
          f"{route[furthest + 1] if furthest + 1 < len(route) else 'the gain'}")
    for k in range(max(0, furthest - 3), furthest + 1):
        here = [i for kk, i in on_route if kk == k]
        if here:
            print(f"  route location {k + 1} {route[k]}: {len(here)} visits, chosen {int(chosen[here].sum())} times "
                  f"({chosen[here].sum() / chosen.sum():.2%} of all choices), energy {[energy_of(views[i]) for i in here[:5]]}")
    rooms = Counter()
    for loc, c in zip(locations, chosen):
        rooms[loc[:3]] += int(c)
    print(f"  choices by room (top 8): {[(room, f'{n / chosen.sum():.0%}') for room, n in rooms.most_common(8)]}")
    segments = np.load(run_dir / "segments.npz")
    tip = {i for kk, i in on_route if kk >= furthest - 1}
    from_tip = np.isin(segments["start"], list(tip))
    print(f"  tries from the furthest two route locations: {int(from_tip.sum())}; of them found something new "
          f"{int((segments['new_places'][from_tip] > 0).sum())}")
    best = max((i for kk, i in on_route if kk == furthest), key=lambda i: -chosen[i])
    out = Path(f"cache/autopsy-{failed.replace('/', '-')}.state")
    out.write_bytes(saved["states"][best])
    print(f"  furthest visit -> {out}")


if __name__ == "__main__":
    main()
