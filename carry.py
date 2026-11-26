"""
Carrying riches to where they're needed: in a finished exploration, rank dead ends by need (checks.need_check:
where holding more of a spendable byte reaches places never seen), then take, for each of the top few, the
nearest state holding more of it than any visit there ever did, and travel it there on the map (graph.py),
keeping riches along the way.
"""

import functools
import json
import pickle
import zlib
from pathlib import Path

import graph
from checks.need_check import needs
from experience import valid_recordings


def held(byte, memory):
    return int(memory[byte])


def deliver(emulator, run_dir, byte, count=2):
    """[(target location, need, delivered state or None)] for the count neediest dead ends."""
    run_dir = Path(run_dir)
    _, ranked, need = needs(emulator, run_dir, byte)
    targets = ranked[:count]
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    # Paths replay from the run's start file: the archive's first slot may since hold a richer visit.
    start_file = json.loads((run_dir / "config.json").read_text()).get("start_state")
    start = Path(start_file).read_bytes() if start_file else saved["states"][0]
    world = graph.build([(start, p) for p in saved["paths"] if p] +
                        [(s, bytes(a)) for s, a in valid_recordings(run_dir)])
    views = [emulator.view(s) for s in saved["states"]]
    held_at = {}
    for v in views:
        held_at[graph.location(v)] = max(held_at.get(graph.location(v), -1), int(v[byte]))
    sources = []
    for target in targets:
        richer = {}
        for state, v in zip(saved["states"], views):
            if int(v[byte]) > held_at.get(target, -1):
                richer.setdefault(graph.location(v), state)
        chains = [(len(chain), state) for loc, state in richer.items()
                  for chain in [graph.plan(world, loc, {target})] if chain]
        sources.append(min(chains, key=lambda c: c[0])[1] if chains else None)
    planned = [(t, s) for t, s in zip(targets, sources) if s is not None]
    results = graph.travel_many([s for _, s in planned], world, [{t} for t, _ in planned],
                                keep=functools.partial(held, byte)) if planned else []
    delivered = {t: (state if arrived else None) for (t, _), (arrived, state, *_rest) in zip(planned, results)}
    return [(t, need[t], delivered.get(t)) for t in targets]
