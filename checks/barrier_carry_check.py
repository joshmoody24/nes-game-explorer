"""
The general composition step, end to end on one finished layer: find its barriers (barriers.py), carry its
richest state to each of the top few (graph.travel_many, keeping riches), then explore briefly from every
delivered state and report what each found. Hoards missiles (the detector's top spendable); judging the
outcome uses Metroid knowledge: diagnostic only.

With --need, barriers are ranked by need (checks.need_check: where holding more of the spendable reaches new
places) instead of by tries.

Usage: python -m checks.barrier_carry_check <run name> [barriers] [explore minutes] [--need]
"""

import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path

import barriers
import graph
from emulator import Emulator
from experience import valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, at, energy_of


def missiles_then_energy(memory):
    return int(memory[at(CURRENT_MISSILES)]), energy_of(memory)


def main():
    run, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    minutes = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "2"
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    run_dir = Path("runs") / run
    if "--need" in sys.argv:
        from checks.need_check import needs
        rich_state, ranked, need = needs(emulator, run_dir, at(CURRENT_MISSILES))
        print(f"need of the top 8: {[(loc[:3], need[loc]) for loc in ranked[:8]]}", flush=True)
    else:
        rich_state, ranked = barriers.find(emulator, run_dir, missiles_then_energy)
    print(f"{len(ranked)} barriers; richest state holds {missiles_then_energy(emulator.memory(rich_state))}; "
          f"top {count}: {ranked[:count]}", flush=True)
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    # Paths are relative to the runs start file: the archives first slot may since hold a richer visit.
    start_file = json.loads((run_dir / "config.json").read_text()).get("start_state")
    start = Path(start_file).read_bytes() if start_file else saved["states"][0]
    journeys = [(start, p) for p in saved["paths"] if p] + [(s, bytes(a)) for s, a in valid_recordings(run_dir)]
    world = graph.build(journeys)
    # From the nearest state holding more than any visit to the target ever did (fewest planned hops).
    held_at = {}
    for st in saved["states"]:
        m = emulator.memory(st)
        held_at[graph.location(m)] = max(held_at.get(graph.location(m), -1), missiles_then_energy(m)[0])
    sources = []
    for target in ranked[:count]:
        richer = {}
        for st in saved["states"]:
            m = emulator.memory(st)
            if missiles_then_energy(m)[0] > held_at.get(target, -1):
                richer.setdefault(graph.location(m), st)
        chains = [(len(chain), richer[loc]) for loc in richer
                  for chain in [graph.plan(world, loc, {target})] if chain]
        sources.append(min(chains, key=lambda c: c[0])[1] if chains else rich_state)
        print(f"  barrier {target}: carrying from {missiles_then_energy(emulator.memory(sources[-1]))}, "
              f"{min(chains)[0] - 1 if chains else -1} hops away", flush=True)
    results = graph.travel_many(sources, world, [{target} for target in ranked[:count]],
                                keep=missiles_then_energy)
    out = Path("runs") / f"{run}-carried"
    out.mkdir(exist_ok=True)
    explorations = []
    for k, (target, (arrived, state, steps, hops, failed)) in enumerate(zip(ranked, results)):
        held = missiles_then_energy(emulator.memory(state))
        print(f"  barrier {target}: {'delivered' if arrived else 'not reached'} after {hops} hops, holding {held}", flush=True)
        if arrived:
            (out / f"delivered-{k}.state").write_bytes(state)
            explorations.append(subprocess.Popen(
                ["python", "go_explore.py", f"{run}-carried/explore-{k}", "--minutes", minutes, "--workers", "3",
                 "--start-state", str(out / f"delivered-{k}.state"), "--keep-more", "4217", "--stop-at-item", "bombs"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in explorations:
        p.wait()
    for k in range(count):
        metrics = out / f"explore-{k}" / "metrics.jsonl"
        if metrics.exists():
            print(f"  from barrier {k}: {open(metrics).readlines()[-1][:0]}" +
                  subprocess.run(["python3", "-c", "import json,sys; d=json.loads(open(sys.argv[1]).readlines()[-1]); "
                                  "print(round(d['seconds']), 's', d['rooms'], 'rooms', d['items'])", str(metrics)],
                                 capture_output=True, text=True).stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
