"""
The whole loop, from power-on, with nothing about Metroid named except hand-made places (and progress
reports, which are only shown to us). Each layer explores from a start state with one branch per detected
spendable, each hoarding only its own byte (see spendables, explorers.Settings.keep_more), side by side.
When a layer's branches finish, its events (lasting acquisitions; see events) each seed a child layer from
their best holder, largest events first. Layer 0 explores from power-on without hoarding, and its
recordings are where spendables are detected.

Usage: python climb.py <name> [--layer-minutes M] [--children N] [--layers N]
"""

import argparse
import json
import os
import pickle
import subprocess
import time
import zlib
from collections import deque
from pathlib import Path

import carry
import events
import spendables
from emulator import Emulator
from experience import valid_recordings
from explorers import ROM

WORKERS = 16


def explore(run_dir, minutes, start_file, keep_more="", workers=WORKERS):
    command = ["python", "go_explore.py", str(run_dir.relative_to("runs")), "--minutes", str(minutes),
               "--workers", str(workers), "--record", "0.05", "--stop-after-stall", "3"]
    if start_file:
        command += ["--start-state", str(start_file)]
    if keep_more:
        command += ["--keep-more", keep_more]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def hoarded_byte(run_dir):
    """The byte a branch hoarded (its first keep_more term), if any."""
    keep_more = json.loads((run_dir / "config.json").read_text()).get("keep_more", "")
    return int(keep_more.split(",")[0].split(":")[0]) if keep_more else None


def summary(run_dir):
    line = json.loads(open(run_dir / "metrics.jsonl").readlines()[-1])
    return {k: line[k] for k in ["frames", "rooms", "areas", "items", "energy tanks", "max missiles", "bosses"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--layer-minutes", type=float, default=12)
    parser.add_argument("--children", type=int, default=4, help="most child layers per layer")
    parser.add_argument("--layers", type=int, default=12, help="most layers in all")
    parser.add_argument("--layer0", help="reuse this run as layer 0 instead of exploring from power-on")
    parser.add_argument("--spendables", type=int, default=2, help="hoarding branches per layer")
    parser.add_argument("--carry", type=int, default=0, help="neediest dead ends each hoarding branch carries riches to")
    args = parser.parse_args()
    root = Path("runs") / args.name
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / "climb.jsonl", "a")
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)

    started = time.perf_counter()
    layer0 = Path("runs") / args.layer0 if args.layer0 else root / "layer0"
    if not args.layer0:
        explore(layer0, args.layer_minutes, None).wait()
    terms = spendables.detect(emulator, valid_recordings(layer0), count=args.spendables)
    print(f"layer0 [{time.perf_counter() - started:.0f}s]: {summary(layer0)}; spendables {terms}", flush=True)
    log.write(json.dumps({"layer": "layer0", "parent": None, "spendables": terms, "summary": summary(layer0)}) + "\n")

    frames_total = summary(layer0)["frames"]
    seen_events = set()
    queue = deque([("layer0", [layer0])])
    count = 1
    while queue and count < args.layers:
        parent, branch_dirs = queue.popleft()
        found = []
        for branch in branch_dirs:
            saved = pickle.loads(zlib.decompress((branch / "checkpoint.pkl.z").read_bytes()))
            for bits, share, holders in events.find(emulator, branch):
                key = frozenset(bits)
                if key not in seen_events:
                    seen_events.add(key)
                    # Best holder: the one reached by the shortest path (its own branch hoards what it hoards).
                    holder = min(holders, key=lambda i: len(saved["paths"][i] or b"x" * 10**9))
                    found.append((len(bits), branch, int(holder), saved["states"][holder]))
        # Riches carried to where they're needed go first: that's the step random play rarely makes.
        carried = []
        for branch in branch_dirs:
            byte = hoarded_byte(branch)
            if byte is not None:
                for target, need, state in carry.deliver(emulator, branch, byte, count=args.carry):
                    print(f"  {branch.name}: carry to {target} (need {need}): {'delivered' if state else 'failed'}", flush=True)
                    if state is not None:
                        carried.append((f"carried to {target}", branch, -1, state))
        events_first = [(f"event of {size} bits", branch, holder, state)
                        for size, branch, holder, state in sorted(found, key=lambda e: -e[0])]
        for why, branch, holder, state in (carried + events_first)[:args.children]:
            if count >= args.layers:
                break
            name = f"layer{count}"
            count += 1
            start_file = root / f"{name}.state"
            start_file.write_bytes(state)
            layer_started = time.perf_counter()
            branches = [root / name / f"hoard-{term.split(':')[0]}" for term in terms]
            processes = [explore(b, args.layer_minutes, start_file, term, WORKERS // len(terms))
                         for b, term in zip(branches, terms)]
            for p in processes:
                p.wait()
            print(f"{name} from {branch} visit {holder} ({why}) "
                  f"[{time.perf_counter() - layer_started:.0f}s]: " +
                  "; ".join(f"{b.name}: {summary(b)}" for b in branches), flush=True)
            frames_total += sum(summary(b)["frames"] for b in branches)
            print(f"  frames emulated so far: {frames_total:,} ({frames_total / 60 / 3600:,.0f} hours of game time)", flush=True)
            log.write(json.dumps({"layer": name, "frames so far": frames_total, "parent": str(branch), "visit": holder,
                                  "why": why, "branches": {b.name: summary(b) for b in branches}}) + "\n")
            log.flush()
            queue.append((name, branches))


if __name__ == "__main__":
    main()
