"""
The beaten-game reference, a diagnostic that may cheat and use Metroid knowledge: explore from a start until any
new acquisition (item, energy tank, missile capacity, boss), continue from a save holding it, repeat. Every try
starts with energy and missiles refilled (explorers.Settings.refill). A save can hold the acquisition yet be a dead
end (a glitched route left Samus in a broken scroll mode), so when a step stalls the chain backtracks to the next
holder of the previous acquisition. Where every holder stalls is a wall; the route gives near starts for fast probes.

Usage: python chain.py <name> <start state> [steps] [minutes per step]
"""

import json
import os
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path

import graph
from emulator import Emulator
from explorers import ROM
from near_starts import gains

REFILL = "4217:5,262:0x99,263:0x03"
FLAGS = ["--workers", "18", "--keep-more", "4217", "--refill", REFILL, "--repeat-probability", "0.99",
         "--repeat-spread", "0.09", "--stop-at-gain", "--stop-after-stall", "2", "--record", "0.05"]
HOLDERS = 6  # saves per acquisition to fall back on, at distinct places, shortest paths first


def holders(emulator, run, start_file):
    """[(acquisition, state)] for the HOLDERS shortest-path saves at distinct places that gained over the start."""
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    start_view = emulator.view(start_file.read_bytes())
    gained = sorted((len(p), i) for i, (s, p) in enumerate(zip(saved["states"], saved["paths"]))
                    if p and gains(emulator.view(s), start_view))
    found, places = [], set()
    for _, i in gained:
        place = graph.location(emulator.view(saved["states"][i]))
        if place not in places:
            places.add(place)
            found.append((gains(emulator.view(saved["states"][i]), start_view), saved["states"][i]))
    return found[:HOLDERS]


def main():
    name, start_file = sys.argv[1], Path(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    minutes = sys.argv[4] if len(sys.argv) > 4 else "12"
    root = Path("runs") / name
    root.mkdir(parents=True, exist_ok=True)
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    log = open(root / "chain.jsonl", "a")
    started = time.perf_counter()
    fallbacks = []  # (acquisition, state) holders of the current start's acquisition not yet tried
    tries = 0
    step = 0
    while step < steps:
        run = f"{name}/step{step}-try{tries}"
        subprocess.run(["python", "-u", "go_explore.py", run, "--minutes", minutes, "--start-state", str(start_file)]
                       + FLAGS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        metrics = json.loads((Path("runs") / run / "metrics.jsonl").read_text().splitlines()[-1])
        found = holders(emulator, run, start_file)
        elapsed = (time.perf_counter() - started) / 60
        if not found:
            print(f"step {step} try {tries}: stalled at {metrics['seconds']:.0f} s, {metrics['rooms']} rooms "
                  f"[{elapsed:.0f} min total]", flush=True)
            log.write(json.dumps({"step": step, "try": tries, "gain": None, "rooms": metrics["rooms"]}) + "\n")
            if not fallbacks:
                print(f"step {step}: wall (no holder left to back up to)", flush=True)
                break
            what, state = fallbacks.pop(0)
            start_file = root / f"step{step}-try{tries + 1}.state"
            start_file.write_bytes(state)
            tries += 1
            continue
        (what, state), fallbacks = found[0], found[1:]
        start_file = root / f"after-step{step}.state"
        start_file.write_bytes(state)
        print(f"step {step} try {tries}: {what} at {metrics['seconds']:.0f} s [{elapsed:.0f} min total]; holding "
              f"{metrics['items']}, tanks {metrics['energy tanks']}, max missiles {metrics['max missiles']}, "
              f"bosses {metrics['bosses']}; {len(fallbacks)} fallbacks", flush=True)
        log.write(json.dumps({"step": step, "try": tries, "gain": what, "state": str(start_file)}) + "\n")
        log.flush()
        step, tries = step + 1, 0


if __name__ == "__main__":
    main()
