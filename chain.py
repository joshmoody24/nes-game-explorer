"""
The beaten-game reference, a diagnostic that may cheat and use Metroid knowledge: explore from a start until any
new acquisition (item, energy tank, missile capacity, boss), restart from the save holding it (shortest path),
repeat. Every try starts with energy and missiles refilled (explorers.Settings.refill). Where it stalls is a wall;
its route gives near starts for fast probes anywhere in the game.

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

from emulator import Emulator
from explorers import ROM
from near_starts import gains

REFILL = "4217:5,262:0x99,263:0x03"
FLAGS = ["--workers", "18", "--keep-more", "4217", "--refill", REFILL, "--repeat-probability", "0.99",
         "--repeat-spread", "0.09", "--stop-at-gain", "--record", "0.05"]


def main():
    name, start_file = sys.argv[1], Path(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    minutes = sys.argv[4] if len(sys.argv) > 4 else "12"
    root = Path("runs") / name
    root.mkdir(parents=True, exist_ok=True)
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    log = open(root / "chain.jsonl", "a")
    total_seconds = 0.0
    for step in range(steps):
        run = f"{name}/step{step}"
        started = time.perf_counter()
        subprocess.run(["python", "-u", "go_explore.py", run, "--minutes", minutes, "--start-state", str(start_file)]
                       + FLAGS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seconds = time.perf_counter() - started
        total_seconds += seconds
        saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
        start_view = emulator.view(start_file.read_bytes())
        gained = [(len(p), i, gains(emulator.view(s), start_view))
                  for i, (s, p) in enumerate(zip(saved["states"], saved["paths"])) if p and gains(emulator.view(s), start_view)]
        metrics = json.loads((Path("runs") / run / "metrics.jsonl").read_text().splitlines()[-1])
        if not gained:
            print(f"step {step}: no acquisition in {seconds:.0f} s ({metrics['rooms']} rooms): wall", flush=True)
            log.write(json.dumps({"step": step, "seconds": seconds, "gain": None, "rooms": metrics["rooms"]}) + "\n")
            break
        length, index, what = min(gained)
        start_file = root / f"after-step{step}.state"
        start_file.write_bytes(saved["states"][index])
        print(f"step {step}: {what} after {seconds:.0f} s (path {length} steps; total {total_seconds / 60:.0f} min); "
              f"now holding {metrics['items']}, tanks {metrics['energy tanks']}, max missiles {metrics['max missiles']}, "
              f"bosses {metrics['bosses']}", flush=True)
        log.write(json.dumps({"step": step, "seconds": seconds, "gain": what, "path": length,
                              "state": str(start_file)}) + "\n")
        log.flush()


if __name__ == "__main__":
    main()
