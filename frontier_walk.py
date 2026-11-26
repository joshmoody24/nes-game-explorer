"""
Diagnostic: walk a frontier with short explorations. Each round explores from a start for a few minutes, then
restarts from a save in the most recently discovered map cell (optionally only cells whose area byte $74 matches),
since a short exploration started at a frontier covers it far faster than one huge archive. Reports Mother Brain's
status ($98) and hit count ($99) and stops when the hit count moves or the ending starts ($6883).

Usage: python frontier_walk.py <name> <start state> <rounds> <minutes> <refill edits> [area byte, e.g. 0x13]
"""

import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path

from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at, energy_of


def main():
    name, start, rounds, minutes, refill = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    area = int(sys.argv[6], 16) if len(sys.argv) > 6 else None
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    seen = set()
    for r in range(rounds):
        run = f"{name}/round{r}"
        subprocess.run(["python", "-u", "go_explore.py", run, "--minutes", minutes, "--workers", "20", "--start-state",
                        str(start), "--refill", refill, "--repeat-probability", "0.99", "--repeat-spread", "0.09",
                        "--place-bytes", "152:1,153:1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
        views = [(s, emulator.view(s)) for s in saved["states"]]
        if area is not None:
            views = [(s, v) for s, v in views if v[0x74] == area]
        cell = lambda v: (int(v[MAP_Y]), int(v[MAP_X]))
        order = {}
        for k, (_, v) in enumerate(views):
            order.setdefault(cell(v), k)
        new = [c for c in order if c not in seen]
        seen |= set(order)
        hits = max((int(v[0x99]) for _, v in views), default=0)
        ending = any(v[at(0x6883)] for _, v in views)
        print(f"round {r}: {len(order)} cells ({len(new)} new: {sorted(new)}); Mother Brain max hits {hits}; "
              f"status values {sorted({int(v[0x98]) for _, v in views})}; ending {ending}", flush=True)
        if ending or not views:
            break
        if hits:
            best = max(views, key=lambda sv: (int(sv[1][0x99]), energy_of(sv[1])))
        else:
            newest = max(order, key=order.get) if not new else max(new, key=order.get)
            best = max(((s, v) for s, v in views if cell(v) == newest), key=lambda sv: energy_of(sv[1]))
        start = Path("runs") / name / f"start{r + 1}.state"
        start.write_bytes(best[0])


if __name__ == "__main__":
    main()
