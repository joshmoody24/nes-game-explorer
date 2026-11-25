"""
Waits until a run's milestones change (items, areas, tanks, missiles, bosses, room count doubling) or the
run finishes, then prints the new state and exits. Standard library only, so plain python3 runs it.

Usage: python3 tools/watch_run.py <run name> [previous milestone JSON]
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

run = sys.argv[1]
previous = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
metrics = Path(__file__).resolve().parent.parent / "runs" / run / "metrics.jsonl"


def milestone(line):
    return {key: line[key] for key in ["items", "areas", "energy tanks", "max missiles", "bosses"]} | {
        "rooms doubled": int(math.log2(max(line["rooms"], 1)))}


while True:
    running = subprocess.run(["pgrep", "-f", f"[g]o_explore.py {run}"], capture_output=True).returncode == 0
    lines = metrics.read_text().splitlines() if metrics.exists() else []
    if lines:
        line = json.loads(lines[-1])
        if not running or milestone(line) != previous:
            print(json.dumps({"status": "RUNNING" if running else "FINISHED", "milestone": milestone(line),
                              "latest": line}))
            break
    time.sleep(10)
