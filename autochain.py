"""
Layers without a human picking start saves: run short chunks; while a chunk finds a new room, resume it; when a chunk
stalls, start a fresh layer from the healthiest, best-holding save among the places the last chunk discovered.
Earlier layers that began with the same permanent progress count as familiar history.
Usage: python autochain.py <name> <start.state> <chunks> [go_explore options...]
"""
import json, os, pickle, subprocess, sys, zlib
from pathlib import Path

from controls import FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y

CHUNK_MINUTES = 3.0
HOLDINGS = (4217, 263)  # the bytes --keep-more hoards (missiles, energy)


def last_metrics(run):
    return json.loads((Path("runs") / run / "metrics.jsonl").read_text().splitlines()[-1])


def keeps_holdings(emulator, state):
    """Standing still for 20 steps loses none of the hoarded bytes."""
    emulator.load(state)
    before = [int(emulator.memory()[b]) for b in HOLDINGS]
    for _ in range(20):
        emulator.run(0, FRAMES_PER_STEP)
    return all(int(emulator.memory()[b]) >= v for b, v in zip(HOLDINGS, before))


def frontier_start(emulator, run):
    """The healthy save with the best holdings in the most recently discovered room that has one."""
    states = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))["states"]
    room = lambda s: (int(emulator.view(s)[0x74]) or 0x10, int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X]))
    first, members = {}, {}
    for i, s in enumerate(states):
        first.setdefault(room(s), i)
        members.setdefault(room(s), []).append(s)
    tops = [max(int(emulator.view(s)[b]) for s in states) or 1 for b in HOLDINGS]
    score = lambda s: sum(int(emulator.view(s)[b]) / t for b, t in zip(HOLDINGS, tops))
    for r in sorted(first, key=lambda r: -first[r]):
        for s in sorted(members[r], key=lambda s: -score(s))[:50]:
            if keeps_holdings(emulator, s):
                return s
    return states[-1]


if __name__ == "__main__":
    name, start, chunks, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    layers, layer, run, rooms = [], 0, None, 0
    Path("cache/auto").mkdir(parents=True, exist_ok=True)
    for chunk in range(chunks):
        new_run = f"{name}/c{chunk:02d}"
        args = ["python", "-u", "go_explore.py", new_run, "--minutes", str(CHUNK_MINUTES), "--workers", "20"] + extra
        if run is None:
            args += ["--start-state", start]
        else:
            args += ["--resume", run, "--start-state", start]
        familiar = [r for r in layers if r != run] + [r for r in os.environ.get("FAMILIAR", "").split(",") if r]
        if familiar:
            args += ["--familiar", ",".join(familiar)]
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        m = last_metrics(new_run)
        grew = m["rooms"] > rooms
        print(f"chunk {chunk}: {new_run} rooms {m['rooms']} areas {m['areas']} items {m['items']} | {'progress' if grew else 'stall'}", flush=True)
        rooms = max(rooms, m["rooms"])
        if grew:
            run = new_run
        else:
            state = frontier_start(emulator, new_run)
            layers.append(new_run)
            layer += 1
            start = f"cache/auto/{name.replace('/', '-')}-layer{layer}.state"
            Path(start).write_bytes(state)
            print(f"  new layer {layer} from {start}", flush=True)
            run, rooms = None, 0
