"""
Layers without a human picking start saves: run short chunks; while each chunk finds at least one new room and at least
as many as the one before, resume it; otherwise start a new layer. A new layer starts either from the newest
room's best-held save (earlier layers become history), or with MULTISTART=1 from every save in the rooms the stalled
layer discovered itself. Rooms it inherited are dropped: carrying them spent 91% of tries in an area already done.
Usage: [CONTINUE_RUN=<run> CONTINUE_AFTER=<index>] python autochain.py <name> <start.state> <chunks> [go_explore options...]
"""
import json, os, pickle, subprocess, sys, zlib
from pathlib import Path

from controls import FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM

CHUNK_MINUTES = float(os.environ.get("CHUNK_MINUTES", 3.0))
HOLDINGS = (4217, 263)  # the bytes --keep-more hoards (missiles, energy)
MULTISTART = os.environ.get("MULTISTART") == "1"


def load(run):
    return pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))


def last_metrics(run):
    return json.loads((Path("runs") / run / "metrics.jsonl").read_text().splitlines()[-1])


def room(place):
    return tuple(place[1]) if place[0] == "record" else tuple(place[:3])


def discovery(place):
    """What a place adds to the map of things found: its room, or for a record place the record itself."""
    return tuple(place[:4]) if place[0] == "record" else tuple(place[:3])


def write_subset(run, saved, keep):
    (Path("runs") / run).mkdir(parents=True, exist_ok=True)
    subset = {k: [saved[k][i] for i in keep] for k in ("states", "places", "paths")}
    (Path("runs") / run / "checkpoint.pkl.z").write_bytes(zlib.compress(pickle.dumps(subset)))


def holdings_after_idle(emulator, state):
    """The hoarded bytes after standing still for 20 steps: a save about to lose them is worth less."""
    emulator.load(state)
    for _ in range(20):
        emulator.run(0, FRAMES_PER_STEP)
    memory = emulator.memory()
    return [int(memory[b]) for b in HOLDINGS]


def frontier_start(emulator, saved):
    """The save holding the most after standing still, in the most recently discovered room."""
    states, places = saved["states"], saved["places"]
    first = {}
    for i, place in enumerate(places):
        first.setdefault(room(place), i)
    newest = max(first, key=first.get)
    here = [i for i, p in enumerate(places) if room(p) == newest and p[0] != "record"] or [i for i, p in enumerate(places) if room(p) == newest]
    tops = [max(int(emulator.view(s)[b]) for s in states) or 1 for b in HOLDINGS]
    return max((states[i] for i in here[:200]), key=lambda s: sum(v / t for v, t in zip(holdings_after_idle(emulator, s), tops)))


if __name__ == "__main__":
    name, start, chunks, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    history = [r for r in os.environ.get("FAMILIAR", "").split(",") if r]
    run, resume_after, previous_found, seen, layer = os.environ.get("CONTINUE_RUN"), int(os.environ.get("CONTINUE_AFTER", 0)), None, set(), 0
    inherited = {room(p) for p in load(run)["places"][resume_after:]} if run else set()
    seen = set(inherited)
    Path("cache/auto").mkdir(parents=True, exist_ok=True)
    for chunk in range(chunks):
        new_run = f"{name}/c{chunk:02d}"
        args = ["python", "-u", "go_explore.py", new_run, "--minutes", str(CHUNK_MINUTES), "--start-state", start] + extra
        if run:
            args += ["--resume", run, "--resume-after", str(resume_after)]
        if history:
            args += ["--familiar", ",".join(history)]
        carried = len(load(run)["states"]) - resume_after if run else 0
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        m = last_metrics(new_run)
        saved = load(new_run)
        now = {room(p) for p in saved["places"]}  # rooms only: record places churn by the hundred and made stalls noisy
        found = len(now - seen)
        seen |= now
        stalled = found == 0 or (previous_found is not None and found < previous_found)
        print(f"chunk {chunk}: {new_run} rooms {m['rooms']} areas {m['areas']} items {m['items']} bosses {m.get('bosses')} "
              f"| {found} new discoveries, {m['frames'] / 1e6:.0f}M frames{' | stall' if stalled else ''}", flush=True)
        if stalled and MULTISTART:
            # The stalled run isn't history here: its saves ARE the new layer, and history would mark them worn.
            layer += 1
            discovered = now - inherited
            keep = [i for i, p in enumerate(saved["places"]) if room(p) in discovered] or list(range(len(saved["states"])))
            run = f"{name}/layer{layer}"
            write_subset(run, saved, keep)
            inherited = {room(saved["places"][i]) for i in keep}
            print(f"  new layer {layer} from {len(keep)} saves in the {len(discovered)} rooms the last layer discovered", flush=True)
            resume_after, previous_found, seen = 0, None, set(inherited)
        elif stalled:
            layer += 1
            history.append(new_run)
            start = f"cache/auto/{name.replace('/', '-')}-layer{layer}.state"
            Path(start).write_bytes(frontier_start(emulator, saved))
            print(f"  new layer {layer} from {start}", flush=True)
            run, resume_after, previous_found, seen = None, 0, None, set()
        else:
            run, resume_after, previous_found = new_run, 0, found
    print("DONE", flush=True)
