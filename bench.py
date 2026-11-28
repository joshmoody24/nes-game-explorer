"""
Wall benchmark: for each wall of the winning route, explore from its start save for one 3-minute run under ONE
shared configuration, and record whether its success test holds, and how far into the archive the first success appears.
Usage: python bench.py <name> <seed> [wall ...]   (config: BENCH_ARGS env, default below; TURBO env as usual)
"""
import json, os, pickle, subprocess, sys, time, zlib
from pathlib import Path
import numpy as np

from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at

MINUTES = float(os.environ.get("BENCH_MINUTES", 3.0))
MAX_FRAMES = int(float(os.environ.get("BENCH_MAX_FRAMES", 0)))  # e.g. 3e6 (about 14 hours of play); 0 = time budget only
DEFAULT_ARGS = "--keep-more 4217,263 --repeat-probability 0.99 --repeat-spread 0.09 --novel-values --workers 20"
brinstar = lambda v: (int(v[0x74]) or 0x10) == 0x10
cell = lambda v: (int(v[MAP_Y]), int(v[MAP_X]))
SHAFT = {(14, 6), (15, 6), (16, 6), (17, 6), (18, 6), (19, 6), (18, 7), (19, 7), (18, 8)}
# Each wall's history is the runs before it in the winning chain (later runs that crossed the wall would mark its
# road as worn out, which a real chain never sees). --familiar keeps only those with the same permanent progress.
HISTORY = {'kraid-fight': 'honest1,honest2,kh-honest', 'kraid-shaft': 'honest3,honest4,honest5', 'column-11': 'honest3,honest4,honest5,honest6', 'lava-corridor': 'honest3,honest4,honest5,honest6,honest8e', 'ridley-door': 'honest8e,lava-layer2c', 'ridley-fight': 'ridley-door3', 'ridley-exit': 'honest9,honest9b', 'ice-beam': 'honest10e,ice2b', 'tourian-entry': 'honest11,honest11b,honest11c', 'missile-door': 'tourian-armed3', 'zebetites': 'tourian-20m3,mb1', 'mother-brain': 'mb2,mb-cross', 'escape': '', 'first-shaft': ''}
WALLS = {
    "kraid-fight":    ("cache/kraid-honest-farmed.state", lambda v: v[at(0x687B)] & 3),
    "kraid-shaft":    ("cache/after-kraid-brinstar.state", lambda v: brinstar(v) and cell(v) not in SHAFT and cell(v)[0] <= 14),
    "column-11":      ("cache/col11-bottom.state", lambda v: brinstar(v) and cell(v)[0] <= 7 and cell(v)[1] >= 12),
    "lava-corridor":  ("cache/lava-edge-farmed.state", lambda v: v[0x74] == 0x14),
    "ridley-door":    ("cache/ridley-door.state", lambda v: v[0x74] == 0x14 and cell(v) == (29, 17) and any(v[0x040F + 16 * k] & 0x40 and v[at(0x6AF4) + 16 * k] for k in range(6))),
    "ridley-fight":   ("cache/ratchet-2.state", lambda v: v[at(0x687C)] & 3),
    "ridley-exit":    ("cache/ridley-exit.state", lambda v: v[0x74] == 0x11),
    "ice-beam":       ("cache/col18-door.state", lambda v: v[at(0x6878)] & 0x80),
    "tourian-entry":  ("cache/col11-top.state", lambda v: v[0x74] == 0x13),
    "missile-door":   ("cache/tourian-20m.state", lambda v: v[0x74] == 0x13 and cell(v)[0] >= 8 and cell(v)[1] == 10),
    "zebetites":      ("cache/mb-rich2.state", lambda v: int(v[0x98]) >= 1),
    "mother-brain":   ("cache/mb-rich3.state", lambda v: int(v[0x98]) == 6),
    "escape":         ("cache/escape-honest.state", lambda v: v[at(0x6883)]),
    # From the game's start: through the first corridor's door into the first shaft (map column 11). Not a wall a
    # hand-place chain picked, so it doesn't favor either kind of place.
    "first-shaft":    ("cache/power-on.state", lambda v: int(v[MAP_X]) == 11),
}

if __name__ == "__main__":
    name, seed = sys.argv[1], sys.argv[2]
    walls = sys.argv[3:] or list(WALLS)
    args = os.environ.get("BENCH_ARGS", DEFAULT_ARGS).split()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    results = {}
    for wall in walls:
        start, test = WALLS[wall]
        run = f"bench/{name}/s{seed}/{wall}"
        familiar = ["--familiar", HISTORY[wall]] if HISTORY[wall] and os.environ.get("BENCH_HISTORY", "1") == "1" else []
        began = time.time()
        subprocess.run(["python", "-u", "go_explore.py", run, "--minutes", str(MINUTES), "--seed", str(seed), "--start-state", start, "--stop-when", wall, "--max-frames", str(MAX_FRAMES)] + args + familiar,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not (Path("runs") / run / "checkpoint.pkl.z").exists():
            print(f"{name} seed {seed} {wall}: crashed (no checkpoint; e.g. killed at the memory cap)", flush=True)
            continue
        states = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))["states"]
        first = next((i for i, s in enumerate(states) if test(emulator.view(s))), None)
        frames = int(np.load(Path("runs") / run / "segments.npz")["frames"].sum())
        results[wall] = None if first is None else {"seconds": round(time.time() - began), "frames": frames, "play hours": round(frames / 60 / 3600, 2)}
        print(f"{name} seed {seed} {wall}: " + ("not broken" if first is None else f"broken in {results[wall]['seconds']}s, {results[wall]['frames'] / 1e6:.1f}M frames = {results[wall]['play hours']} hours of play"), flush=True)
    out = Path("runs/bench") / name / f"s{seed}.json"; out.write_text(json.dumps(results))
    print(name, seed, "broke", sum(1 for v in results.values() if v), "of", len(results), "walls")
