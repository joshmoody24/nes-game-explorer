"""
Wall benchmark: for each wall of the winning route, explore from its start save in 1-minute chunks (resuming) under ONE
shared configuration, and record the minute its success test first holds (None = not within the budget).
Usage: python bench.py <name> <seed> [wall ...]   (config: BENCH_ARGS env, default below; TURBO env as usual)
"""
import json, os, pickle, subprocess, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at

CHUNKS = 3
DEFAULT_ARGS = "--keep-more 4217,263 --repeat-probability 0.99 --repeat-spread 0.09 --novel-values --workers 10"
brinstar = lambda v: (int(v[0x74]) or 0x10) == 0x10
cell = lambda v: (int(v[MAP_Y]), int(v[MAP_X]))
SHAFT = {(14, 6), (15, 6), (16, 6), (17, 6), (18, 6), (19, 6), (18, 7), (19, 7), (18, 8)}
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
}

if __name__ == "__main__":
    name, seed = sys.argv[1], sys.argv[2]
    walls = sys.argv[3:] or list(WALLS)
    args = os.environ.get("BENCH_ARGS", DEFAULT_ARGS).split()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    results = {}
    for wall in walls:
        start, test = WALLS[wall]
        previous, solved = None, None
        for minute in range(1, CHUNKS + 1):
            run = f"bench/{name}/s{seed}/{wall}/m{minute}"
            cmd = ["python", "-u", "go_explore.py", run, "--minutes", "1", "--seed", str(seed), "--start-state", start] + args
            if previous: cmd += ["--resume", previous]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            states = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))["states"]
            if any(test(emulator.view(s)) for s in states):
                solved = minute; break
            previous = run
        results[wall] = solved
        print(f"{name} seed {seed} {wall}: {'minute ' + str(solved) if solved else 'not broken in ' + str(CHUNKS) + ' min'}", flush=True)
    out = Path("runs/bench") / name / f"s{seed}.json"; out.write_text(json.dumps(results))
    print(name, seed, "broke", sum(1 for v in results.values() if v), "of", len(results), "walls")
