"""Selection race report: per arm, rooms / map cells / distinct places / saves / speed at 5 and 10 minutes."""
import json, os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y

if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run_dir in sorted(Path("runs/select").iterdir()):
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
        at5 = min(rows, key=lambda r: abs(r["seconds"] - 300))
        end = rows[-1]
        saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
        cells = {(int(emulator.view(s)[MAP_Y]), int(emulator.view(s)[MAP_X])) for s in saved["states"]}
        places = {tuple(p) for p in saved["places"] if p[0] != "record"}
        print(f"{run_dir.name:34s} rooms {at5['rooms']:3d} -> {end['rooms']:3d} | map cells {len(cells):3d} | distinct places {len(places):6d} "
              f"| saves {len(saved['places']):6d} | {end['frames per second'] // 1000}K fps | items {end['items']} tanks {end['energy tanks']}")
