"""
Sped-up video of one stage: the stored button path of a chosen visit, replayed from the run's start file.
Usage: python segvideo.py <run> <pick: kraid-room | kraid-dead | richest> <out.mp4> [speed]
"""
import json, os, pickle, subprocess, sys, zlib
from pathlib import Path
import numpy as np
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import MAP_X, MAP_Y, at, energy_of

if __name__ == "__main__":
    run, pick, out = sys.argv[1], sys.argv[2], sys.argv[3]
    speed = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    run_dir = Path("runs") / run
    start = Path(json.loads((run_dir / "config.json").read_text())["start_state"]).read_bytes()
    saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    visits = [(s, p) for s, p in zip(saved["states"], saved["paths"]) if p]
    view = lambda s: emulator.view(s)
    if pick == "kraid-room":
        chosen = min((v for v in visits if (int(view(v[0])[MAP_Y]), int(view(v[0])[MAP_X])) == (29, 8)), key=lambda v: len(v[1]))
    elif pick == "kraid-dead":
        chosen = min((v for v in visits if view(v[0])[at(0x687B)] & 3), key=lambda v: len(v[1]))
    else:
        rich = lambda s: (int(view(s)[263]) >> 4, energy_of(view(s)), int(view(s)[at(0x6879)]))
        chosen = max(visits, key=lambda v: (rich(v[0]), -len(v[1])))
    path = chosen[1]
    emulator.load(start)
    height, width = 224, 240
    encoder = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                "-r", "60", "-i", "-", "-vf", "scale=720:672:flags=neighbor", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                                "-crf", "26", "-r", "60", out], stdin=subprocess.PIPE)
    frame_number = 0
    for action in path:
        for _ in range(FRAMES_PER_STEP):
            if frame_number % speed == 0:
                frame = emulator.frame(ACTIONS[action])
                encoder.stdin.write(frame[:height, :width].tobytes())
            else:
                emulator.run(ACTIONS[action], 1)
            frame_number += 1
    encoder.stdin.close(); encoder.wait()
    m = emulator.memory()
    print(f"{run} {pick}: {len(path)} steps ({len(path) * FRAMES_PER_STEP / 3600:.1f} min of play) -> {out}; ends at map {int(m[MAP_Y])},{int(m[MAP_X])}, energy {energy_of(m)}, Kraid statue {int(m[at(0x687B)]):#x}")
