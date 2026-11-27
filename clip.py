"""
Replays a run's shortest stored path to the first save meeting a condition, as a sped-up video.
Usage: python clip.py <run> <condition> <out.mp4> [speed] [last_steps]
"""
import json, os, pickle, subprocess, sys, zlib
from pathlib import Path
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator, TURBO, TURBO_BUTTONS
from explorers import ROM
from metroid import MAP_X, MAP_Y, at

CONDITIONS = {
    "norfair": lambda v: v[0x74] == 0x11,
    "ridley-lair": lambda v: v[0x74] == 0x14,
    "ridley-room": lambda v: v[0x74] == 0x14 and (int(v[MAP_Y]), int(v[MAP_X])) == (29, 17),
    "ridley-dead": lambda v: v[at(0x687C)] & 3,
    "brinstar": lambda v: (int(v[0x74]) or 0x10) == 0x10,
    "ice-beam": lambda v: v[at(0x6878)] & 0x80,
    "tourian": lambda v: v[0x74] == 0x13,
    "missiles-60": lambda v: int(v[at(0x6879)]) >= 60,
    "row-11": lambda v: v[0x74] == 0x13 and int(v[MAP_Y]) == 11,
    "mb-hit": lambda v: int(v[0x99]) >= 1,
    "mb-dead": lambda v: int(v[0x98]) == 6,
    "credits": lambda v: v[0x7B],
    "kraid-shaft-entry": lambda v: (int(v[0x74]) or 0x10) == 0x10 and (int(v[MAP_Y]), int(v[MAP_X])) == (16, 6),
    "out-of-shaft": lambda v: (int(v[0x74]) or 0x10) == 0x10 and int(v[MAP_Y]) == 14 and int(v[MAP_X]) in (5, 7),
}

def frame_mask(action, k):
    """The buttons on frame k of a step, as Emulator.run would press them (turbo lets go of A/B halfway)."""
    if action & TURBO:
        held = action & ~TURBO
        return held if k < FRAMES_PER_STEP // 2 else held & ~TURBO_BUTTONS
    return action


if __name__ == "__main__":
    run, cond, out = sys.argv[1], sys.argv[2], sys.argv[3]
    speed = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    last = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    d = Path("runs") / run
    start = Path(json.loads((d / "config.json").read_text())["start_state"]).read_bytes()
    saved = pickle.loads(zlib.decompress((d / "checkpoint.pkl.z").read_bytes()))
    hits = [p for s, p in zip(saved["states"], saved["paths"]) if p and CONDITIONS[cond](emulator.view(s))]
    path = min(hits, key=len)
    emulator.load(start)
    skip = max(0, len(path) - last) if last else 0
    for a in path[:skip]:
        emulator.run(ACTIONS[a], FRAMES_PER_STEP)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "240x224", "-r", "60", "-i", "-",
                            "-vf", "scale=480:448:flags=neighbor", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "26", out], stdin=subprocess.PIPE)
    f = 0
    for a in path[skip:]:
        for k in range(FRAMES_PER_STEP):
            mask = frame_mask(ACTIONS[a], k)
            if f % speed == 0: enc.stdin.write(emulator.frame(mask)[:224, :240].tobytes())
            else: emulator.run(mask, 1)
            f += 1
    for _ in range(90):
        enc.stdin.write(emulator.frame(0)[:224, :240].tobytes())
    enc.stdin.close(); enc.wait()
    print(f"{run} {cond}: {len(path)} steps ({len(path) * FRAMES_PER_STEP / 3600:.1f} min of play), showing last {len(path) - skip} at {speed}x -> {out}")
