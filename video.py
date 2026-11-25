"""
Videos of play, for watching what a player actually does. Emulation is deterministic, so a run needs only
its start state and the buttons pressed; this replays them with the picture on and encodes an MP4.
"""

import subprocess

from controls import ACTIONS, FRAMES_PER_STEP

SCALE = 3


def record(emulator, start, actions, path, fps=60):
    """Replays actions (button indices, one per step) from the save state start into an MP4 at path."""
    emulator.load(start)
    frames = (emulator.frame(ACTIONS[action]) for action in actions for _ in range(FRAMES_PER_STEP))
    first = next(frames)
    height, width, _ = first.shape
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-vf", f"scale={width * SCALE}:{height * SCALE}:flags=neighbor",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)
    for frame in [first, *frames]:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")
