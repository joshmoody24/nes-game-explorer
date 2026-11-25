"""
One continuous playthrough from power-on to Bombs, stitched from the stored paths of three explorations:
power-on -> first missiles (resources-15min), -> Long Beam (layer-keep-45min, started from that state),
-> Bombs (layer2-keep-30min, started from the Long Beam state). Replays the joined buttons from power-on,
checks it really ends holding Bombs, and records videos (real time and 4x).
"""

import os
import pickle
import subprocess
import zlib
from pathlib import Path

import numpy as np

import video
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import CURRENT_MISSILES, ITEMS, at, energy_of


def archive(run):
    return pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))


def path_to(saved, state):
    return saved["paths"][saved["states"].index(state)]


if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    first = archive("resources-15min")
    layer1 = archive("layer-keep-45min")
    layer2 = archive("layer2-keep-30min")
    missiles_state = Path("cache/start-first-missiles.state").read_bytes()
    long_beam_state = Path("cache/start-long-beam.state").read_bytes()
    bombs = [i for i, s in enumerate(layer2["states"]) if emulator.memory(s)[at(ITEMS)] & 0x01]
    to_bombs = min((layer2["paths"][i] for i in bombs), key=len)
    legs = [("power-on to first missiles", path_to(first, missiles_state)),
            ("to the Long Beam", path_to(layer1, long_beam_state)),
            ("to Bombs", to_bombs)]
    buttons = b"".join(leg for _, leg in legs)
    emulator.load(first["states"][0])
    for action in buttons:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    end = emulator.memory()
    print(" + ".join(f"{name}: {len(leg)} steps" for name, leg in legs)
          + f" = {len(buttons)} steps ({len(buttons) * FRAMES_PER_STEP / 60 / 60:.1f} min of game time)")
    print(f"replayed from power-on, it ends with items {end[at(ITEMS)]:#04x} (bombs bit {'set' if end[at(ITEMS)] & 1 else 'NOT set'}), "
          f"{end[at(CURRENT_MISSILES)]} missiles, {energy_of(end)} energy")
    Path("videos").mkdir(exist_ok=True)
    video.record(emulator, first["states"][0], np.frombuffer(buttons, np.uint8), Path("videos/journey-to-bombs.mp4"))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "videos/journey-to-bombs.mp4", "-vf", "setpts=PTS/4",
                    "-r", "60", "-an", "videos/journey-to-bombs-4x.mp4"], check=True)
    pickle.dump((first["states"][0], buttons), open("cache/journey-to-bombs.pkl", "wb"))
