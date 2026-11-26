"""
How many of an archive's saves in a room are doomed (die within STEPS steps with no buttons pressed, e.g. sinking
in lava)? Doomed saves look like progress to the archive but every try from them fails.

Usage: python -m checks.doomed_check <area,map y,map x> <run> [<run> ...]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import graph
from controls import FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import GAME_MODE, GAME_OVER, energy_of

STEPS = 60


def doomed(emulator, state):
    emulator.load(state)
    for _ in range(STEPS):
        emulator.run(0, FRAMES_PER_STEP)
        if emulator.memory()[GAME_MODE] == GAME_OVER:
            return True
    return False


def main():
    room = tuple(int(v) for v in sys.argv[1].split(","))
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run in sys.argv[2:]:
        saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
        here = [s for s in saved["states"] if graph.location(emulator.view(s))[:3] == room]
        lost = [doomed(emulator, s) for s in here]
        energies = sorted(energy_of(emulator.view(s)) for s in here)
        print(f"{run}: {len(here)} saves in room {room}; doomed {sum(lost)} ({sum(lost) / max(len(here), 1):.0%}); "
              f"energy median {energies[len(energies) // 2] if energies else '-'}", flush=True)


if __name__ == "__main__":
    main()
