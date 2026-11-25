"""
Do the archive's stored paths replay? For sampled places, replays the path from the run's start state (no
sticky noise) and checks it ends in the same hand-made place as the kept state.

Usage: python -m checks.path_check <run name> [places]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM, Settings, hand_made_place


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / sys.argv[1] / "checkpoint.pkl.z").read_bytes()))
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    start = saved["states"][0]
    picks = np.random.default_rng(0).choice(len(saved["places"]), min(count, len(saved["places"])), replace=False)
    matches, lengths = 0, []
    for i in picks:
        emulator.load(start)
        for action in saved["paths"][i]:
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        matches += hand_made_place(emulator.memory(), Settings()) == saved["places"][i]
        lengths.append(len(saved["paths"][i]))
    print(f"{matches} of {len(picks)} sampled paths replay to their place; path length median "
          f"{np.median(lengths):.0f}, longest {max(lengths)} steps")


if __name__ == "__main__":
    main()
