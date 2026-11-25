"""
The best possible accuracy for "which button was pressed?" given the state before and after, measured
directly: from sample states, press each button for N steps and count distinct resulting observations
(chosen memory bytes). With buttons equally likely, the best any model can do is (distinct outcomes) / 25.

Usage: python -m checks.button_ceiling_check <run name> [states]
"""

import os
import sys
from pathlib import Path

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from experience import choose_bytes, valid_recordings
from explorers import ROM


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    np.random.default_rng(0).shuffle(recordings)
    byte_indices = choose_bytes(emulator, recordings[:2000])
    offsets = emulator.state_offsets(byte_indices)
    rng = np.random.default_rng(1)
    for steps in [1, 2, 4]:
        ceilings, held_ceilings = [], []
        for state, actions in recordings[:count]:
            emulator.load(state)
            for action in actions[:int(rng.integers(len(actions)))]:
                emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            start = emulator.save()
            outcomes = []
            for action in range(len(ACTIONS)):
                emulator.load(start)
                for _ in range(steps):
                    emulator.run(ACTIONS[action], FRAMES_PER_STEP)
                outcomes.append(np.frombuffer(emulator.save(), np.uint8)[offsets].tobytes())
            ceilings.append(len(set(outcomes)) / len(ACTIONS))
        print(f"holding one button for {steps} step(s) ({steps * FRAMES_PER_STEP} frames): distinct outcomes / 25 "
              f"= best possible accuracy {np.mean(ceilings):.0%} (median state {np.median(ceilings):.0%})", flush=True)


if __name__ == "__main__":
    main()
