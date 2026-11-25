"""
Empowerment, cheaply: how much a state lets my actions change the world.

The emulator is deterministic, so replaying different random buttons from the same saved state
produces RAM differences caused by nothing but those buttons. Things that change on their own (frame
counters, timers, enemies on fixed paths) come out the same in every replay and don't count.
"""

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP


def action_diversity(emulator, state, rng, replays=8, steps=30, repeat_probability=0.9):
    """Number of memory bits that end up different between replays of random buttons from state."""
    endings = []
    for _ in range(replays):
        emulator.load(state)
        action = rng.integers(len(ACTIONS))
        for _ in range(steps):
            if rng.random() > repeat_probability:
                action = rng.integers(len(ACTIONS))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        endings.append(np.unpackbits(emulator.memory()))
    endings = np.array(endings)
    return int((endings != endings[0]).any(0).sum())
