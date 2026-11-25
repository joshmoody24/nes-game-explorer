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
    return int(controllable_bits(emulator, state, rng, replays, steps, repeat_probability).sum())


def random_play(emulator, state, rng, steps, repeat_probability=0.9):
    """Loads state and presses random buttons, mostly repeating the previous one. Leaves the emulator there."""
    emulator.load(state)
    action = rng.integers(len(ACTIONS))
    for _ in range(steps):
        if rng.random() > repeat_probability:
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)


def controllable_bits(emulator, state, rng, replays=8, steps=30, repeat_probability=0.9):
    """Boolean per memory bit: did it end up different between replays of random buttons from state?"""
    endings = []
    for _ in range(replays):
        random_play(emulator, state, rng, steps, repeat_probability)
        endings.append(np.unpackbits(emulator.memory()))
    endings = np.array(endings)
    return (endings != endings[0]).any(0)


# A state where my buttons change fewer bits than this is treated as "no longer in control" (dead,
# game over). Live states measure in the hundreds.
NO_CONTROL_BITS = 20


def lost_control_rate(emulator, state, rng, replays=3, steps=30, probe_replays=2, probe_steps=6):
    """
    Danger: the share of random-play replays from state that end somewhere my buttons barely affect.
    Measures how close a state is to death without knowing which byte is health.
    """
    lost = 0
    for _ in range(replays):
        random_play(emulator, state, rng, steps)
        ending = emulator.save()
        lost += controllable_bits(emulator, ending, rng, probe_replays, probe_steps).sum() < NO_CONTROL_BITS
    return lost / replays
