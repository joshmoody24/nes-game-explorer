"""
Finding the player without knowing the game: the bytes that respond to the controller. From saves reached by random
play, press one button and, separately, its opposite (or nothing) for a short and a longer while. A byte that moves in
the same direction every time, and further the longer the press, is the player's position (or a working copy of it).
Facing flags and the controller byte itself respond too, but don't grow with a longer press.
"""
import numpy as np

from controls import FRAMES_PER_STEP, actions
from emulator import button_mask

SHORT, LONG = 3, 6  # steps of pressing
AXES = {"across": (button_mask("RIGHT"), button_mask("LEFT")), "up and down": (button_mask("A"), 0)}


def _after(emulator, state, buttons, steps):
    emulator.load(state)
    for _ in range(steps):
        emulator.run(buttons, FRAMES_PER_STEP)
    return emulator.memory().astype(np.int16)


def varied_saves(emulator, start_state, rng, count, turbo, steps=300):
    """Saves spread around the start by random play, so the probe isn't fooled by one spot (e.g. a wall)."""
    buttons, saves = actions(turbo), []
    for _ in range(count):
        emulator.load(start_state)
        for _ in range(rng.integers(steps)):
            emulator.run(buttons[rng.integers(len(buttons))], FRAMES_PER_STEP)
        saves.append(emulator.save())
    return saves


def scores(emulator, saves, press, opposite):
    """Per byte: the share of saves where pressing moved it the same way and further with a longer press."""
    agree, grew = 0, 0
    for state in saves:
        short = _after(emulator, state, press, SHORT) - _after(emulator, state, opposite, SHORT)
        long = _after(emulator, state, press, LONG) - _after(emulator, state, opposite, LONG)
        agree, grew = agree + np.sign(long), grew + (np.abs(long) > np.abs(short))
    return np.minimum(np.abs(agree), grew) / len(saves)


def player_bytes(emulator, start_state, rng, turbo, per_axis=2, count=150):
    """memory() indices of the player's position: the per_axis most controller-driven bytes along each axis."""
    saves = varied_saves(emulator, start_state, rng, count, turbo)
    chosen = []
    for press, opposite in AXES.values():
        ranked = [int(b) for b in np.argsort(-scores(emulator, saves, press, opposite)) if int(b) not in chosen]
        chosen += ranked[:per_axis]
    return chosen
