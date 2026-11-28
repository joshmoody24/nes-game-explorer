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


def x_is_scrolled(emulator, saves, x_byte):
    """Whether the player's x byte counts on the background (screen x = byte - scroll x) rather than on screen: whichever
    reading has a visible sprite within 8 pixels in more of the saves."""
    hits = {False: 0, True: 0}
    for state in saves:
        sprites = np.frombuffer(emulator.block(state, "SPRT"), np.uint8).reshape(64, 4)
        sprites = sprites[sprites[:, 0] < 0xEF]
        if not len(sprites):
            continue
        value, scroll_x = int(emulator.view(state)[x_byte]), emulator.scroll(state)[0]
        for scrolled in hits:
            target = (value - scroll_x) % 256 if scrolled else value
            hits[scrolled] += int(np.abs(sprites[:, 3].astype(int) - target).min() <= 8)
    return hits[True] > hits[False]


def y_is_scrolled(emulator, saves, x_byte, y_byte, x_scrolled):
    """Like x_is_scrolled for the height byte: the sprite nearest the player's screen x is the player; does its height
    line up better with the raw byte or with byte - scroll y? Games needn't use the same convention on both axes."""
    hits = {False: 0, True: 0}
    for state in saves:
        sprites = np.frombuffer(emulator.block(state, "SPRT"), np.uint8).reshape(64, 4)
        sprites = sprites[sprites[:, 0] < 0xEF]
        if not len(sprites):
            continue
        view, (scroll_x, scroll_y) = emulator.view(state), emulator.scroll(state)
        screen_x = (int(view[x_byte]) - scroll_x) % 256 if x_scrolled else int(view[x_byte])
        player = sprites[np.abs(sprites[:, 3].astype(int) - screen_x).argmin()]
        for scrolled in hits:
            height = (int(view[y_byte]) - scroll_y) % 240 if scrolled else int(view[y_byte])
            hits[scrolled] += int(abs(int(player[0]) - height) <= 16)
    return hits[True] > hits[False]
