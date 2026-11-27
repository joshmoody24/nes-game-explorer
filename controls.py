"""The buttons an explorer can press, and how long each press is held."""

import os

from emulator import TURBO, TURBO_BUTTONS, button_mask

FRAMES_PER_STEP = 4


def actions(turbo):
    # Every button except START, which pauses on most NES games. Don't prune by guessing what a button does in
    # one game: leaving out SELECT (Metroid's missile switch) once made every red door impossible to open.
    held = [button_mask(*direction, *buttons)
            for direction in [(), ("LEFT",), ("RIGHT",), ("UP",), ("DOWN",)]
            for buttons in [(), ("A",), ("B",), ("A", "B"), ("SELECT",)]]
    # Turbo copies (A/B pressed every step instead of held) go last, so stored paths of older runs keep their meaning.
    return (held + [action | TURBO for action in held if action & TURBO_BUTTONS]) if turbo else held


TURBO_FROM_ENVIRONMENT = bool(os.environ.get("TURBO"))
# For scripts that replay or press buttons outside go_explore.py, which takes Settings.turbo instead.
ACTIONS = actions(TURBO_FROM_ENVIRONMENT)
