"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the set of known places, so it can end a segment that has stopped
finding anything new and send back only places that are new. The main process's list of places only
ever grows, so keeping a worker in sync just means sending the places added since its last task.
"""

import os
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from emulator import Emulator, button_mask
from metroid import (AREA, ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, ROOM_Y,
                     at)
from progress import Progress

ROM = "roms/metroid.nes"
FRAMES_PER_STEP = 4
PROGRESS_SAMPLE_EVERY = 25
# START pauses the game and SELECT switches weapons; both are left out.
ACTIONS = [button_mask(*direction, *buttons)
           for direction in [(), ("LEFT",), ("RIGHT",), ("UP",), ("DOWN",)]
           for buttons in [(), ("A",), ("B",), ("A", "B")]]


@dataclass(frozen=True)
class Settings:
    steps_per_segment: int = 100
    # End a segment after this many steps in a row without a new place. None never ends early.
    stale_steps: int | None = None
    # Go-Explore's random exploration repeats the previous action most of the time, so it makes
    # sustained moves (a full jump, a run across a room) instead of jittering in place.
    repeat_probability: float = 0.9
    block_size: int = 32  # pixels per side of the position blocks that split a room into places
    selection: str = "least chosen"  # or "productive": also favor places that recently led somewhere new


def hand_made_place(memory, block_size):
    """Area, room, Samus's rough position in the room, and progress counters."""
    return (int(memory[AREA]), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // block_size, int(memory[ROOM_Y]) // block_size,
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5)


@dataclass
class Segment:
    """What one stretch of random buttons found: places not known when it started, with a state at each."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)


def explore(emulator, rng, settings, known, start, start_state):
    emulator.load(start_state)
    found, progress, gaps = {}, Progress(), []
    action = rng.integers(len(ACTIONS))
    stale = 0
    for step in range(settings.steps_per_segment):
        if rng.random() > settings.repeat_probability:
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        state = emulator.save()
        memory = emulator.view(state)
        if memory[GAME_MODE] == GAME_OVER:
            break
        place = hand_made_place(memory, settings.block_size)
        if place in known or place in found:
            stale += 1
            if settings.stale_steps is not None and stale >= settings.stale_steps:
                break
        else:
            found[place] = state
            gaps.append(stale + 1)
            stale = 0
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps)


def worker(connection, settings, seed):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])
    known = set()
    while (task := connection.recv()) is not None:
        start, start_state, new_places = task
        known.update(new_places)
        connection.send(explore(emulator, rng, settings, known, start, start_state))


class Explorers:
    def __init__(self, count, settings, seed):
        context = get_context("spawn")
        pipes = [context.Pipe() for _ in range(count)]
        self._connections = [main_end for main_end, _ in pipes]
        self._processes = [context.Process(target=worker, args=(worker_end, settings, seed), daemon=True)
                           for _, worker_end in pipes]
        for process in self._processes:
            process.start()
        self._places_sent = [0] * count
        self.count = count

    def send(self, worker_index, start, start_state, places):
        """places is the main process's full, append-only list of known places."""
        self._connections[worker_index].send((start, start_state, places[self._places_sent[worker_index]:]))
        self._places_sent[worker_index] = len(places)

    def finished(self):
        """Blocks until at least one worker is done. Returns [(worker index, Segment)]."""
        ready = wait(self._connections)
        return [(self._connections.index(connection), connection.recv()) for connection in ready]

    def close(self):
        for connection in self._connections:
            connection.send(None)
        for process in self._processes:
            process.join(timeout=5)
