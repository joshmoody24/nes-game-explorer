"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the known places and the path length of the state kept for each, so
it can end a segment that has stopped finding anything new and send back only states that are new or
reached by a shorter path. The main process keeps an append-only log of (place, path length) updates,
so keeping a worker in sync just means sending the updates logged since its last task.
"""

import os
import zlib
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator, button_mask
from metroid import (CURRENT_MISSILES, ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES,
                     ROOM_X, ROOM_Y, area_of, at, energy_of)
from progress import Progress

ROM = "roms/metroid.nes"
PROGRESS_SAMPLE_EVERY = 25


@dataclass(frozen=True)
class Settings:
    steps_per_segment: int = 100
    # End a segment after this many steps in a row without a new place. None never ends early.
    stale_steps: int | None = None
    # Go-Explore's random exploration repeats the previous action most of the time, so it makes
    # sustained moves (a full jump, a run across a room) instead of jittering in place.
    repeat_probability: float = 0.9
    # Each segment lowers its repeat probability by a random amount up to this, so some segments jitter: tapping
    # a button many times in a row (firing several shots) is near impossible when presses are long holds.
    repeat_spread: float = 0.0
    # Hand-given knowledge, for diagnosis: relative chance of button combos with SELECT (Metroid's missile switch)
    # in random play. Below 1, random play wastes missiles less often.
    select_weight: float = 1.0
    # A memory index (memory() numbering) whose higher value makes a place's state better, before path length:
    # keep the state that holds more of something random play only spends (one-way reachability: a state
    # holding more can become one holding less, not back). None ranks by path length alone.
    keep_more: int | None = None
    block_size: int = 32  # pixels per side of the position blocks that split a room into places
    # Hand-given knowledge, for diagnosis: count current missiles, not just the maximum, as part of a place.
    current_missiles: bool = False
    # Hand-given knowledge, for diagnosis: split places by energy band, so healthy and nearly dead versions
    # of the same spot are kept separately.
    health_buckets: bool = False


def hand_made_place(memory, settings):
    """Area, room, Samus's rough position in the room, and progress counters."""
    return (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // settings.block_size, int(memory[ROOM_Y]) // settings.block_size,
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5,
            int(memory[at(CURRENT_MISSILES)]) if settings.current_missiles else None,
            energy_band(energy_of(memory)) if settings.health_buckets else None)


ENERGY_BANDS = (5, 15, 30)  # uneven: the difference between 2 and 12 energy matters more than 52 vs 62


def energy_band(energy):
    return sum(energy >= edge for edge in ENERGY_BANDS)


@dataclass
class Segment:
    """What one stretch of random buttons found: {place: (state, path length from the start, rank)} for places
    that were new, or reached with a better rank (see rank) than the kept state's when the segment started."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)
    actions: bytes  # index into ACTIONS for every step taken, so the segment can be replayed exactly
    start_checksum: int  # zlib.crc32 of the start state; a replay is only valid from a state that matches


WORST_RANK = (-np.inf, -np.inf)


def rank(memory, path_length, settings):
    """How a place's candidate states compare (higher is kept): what they hold (see keep_more), then shorter paths."""
    return (int(memory[settings.keep_more]) if settings.keep_more is not None else 0, -path_length)


def explore(emulator, rng, settings, known, start, start_state, start_path_length):
    """known: {place: rank of the kept state}."""
    emulator.load(start_state)
    found, progress, gaps = {}, Progress(), []
    weights = np.array([settings.select_weight if a & button_mask("SELECT") else 1.0 for a in ACTIONS])
    probabilities = weights / weights.sum()
    action = rng.choice(len(ACTIONS), p=probabilities)
    stale = 0
    pressed = bytearray()
    repeat_probability = settings.repeat_probability - rng.random() * settings.repeat_spread
    for step in range(settings.steps_per_segment):
        if rng.random() > repeat_probability:
            action = rng.choice(len(ACTIONS), p=probabilities)
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        pressed.append(action)
        state = emulator.save()
        memory = emulator.view(state)
        if memory[GAME_MODE] == GAME_OVER:
            break
        place = hand_made_place(memory, settings)
        new = place not in known and place not in found
        length = start_path_length + step + 1
        candidate = rank(memory, length, settings)
        if candidate > max(known.get(place, WORST_RANK), found[place][2] if place in found else WORST_RANK):
            found[place] = (state, length, candidate)
        if new:
            gaps.append(stale + 1)
            stale = 0
        else:
            stale += 1
            if settings.stale_steps is not None and stale >= settings.stale_steps:
                break
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps, bytes(pressed), zlib.crc32(start_state))


def worker(connection, settings, seed):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])
    known = {}
    while (task := connection.recv()) is not None:
        start, start_state, start_path_length, updates = task
        known.update(updates)
        connection.send(explore(emulator, rng, settings, known, start, start_state, start_path_length))


class Explorers:
    def __init__(self, count, settings, seed):
        context = get_context("spawn")
        pipes = [context.Pipe() for _ in range(count)]
        self._connections = [main_end for main_end, _ in pipes]
        self._processes = [context.Process(target=worker, args=(worker_end, settings, seed), daemon=True)
                           for _, worker_end in pipes]
        for process in self._processes:
            process.start()
        self._updates_sent = [0] * count
        self.count = count

    def send(self, worker_index, start, start_state, start_path_length, updates):
        """updates is the main process's full, append-only log of (place, path length) updates."""
        self._connections[worker_index].send((start, start_state, start_path_length,
                                              updates[self._updates_sent[worker_index]:]))
        self._updates_sent[worker_index] = len(updates)

    def finished(self):
        """Blocks until at least one worker is done. Returns [(worker index, Segment)]."""
        ready = wait(self._connections)
        return [(self._connections.index(connection), connection.recv()) for connection in ready]

    def close(self):
        for connection in self._connections:
            connection.send(None)
        for process in self._processes:
            process.join(timeout=5)
