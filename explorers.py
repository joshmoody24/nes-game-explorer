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
    # Hoarding: comma-separated memory indices (memory() numbering, optionally "index:scale") of things random
    # play spends. Each place keeps every visit that no other visit beats on all of them (see beats), so
    # nothing is traded for anything else; a state holding more can become one holding less, not back. Empty
    # keeps just the shortest path.
    keep_more: str = ""
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
    """What one stretch of random buttons found: {place: (state, path length from the start, holdings)} for
    places that were new, or reached by a visit no kept visit beats (see beats)."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)
    actions: bytes  # index into ACTIONS for every step taken, so the segment can be replayed exactly
    start_checksum: int  # zlib.crc32 of the start state; a replay is only valid from a state that matches


def holdings(memory, settings):
    """What a state holds of each hoarded byte (see keep_more), divided by its scale."""
    return tuple(int(memory[int(index)]) / float(scale or 1)
                 for index, _, scale in (item.partition(":") for item in settings.keep_more.split(",") if item))


def beats(a, b):
    """
    Whether visit a = (holdings, path length) makes visit b unnecessary: a holds at least as much of
    everything, and either more of something or the same with a path no longer.
    """
    (held_a, length_a), (held_b, length_b) = a, b
    return all(x >= y for x, y in zip(held_a, held_b)) and (held_a != held_b or length_a <= length_b)


def explore(emulator, rng, settings, known, start, start_state, start_path_length):
    """known: {place: [(holdings, path length) of every visit the archive kept there]}."""
    emulator.load(start_state)
    found, progress, gaps = {}, Progress(), []
    previous_memory = None
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
        full_memory = emulator.memory(state)
        if previous_memory is not None:
            progress.see_change(previous_memory, full_memory)
        previous_memory = full_memory
        if memory[GAME_MODE] == GAME_OVER:
            break
        place = hand_made_place(memory, settings)
        new = place not in known and place not in found
        length = start_path_length + step + 1
        held = holdings(memory, settings)
        rivals = known.get(place, []) + ([(found[place][2], found[place][1])] if place in found else [])
        if not any(beats(rival, (held, length)) for rival in rivals):
            found[place] = (state, length, held)
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
        for place, visit in updates:
            known.setdefault(place, []).append(visit)
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
        """updates is the main process's full, append-only log of (place, (holdings, path length)) updates."""
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
