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

from controls import FRAMES_PER_STEP, TURBO_FROM_ENVIRONMENT, actions
from emulator import Emulator, button_mask
from metroid import ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, ROOM_Y, area_of, at
from progress import Progress

ROM = "roms/metroid.nes"
PROGRESS_SAMPLE_EVERY = 25


@dataclass(frozen=True)
class Settings:
    steps_per_segment: int = 100
    turbo: bool = TURBO_FROM_ENVIRONMENT  # also offer A/B pressed every step instead of held (see controls.actions)
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
    # Diagnostic focus: map cells "y,x;y,x;..." a try must stay in (leaving ends it like a death). Empty: off.
    stay: str = ""
    # Add a fingerprint of the background tile layout to places, so scenery damage (e.g. a barrier being shot
    # down) counts as progress. Pair with stay: scrolling also changes it.
    place_nametable: bool = False
    # Keep a save whenever a rarely-changing byte sets a new record low or high in this room (e.g. a boss's hit
    # points dropping, a hit counter rising), as its own place: progress that rooms and positions can't see. A byte
    # that has set records both ways in a room is not a progress counter; one counts only from its PROGRESS_RECORDS-th
    # one-way record on.
    novel_values: bool = False
    # Chance a try starts by continuing the last button that led to its start visit (at a frontier, the way
    # you were heading is usually the way on), instead of a random one.
    momentum: float = 0.0


def location(memory, settings):
    """Which room and roughly where in it."""
    return (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // settings.block_size, int(memory[ROOM_Y]) // settings.block_size)


def hand_made_place(memory, settings):
    """Where (see location) and progress counters."""
    return (*location(memory, settings),
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5)


@dataclass
class Segment:
    """What one stretch of random buttons found: {place: (state, path length from the start, holdings)} for
    places that were new, or reached by a visit no kept visit beats (see beats)."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)
    actions: bytes  # index into controls.actions(Settings.turbo) for every step, so it can be replayed exactly
    start_checksum: int  # zlib.crc32 of the start state; a replay is only valid from a state that matches
    died: bool = False  # ended in game over


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


RARE_CHANGE_RATE = 0.002  # bytes changing on fewer steps than this count as rarely changing (see novel_values)
WARMUP_STEPS = 2000  # steps a worker watches before it trusts its change rates
_change_counts, _steps_watched = None, 0  # per worker process: how often each memory byte changed
_records = {}  # per worker process: (room, byte) -> (lowest, highest, went down, went up, records), for novel_values
PROGRESS_RECORDS = 3  # one-way records a byte must set in a room before its records count (junk bytes stop sooner)


def explore(emulator, rng, settings, known, start, start_state, start_path_length, last_action=None):
    """known: {place: [(holdings, path length) of every visit the archive kept there]}."""
    global _change_counts, _steps_watched, _records
    died = False
    emulator.load(start_state)
    stay = {tuple(int(v) for v in cell.split(",")) for cell in settings.stay.split(";") if cell}
    rare_values, changed = [], None
    found, progress, gaps = {}, Progress(), []
    previous_memory = None
    buttons = actions(settings.turbo)
    weights = np.array([settings.select_weight if a & button_mask("SELECT") else 1.0 for a in buttons])
    probabilities = weights / weights.sum()
    action = rng.choice(len(buttons), p=probabilities)
    if last_action is not None and rng.random() < settings.momentum:
        action = last_action
    stale = 0
    pressed = bytearray()
    repeat_probability = settings.repeat_probability - rng.random() * settings.repeat_spread
    for step in range(settings.steps_per_segment):
        if rng.random() > repeat_probability:
            action = rng.choice(len(buttons), p=probabilities)
        emulator.run(buttons[action], FRAMES_PER_STEP)
        pressed.append(action)
        state = emulator.save()
        memory = emulator.view(state)
        full_memory = emulator.memory(state)
        if previous_memory is not None:
            progress.see_change(previous_memory, full_memory)
            if settings.novel_values:
                changed_before = changed if changed is not None else np.zeros(len(full_memory), bool)
                changed = full_memory != previous_memory
                _change_counts = changed.astype(np.int64) if _change_counts is None else _change_counts + changed
                _steps_watched += 1
                if _steps_watched > WARMUP_STEPS:
                    rare = np.flatnonzero(changed & (_change_counts < RARE_CHANGE_RATE * _steps_watched))
                    # Only isolated changes: a hit lowers hit points in one jump; moving things change step after step.
                    rare_values = [(int(b), int(full_memory[b])) for b in rare if not changed_before[b]]
        previous_memory = full_memory
        if memory[GAME_MODE] == GAME_OVER or (stay and (int(memory[MAP_Y]), int(memory[MAP_X])) not in stay):
            died = True
            break
        place = hand_made_place(memory, settings) + ((zlib.crc32(emulator.nametable(state)),) if settings.place_nametable else ())
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
        if settings.novel_values:
            room = (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X]))
            for b, value in rare_values:
                low, high, went_down, went_up, count = _records.get((room, b), (value, value, False, False, 0))
                record = value < low or value > high
                went_down, went_up = went_down or value < low, went_up or value > high
                count += record
                # A save matching the record is offered too, under the same place, so the keep rule (beats) swaps in
                # one that got there holding more: the cheapest way to each level of progress.
                at_record = value <= low if went_down else value >= high
                if at_record and went_down != went_up and count >= PROGRESS_RECORDS:
                    value_place = ("record", room, b, value)
                    rivals = known.get(value_place, []) + ([(found[value_place][2], found[value_place][1])] if value_place in found else [])
                    if not any(beats(rival, (held, length)) for rival in rivals):
                        found[value_place] = (state, length, held)
                _records[(room, b)] = (min(low, value), max(high, value), went_down, went_up, count)
            rare_values = []
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps, bytes(pressed), zlib.crc32(start_state),
                   died)


def worker(connection, settings, seed, index):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, index])  # by worker index, not pid, so a run can be reproduced exactly
    known = {}
    while (task := connection.recv()) is not None:
        start, start_state, start_path_length, last_action, updates = task
        for place, visit in updates:
            known.setdefault(place, []).append(visit)
        connection.send(explore(emulator, rng, settings, known, start, start_state, start_path_length, last_action))


class Explorers:
    def __init__(self, count, settings, seed):
        context = get_context("spawn")
        pipes = [context.Pipe() for _ in range(count)]
        self._connections = [main_end for main_end, _ in pipes]
        self._processes = [context.Process(target=worker, args=(worker_end, settings, seed, index), daemon=True)
                           for index, (_, worker_end) in enumerate(pipes)]
        for process in self._processes:
            process.start()
        self._updates_sent = [0] * count
        self.count = count

    def send(self, worker_index, start, start_state, start_path_length, updates, last_action=None):
        """updates is the main process's full, append-only log of (place, (holdings, path length)) updates."""
        self._connections[worker_index].send((start, start_state, start_path_length, last_action,
                                              updates[self._updates_sent[worker_index]:]))
        self._updates_sent[worker_index] = len(updates)

    def finished(self):
        """Blocks until at least one worker is done. Returns [(worker index, Segment)]."""
        ready = wait(self._connections)
        return [(self._connections.index(connection), connection.recv()) for connection in ready]

    def close(self):
        """Ends the workers at once: their unfinished tries are thrown away, and waiting for each to finish cost
        about a minute per run."""
        for process in self._processes:
            process.terminate()
        for process in self._processes:
            process.join()
