"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the known places and the path length of the state kept for each, so
it can end a segment that has stopped finding anything new and send back only states that are new or
reached by a shorter path. The main process keeps an append-only log of (place, path length) updates,
so keeping a worker in sync just means sending the updates logged since its last task.
"""

import functools
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
    # Surprise: weight of the selection bonus for visits whose segments produced something never seen before:
    # a rarely-changing memory byte taking a value it never had (a door taking its first missile hit). 0: off.
    surprise: float = 0.0
    # Choose like a hoarder: favor exploring from visits holding more of the hoarded bytes (see keep_more), so
    # lineages that carry resources advance as fast as those that spent them. Weight of the bonus; 0: off.
    favor_holdings: float = 0.0
    # Diagnostic ceiling ("what if resources were free?"): comma-separated "index:value" memory edits applied to
    # every segment's start state, e.g. refilling ammo and health. Empty: off.
    refill: str = ""
    # A saved danger model (see danger.py) and how strongly to avoid exploring from visits it scores as
    # dangerous: weight / (1 + avoid_danger x danger / typical danger). Empty path: off.
    danger_model: str = ""
    avoid_danger: float = 0.0
    # Mastery: favor visits whose explorations find something new only sometimes (the edge of current
    # skill): weight x (1 + mastery x 4p(1-p)), p = share of segments from the visit that found a new place.
    mastery: float = 0.0
    # Learned chooser: weight of a logistic model, refit online, that predicts from a visit's drive signals
    # (see go_explore.Archive.signals) whether exploring from it finds a new place. 0: off (signals and
    # outcomes are still logged to decisions.npz).
    learned_chooser: float = 0.0
    # What the learned chooser predicts: "now" (the segment found a new place) or "later" (the visits it
    # created grew into a top-2% lineage, judged on decisions old enough to have grown).
    chooser_target: str = "now"
    chooser_file: str = ""  # a fixed learned chooser (from checks.downstream_check --save), used from the start
    # Pay back a cheat: a file with the ordered locations of a route found with edited-in resources, and how
    # strongly to favor visits further along it: weight x (1 + guide_weight x progress). Empty path: off.
    guide: str = ""
    guide_weight: float = 0.0
    # Hoarding: comma-separated memory indices (memory() numbering, optionally "index:scale") of things random
    # play spends. Each place keeps every visit that no other visit beats on all of them (see beats), so
    # nothing is traded for anything else; a state holding more can become one holding less, not back. Empty
    # keeps just the shortest path.
    keep_more: str = ""
    block_size: int = 32  # pixels per side of the position blocks that split a room into places
    # Save-scum on damage: bytes of one number, most significant first (e.g. "263,262"); a try ends, like a
    # death, the moment that number drops, so nothing reached after taking damage is kept.
    no_damage: str = ""
    # Extra bytes that split places, "index:bucket,..." (e.g. an enemy's hit points, so damage dealt counts as
    # progress the archive keeps). Diagnostic until such bytes are found automatically.
    place_bytes: str = ""
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
    generic_places: str = ""  # "room", "position" or "both": parts of location from bytes found by generic_places.py
    # Hand-given knowledge, for diagnosis: count current missiles, not just the maximum, as part of a place.
    current_missiles: bool = False
    # Hand-given knowledge, for diagnosis: split places by energy band, so healthy and nearly dead versions
    # of the same spot are kept separately.
    health_buckets: bool = False


@functools.lru_cache(maxsize=1)
def found_place_bytes():
    found = np.load("cache/generic-places.npz")
    return found["scene"], found["position"]


def location(memory, settings):
    """Which room and roughly where in it."""
    found = settings.generic_places
    if found:
        scene, position = found_place_bytes()
    room = ((zlib.crc32(memory.take(scene).tobytes()),) if found in ("room", "both")
            else (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X])))
    where = (tuple(int(memory[b]) // bucket for b, bucket in position) if found in ("position", "both")
             else (int(memory[ROOM_X]) // settings.block_size, int(memory[ROOM_Y]) // settings.block_size))
    return room + where


def extra_place(memory, settings):
    return tuple(int(memory[int(index)]) // int(bucket or 1)
                 for index, _, bucket in (item.partition(":") for item in settings.place_bytes.split(",") if item))


def hand_made_place(memory, settings):
    """Where (see location), any extra place bytes, and progress counters."""
    return (*location(memory, settings), *extra_place(memory, settings),
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
    patterns: frozenset = frozenset()  # (rarely-changing byte, new value) pairs seen, as byte * 256 + value
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


RARE_CHANGE_RATE = 0.002  # bytes changing on fewer steps than this count toward change patterns
WARMUP_STEPS = 2000  # steps a worker watches before it trusts its change rates
_change_counts, _steps_watched = None, 0  # per worker process: how often each memory byte changed
_records = {}  # per worker process: (room, byte) -> (lowest, highest, went down, went up, records), for novel_values
PROGRESS_RECORDS = 3  # one-way records a byte must set in a room before its records count (junk bytes stop sooner)


def refilled(emulator, state, edits):
    """state with the "index:value" memory edits applied."""
    pairs = [tuple(int(x, 0) for x in item.split(":")) for item in edits.split(",") if item]
    edited = bytearray(state)
    for offset, (_, value) in zip(emulator.state_offsets([index for index, _ in pairs]), pairs):
        edited[offset] = value
    return bytes(edited)


def guarded_value(memory, indices):
    return tuple(int(memory[i]) for i in indices)


def explore(emulator, rng, settings, known, start, start_state, start_path_length, last_action=None):
    """known: {place: [(holdings, path length) of every visit the archive kept there]}."""
    global _change_counts, _steps_watched, _records
    patterns, died = set(), False
    if settings.refill:
        start_state = refilled(emulator, start_state, settings.refill)
    emulator.load(start_state)
    guarded = [int(b) for b in settings.no_damage.split(",") if b]
    stay = {tuple(int(v) for v in cell.split(",")) for cell in settings.stay.split(";") if cell}
    rare_values, changed = [], None
    guarded_before = guarded_value(emulator.view(start_state), guarded)
    found, progress, gaps = {}, Progress(), []
    previous_memory = None
    weights = np.array([settings.select_weight if a & button_mask("SELECT") else 1.0 for a in ACTIONS])
    probabilities = weights / weights.sum()
    action = rng.choice(len(ACTIONS), p=probabilities)
    if last_action is not None and rng.random() < settings.momentum:
        action = last_action
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
            if settings.surprise or settings.novel_values:
                changed_before = changed if changed is not None else np.zeros(len(full_memory), bool)
                changed = full_memory != previous_memory
                _change_counts = changed.astype(np.int64) if _change_counts is None else _change_counts + changed
                _steps_watched += 1
                if _steps_watched > WARMUP_STEPS:
                    rare = np.flatnonzero(changed & (_change_counts < RARE_CHANGE_RATE * _steps_watched))
                    patterns.update(int(b) * 256 + int(full_memory[b]) for b in rare)
                    # Only isolated changes: a hit lowers hit points in one jump; moving things change step after step.
                    rare_values = [(int(b), int(full_memory[b])) for b in rare if not changed_before[b]]
        previous_memory = full_memory
        if (memory[GAME_MODE] == GAME_OVER or (guarded and guarded_value(memory, guarded) < guarded_before)
                or (stay and (int(memory[MAP_Y]), int(memory[MAP_X])) not in stay)):
            died = True
            break
        if guarded:
            guarded_before = guarded_value(memory, guarded)
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
                if record and not (went_down and went_up) and count >= PROGRESS_RECORDS:
                    value_place = ("record", room, b, value)
                    if value_place not in known and value_place not in found:
                        found[value_place] = (state, length, held)
                _records[(room, b)] = (min(low, value), max(high, value), went_down, went_up, count)
            rare_values = []
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps, bytes(pressed), zlib.crc32(start_state),
                   frozenset(patterns), died)


def worker(connection, settings, seed):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])
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
        self._processes = [context.Process(target=worker, args=(worker_end, settings, seed), daemon=True)
                           for _, worker_end in pipes]
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
        for connection in self._connections:
            connection.send(None)
        for process in self._processes:
            process.join(timeout=5)
