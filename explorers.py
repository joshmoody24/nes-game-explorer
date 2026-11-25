"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the known places and the score of the state kept for each, so it
can end a segment that has stopped finding anything new and send back only states that are new or
better. The main process keeps an append-only log of (place, score) updates, so keeping a worker in
sync just means sending the updates logged since its last task.
"""

import os
import zlib
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from empowerment import controllable_bits, lost_control_rate
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
    block_size: int = 32  # pixels per side of the position blocks that split a room into places
    # "least-chosen"; "productive": also favor places whose trips found new places or stronger states;
    # "powerful": productive, and also favor places whose kept state scores high under the keep rule.
    selection: str = "least-chosen"
    # Hand-given knowledge, for diagnosis: count current missiles, not just the maximum, as part of a place.
    current_missiles: bool = False
    # Hand-given knowledge, for diagnosis: split places by energy band, so healthy and nearly dead versions
    # of the same spot are kept separately.
    health_buckets: bool = False
    # Which state a place keeps, by score (ties go to the shorter button path from the start): "shortest"
    # (score 0, so purely the shortest path); "empowerment" (the state from which my
    # actions change the most memory bits, see empowerment.py); "rare-empowerment" (the same, with each
    # bit weighted by how rarely it is changeable, so scarce abilities like ammo count most); "safety" (the
    # state from which random play least often ends with my buttons doing nothing, i.e. furthest from
    # death); or, as a hand-given diagnostic, "more-missiles" / "more-health" (whenever a visit arrives with more missiles / energy).
    keep: str = "shortest"

    def __post_init__(self):
        # Catches typos in flags, which would otherwise silently fall back to the default behavior.
        assert self.keep in ("shortest", "empowerment", "rare-empowerment", "safety", "more-missiles", "more-health"), f"unknown keep {self.keep!r}"
        assert self.selection in ("least-chosen", "productive", "powerful"), f"unknown selection {self.selection!r}"


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


def score(memory, settings):
    """Higher-scoring states replace lower ones in the same place. Keep rules measured by replays are scored
    separately, after a segment (see replay_scorer), because replays are expensive."""
    if settings.keep == "more-missiles":
        return int(memory[at(CURRENT_MISSILES)])
    if settings.keep == "more-health":
        return energy_of(memory)
    return 0


EMPOWERMENT_REPLAYS, EMPOWERMENT_STEPS = 4, 10  # cheapest setting that still tells missiles from none ~80% of the time


class Empowerment:
    """
    Scores a state by the memory bits my actions can change from it. If weighted, each bit counts by how
    rarely it has been changeable across the states this worker has scored, so precious abilities
    (firing a missile) outweigh ones available almost everywhere (moving).
    """

    def __init__(self, weighted):
        self.weighted = weighted
        self.changeable_counts = None
        self.scored = 0

    def __call__(self, emulator, rng, state):
        bits = controllable_bits(emulator, state, rng, EMPOWERMENT_REPLAYS, EMPOWERMENT_STEPS)
        if not self.weighted:
            return int(bits.sum())
        if self.changeable_counts is None:
            self.changeable_counts = np.zeros(len(bits))
        self.changeable_counts += bits
        self.scored += 1
        rarity = -np.log((self.changeable_counts + 1) / (self.scored + 2))
        return float((rarity * bits).sum())


@dataclass
class Segment:
    """What one stretch of random buttons found: {place: (state, score, steps into the segment)} for places
    that were new, or reached with a better rank (see rank) than the kept state's when the segment started."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)
    actions: bytes  # index into ACTIONS for every step taken, so the segment can be replayed exactly
    start_checksum: int  # zlib.crc32 of the start state; a replay is only valid from a state that matches


def replay_scorer(settings):
    """For keep rules measured by replays after a segment: a function (emulator, rng, state) -> score, or None."""
    if settings.keep in ("empowerment", "rare-empowerment"):
        return Empowerment(weighted=settings.keep == "rare-empowerment")
    if settings.keep == "safety":
        return lambda emulator, rng, state: 1 - lost_control_rate(emulator, state, rng)
    return None


def rank(score, path_length):
    """How a place's candidate states compare: higher score first, then the shorter path from the start."""
    return score, -path_length


def explore(emulator, rng, settings, known, start, start_state, start_path_length, replay_score=None):
    """known: {place: rank of the kept state}."""
    emulator.load(start_state)
    found, progress, gaps = {}, Progress(), []
    action = rng.integers(len(ACTIONS))
    stale = 0
    revisit, revisits = None, 0  # one revisit of a known place, sampled uniformly, to score for empowerment
    pressed = bytearray()
    for step in range(settings.steps_per_segment):
        if rng.random() > settings.repeat_probability:
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        pressed.append(action)
        state = emulator.save()
        memory = emulator.view(state)
        if memory[GAME_MODE] == GAME_OVER:
            break
        place = hand_made_place(memory, settings)
        new = place not in known and place not in found
        state_score = score(memory, settings)
        candidate = rank(state_score, start_path_length + step + 1)
        best_so_far = max(known.get(place, (-np.inf, 0)), rank(*found[place][1:]) if place in found else (-np.inf, 0))
        if new or candidate > best_so_far:
            found[place] = (state, state_score, start_path_length + step + 1)
        if place in known:
            revisits += 1
            if rng.random() < 1 / revisits:
                revisit = (place, state, start_path_length + step + 1)
        if new:
            gaps.append(stale + 1)
            stale = 0
        else:
            stale += 1
            if settings.stale_steps is not None and stale >= settings.stale_steps:
                break
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    if replay_score:
        found = {place: (state, replay_score(emulator, rng, state), length)
                 for place, (state, _, length) in found.items()}
        if revisit:
            place, state, length = revisit
            state_score = replay_score(emulator, rng, state)
            if rank(state_score, length) > known[place]:
                found[place] = (state, state_score, length)
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps, bytes(pressed), zlib.crc32(start_state))


def worker(connection, settings, seed):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])
    known = {}
    replay_score = replay_scorer(settings)
    while (task := connection.recv()) is not None:
        start, start_state, start_path_length, updates = task
        known.update(updates)
        connection.send(explore(emulator, rng, settings, known, start, start_state, start_path_length, replay_score))


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
        """updates is the main process's full, append-only log of (place, rank) updates."""
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
