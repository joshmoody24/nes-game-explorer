"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the known places and the score of the state kept for each, so it
can end a segment that has stopped finding anything new and send back only states that are new or
better. The main process keeps an append-only log of (place, score) updates, so keeping a worker in
sync just means sending the updates logged since its last task.
"""

import os
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from empowerment import controllable_bits
from metroid import (CURRENT_MISSILES, ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES,
                     ROOM_X, ROOM_Y, area_of, at)
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
    selection: str = "least-chosen"  # or "productive": also favor places that recently led somewhere new
    # Hand-given knowledge, for diagnosis: count current missiles, not just the maximum, as part of a place.
    current_missiles: bool = False
    # Which state a place keeps: "first" (the first one seen); "empowerment" (the state from which my
    # actions change the most memory bits, see empowerment.py); "rare-empowerment" (the same, with each
    # bit weighted by how rarely it is changeable, so scarce abilities like ammo count most); or, as a
    # hand-given diagnostic, "more-missiles" (replace it whenever a visit arrives with more missiles).
    keep: str = "first"

    def __post_init__(self):
        # Catches typos in flags, which would otherwise silently fall back to the default behavior.
        assert self.keep in ("first", "empowerment", "rare-empowerment", "more-missiles"), f"unknown keep {self.keep!r}"
        assert self.selection in ("least-chosen", "productive"), f"unknown selection {self.selection!r}"


def hand_made_place(memory, settings):
    """Area, room, Samus's rough position in the room, and progress counters."""
    return (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // settings.block_size, int(memory[ROOM_Y]) // settings.block_size,
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5,
            int(memory[at(CURRENT_MISSILES)]) if settings.current_missiles else None)


def score(memory, settings):
    """Higher-scoring states replace lower ones in the same place. Empowerment is scored separately, after a
    segment, because measuring it costs replays."""
    return int(memory[at(CURRENT_MISSILES)]) if settings.keep == "more-missiles" else 0


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
    """What one stretch of random buttons found: {place: (state, score)} for places that were new, or
    reached with a better score than the kept state's when the segment started."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)


def explore(emulator, rng, settings, known, start, start_state, empowerment=None):
    emulator.load(start_state)
    found, progress, gaps = {}, Progress(), []
    action = rng.integers(len(ACTIONS))
    stale = 0
    revisit, revisits = None, 0  # one revisit of a known place, sampled uniformly, to score for empowerment
    for step in range(settings.steps_per_segment):
        if rng.random() > settings.repeat_probability:
            action = rng.integers(len(ACTIONS))
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        state = emulator.save()
        memory = emulator.view(state)
        if memory[GAME_MODE] == GAME_OVER:
            break
        place = hand_made_place(memory, settings)
        new = place not in known and place not in found
        state_score = score(memory, settings)
        if new or state_score > max(known.get(place, -1), found.get(place, (None, -1))[1]):
            found[place] = (state, state_score)
        if place in known:
            revisits += 1
            if rng.random() < 1 / revisits:
                revisit = (place, state)
        if new:
            gaps.append(stale + 1)
            stale = 0
        else:
            stale += 1
            if settings.stale_steps is not None and stale >= settings.stale_steps:
                break
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    if empowerment:
        found = {place: (state, empowerment(emulator, rng, state)) for place, (state, _) in found.items()}
        if revisit:
            place, state = revisit
            state_score = empowerment(emulator, rng, state)
            if state_score > known[place]:
                found[place] = (state, state_score)
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP, gaps)


def worker(connection, settings, seed):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, os.getpid()])
    known = {}
    empowerment = (Empowerment(weighted=settings.keep == "rare-empowerment")
                   if settings.keep in ("empowerment", "rare-empowerment") else None)
    while (task := connection.recv()) is not None:
        start, start_state, updates = task
        known.update(updates)
        connection.send(explore(emulator, rng, settings, known, start, start_state, empowerment))


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

    def send(self, worker_index, start, start_state, updates):
        """updates is the main process's full, append-only log of (place, score) updates."""
        self._connections[worker_index].send((start, start_state, updates[self._updates_sent[worker_index]:]))
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
