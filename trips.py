"""
Goal-directed trips run in parallel worker processes, each with its own emulator and a CPU copy of the
player. A trip starts from a save state, aims at a goal (a full memory snapshot), and lasts a budget of
steps under sticky-action noise. Each trip returns the buttons pressed, so trips double as training data.
Success is judged with hand-made places, which is evaluation-only Metroid knowledge.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM, Settings, hand_made_place

STICKY = 0.25  # share of steps that repeat the previous button instead of the chosen one


@dataclass(frozen=True)
class Trip:
    start: bytes  # save state
    goal: np.ndarray  # full memory snapshot to aim for
    budget: int  # steps allowed
    player: str  # "learned", "learned-greedy", "random", or "replay"
    replay: tuple = ()  # buttons to press, for the "replay" player
    seed: int = 0


@dataclass(frozen=True)
class Outcome:
    reached: bool
    start_distance: float
    closest_distance: float
    actions: bytes  # every button pressed (after sticky noise), so the trip can be replayed


def distance(place, goal_place):
    """Evaluation only: map cells apart, plus position blocks apart within a cell (8 blocks ~ one cell)."""
    return (abs(place[1] - goal_place[1]) + abs(place[2] - goal_place[2])
            + (abs(place[3] - goal_place[3]) + abs(place[4] - goal_place[4])) / 8)


def _start_worker(params, byte_indices):
    # The player runs on the CPU, one thread per worker: 18 workers must not all grab the GPU or spawn
    # full thread pools. Set before JAX is imported in this process.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    global _emulator, _policy
    from policy import GoalPolicy
    _emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    _policy = GoalPolicy(params, byte_indices) if params is not None else None


def _run(trip):
    rng = np.random.default_rng(trip.seed)
    settings = Settings()
    goal_place = hand_made_place(trip.goal, settings)
    _emulator.load(trip.start)
    replay = iter(trip.replay)
    first = closest = distance(hand_made_place(_emulator.memory(), settings), goal_place)
    pressed, previous = bytearray(), 0
    for _ in range(trip.budget):
        memory = _emulator.memory()
        if trip.player == "learned":
            action = _policy.act(memory, trip.goal, rng)
        elif trip.player == "learned-greedy":
            action = _policy.act(memory, trip.goal, rng, temperature=0)
        elif trip.player == "random":
            action = int(rng.integers(len(ACTIONS)))
        else:
            action = next(replay, 0)
        if rng.random() < STICKY:
            action = previous
        _emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        pressed.append(action)
        previous = action
        place = hand_made_place(_emulator.memory(), settings)
        closest = min(closest, distance(place, goal_place))
        if place == goal_place:
            return Outcome(True, first, 0.0, bytes(pressed))
    return Outcome(False, first, closest, bytes(pressed))


def run(trips, policy=None, workers=18):
    """Outcomes of trips, in order. policy: a GoalPolicy, needed for the learned players."""
    params = None if policy is None else policy.host_params()
    byte_indices = None if policy is None else policy.byte_indices
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_start_worker,
                             initargs=(params, byte_indices)) as pool:
        return list(pool.map(_run, trips, chunksize=max(1, len(trips) // (workers * 8))))


def _start_sampler():
    global _emulator
    _emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)


def _sample(task):
    """In a worker: one trip from one recording, or None if its goal is already the starting place."""
    (state, actions), seed, gaps, player = task
    rng = np.random.default_rng(seed)
    start_step = int(rng.integers(0, len(actions) - gaps[0]))
    gap = int(rng.integers(gaps[0], min(gaps[1], len(actions) - start_step) + 1))
    _emulator.load(state)
    for action in actions[:start_step]:
        _emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    start = _emulator.save()
    for action in actions[start_step:start_step + gap]:
        _emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    goal = _emulator.memory()
    settings = Settings()
    if hand_made_place(_emulator.memory(start), settings) == hand_made_place(goal, settings):
        return None
    return Trip(start, goal, 2 * gap, player, tuple(int(a) for a in actions[start_step:start_step + gap]),
                int(rng.integers(2**31)))


def from_recordings(recordings, rng, count, gaps=(10, 30), player="learned", min_length=40, workers=18):
    """
    Trips sampled from recordings, in parallel: start at a random step, aim at where the recording was
    10-30 steps later, with twice that many steps. Trips whose goal is already the starting place are skipped.
    """
    usable = [r for r in recordings if len(r[1]) >= min_length]
    trips = []
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_start_sampler) as pool:
        while len(trips) < count:
            wanted = int((count - len(trips)) * 1.2) + workers
            tasks = [(usable[i], int(seed), gaps, player)
                     for i, seed in zip(rng.integers(len(usable), size=wanted), rng.integers(2**31, size=wanted))]
            trips += [t for t in pool.map(_sample, tasks, chunksize=max(1, wanted // (workers * 8))) if t]
    return trips[:count]


def summary(outcomes):
    reached = np.mean([o.reached for o in outcomes])
    closed = np.mean([(o.start_distance - o.closest_distance) / o.start_distance
                      for o in outcomes if o.start_distance > 0])
    return f"reached the goal in {reached:.0%} of {len(outcomes)} trips; closed {closed:.0%} of the distance"
