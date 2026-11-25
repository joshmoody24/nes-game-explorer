"""
Goal-directed trips run in the worker processes (see workers.py), each with its own emulator and a CPU
copy of the player. A trip starts from a save state, aims at a goal (a full memory snapshot), and lasts a
budget of steps under sticky-action noise. Each trip returns the buttons pressed, so trips double as
training data. Success is judged with hand-made places, which is evaluation-only Metroid knowledge.
"""

import functools
import pickle
import tempfile
from dataclasses import dataclass

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from explorers import Settings, hand_made_place
from metroid import GAME_MODE, GAME_OVER

STICKY = 0.25  # share of steps that repeat the previous button instead of the chosen one
MAX_LOCKSTEP = 32  # most trips a worker runs side by side


@dataclass(frozen=True)
class Trip:
    start: bytes  # save state
    goal: np.ndarray  # full memory snapshot to aim for
    budget: int  # steps allowed per target (each waypoint, then the goal)
    # "learned", "learned-greedy", "random", "replay", or a search player that looks ahead in the emulator:
    # "search" (judges outcomes by memory bytes matching the goal), "search-learned" (by a learned
    # distance.Distance, passed as the model), or "search-true-distance" (diagnostic: hand-made places)
    player: str
    replay: tuple = ()  # buttons to press, for the "replay" player
    seed: int = 0
    # Full memory snapshots to reach in order before the goal. Landing on a later one skips ahead.
    waypoints: tuple = ()


@dataclass(frozen=True)
class Outcome:
    reached: bool
    start_distance: float
    closest_distance: float
    actions: bytes  # every button pressed (after sticky noise), so the trip can be replayed
    chosen: bytes  # the button the player chose at each step, before sticky noise
    search_scores: np.ndarray | None  # search players: [steps, buttons] how good each button looked (higher better)
    targets_reached: int  # waypoints and goal reached
    progress_steps: tuple  # steps taken each time the trip reached a waypoint or the goal
    died: bool  # evaluation only: game over


@dataclass(frozen=True)
class _Setup:
    """What every trip in one run() shares."""
    model_path: str | None  # the model, pickled in shared memory: too big to send with every chunk
    byte_indices: np.ndarray | None  # compared by the "search" player
    search_hold: int  # steps each candidate button is held during a search player's lookahead


def distance(place, goal_place):
    """Evaluation only: map cells apart, plus position blocks apart within a cell (8 blocks ~ one cell)."""
    return (abs(place[1] - goal_place[1]) + abs(place[2] - goal_place[2])
            + (abs(place[3] - goal_place[3]) + abs(place[4] - goal_place[4])) / 8)


@functools.lru_cache(maxsize=1)
def _model(path):
    """In a worker: the model a run uses, loaded once per run."""
    import jax
    assert jax.default_backend() == "cpu", "models must run on the CPU in workers; see workers._CPU_ONLY"
    model_class, params, byte_indices = pickle.loads(open(path, "rb").read())
    return model_class(params, byte_indices)


def _searched(setup, goal, judge, rng):
    """
    Search player: try every button for setup.search_hold steps from the current state and keep the best
    outcome: judge is "bytes" (most memory bytes matching the goal), "learned" (smallest learned distance),
    or "true" (diagnostic: smallest hand-made place distance). Uses save states: a teacher for training,
    not a way to play. Returns the button and every button's score.
    """
    emulator = workers.emulator()
    start = emulator.save()
    outcomes = []
    for action in range(len(ACTIONS)):
        emulator.load(start)
        for _ in range(setup.search_hold):
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        outcomes.append(emulator.memory())
    emulator.load(start)
    outcomes = np.stack(outcomes)
    if judge == "learned":
        scores = -_model(setup.model_path)(outcomes, np.broadcast_to(goal, outcomes.shape))
    elif judge == "true":
        goal_place = hand_made_place(goal, Settings())
        scores = np.array([-distance(hand_made_place(memory, Settings()), goal_place) for memory in outcomes])
    else:
        scores = (outcomes[:, setup.byte_indices] == goal[setup.byte_indices]).sum(1)
    # Random among ties: with a coarse distance most buttons tie, and always taking the first (no button)
    # would mean standing still.
    return int(rng.choice(np.flatnonzero(scores == scores.max()))), scores


def _sampled(scores, rng):
    probabilities = np.exp(scores - scores.max())
    return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))


def _run_chunk(task):
    """
    In a worker: runs trips in lockstep, so a learned player decides for all of them in one batch: deciding
    for one trip at a time reads the player's whole first layer (~19 MB) per decision, more memory traffic
    than 18 workers can get. One emulator juggles the trips by save state (loading one takes ~1 us).
    """
    setup, chunk = task
    emulator = workers.emulator()
    settings = Settings()
    rngs = [np.random.default_rng(trip.seed) for trip in chunk]
    targets = [[*trip.waypoints, trip.goal] for trip in chunk]
    target_places = [[hand_made_place(t, settings) for t in trip_targets] for trip_targets in targets]
    goal_places = [places[-1] for places in target_places]
    replays = [iter(trip.replay) for trip in chunk]
    states = [trip.start for trip in chunk]
    first = [distance(hand_made_place(emulator.memory(s), settings), g) for s, g in zip(states, goal_places)]
    closest = list(first)
    pressed = [bytearray() for _ in chunk]
    chosen = [bytearray() for _ in chunk]
    search_scores = [[] for _ in chunk]
    current = [0] * len(chunk)  # index of the target each trip is heading for
    progress_steps = [[] for _ in chunk]
    steps_on_target = [0] * len(chunk)
    died = [False] * len(chunk)
    learned = any(trip.player.startswith("learned") for trip in chunk)

    def finished(i):
        return current[i] == len(targets[i]) or died[i] or steps_on_target[i] >= chunk[i].budget

    while not all(finished(i) for i in range(len(chunk))):
        active = [i for i in range(len(chunk)) if not finished(i)]
        aims = np.stack([targets[i][min(current[i], len(targets[i]) - 1)] for i in range(len(chunk))])
        # Scores for every trip, finished or not, so the batch keeps one shape and compiles once.
        scores = _model(setup.model_path)(np.stack([emulator.memory(s) for s in states]), aims) if learned else None
        for i in active:
            trip, rng = chunk[i], rngs[i]
            emulator.load(states[i])
            if trip.player == "learned":
                action = _sampled(scores[i], rng)
            elif trip.player == "learned-greedy":
                action = int(scores[i].argmax())
            elif trip.player.startswith("search"):
                judge = {"search": "bytes", "search-learned": "learned", "search-true-distance": "true"}[trip.player]
                action, button_scores = _searched(setup, aims[i], judge, rng)
                search_scores[i].append(button_scores)
            elif trip.player == "random":
                action = int(rng.integers(len(ACTIONS)))
            else:
                action = next(replays[i], 0)
            chosen[i].append(action)
            if rng.random() < STICKY:
                action = pressed[i][-1] if pressed[i] else 0
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            states[i] = emulator.save()
            pressed[i].append(action)
            memory = emulator.memory(states[i])
            place = hand_made_place(memory, settings)
            closest[i] = min(closest[i], distance(place, goal_places[i]))
            died[i] = memory[GAME_MODE] == GAME_OVER
            steps_on_target[i] += 1
            matched = [j for j in range(current[i], len(targets[i])) if target_places[i][j] == place]
            if matched:
                current[i], steps_on_target[i] = matched[-1] + 1, 0
                progress_steps[i].append(len(pressed[i]))
    return [Outcome(n == len(t), f, 0.0 if n == len(t) else c, bytes(p), bytes(h),
                    np.array(s, np.float32) if s else None, n, tuple(g), d)
            for n, t, f, c, p, h, s, g, d in zip(current, targets, first, closest, pressed, chosen, search_scores,
                                                 progress_steps, died)]


def run(trips, model=None, byte_indices=None, search_hold=2):
    """
    Outcomes of trips, in order. model: needed by the learned players (policy.GoalPolicy, critic.Critic)
    and "search-learned" (distance.Distance). The "search" player compares the bytes in byte_indices (or
    the model's), holding each candidate for search_hold steps.
    """
    with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".model") as model_file:
        if model is not None:
            model_file.write(pickle.dumps((type(model), model.host_params(), model.byte_indices)))
            model_file.flush()
            byte_indices = model.byte_indices
        setup = _Setup(None if model is None else model_file.name, byte_indices, search_hold)
        chunks = workers.split(trips, pieces_per_worker=2, most=MAX_LOCKSTEP)
        return [outcome for outcomes in workers.map(_run_chunk, [(setup, chunk) for chunk in chunks])
                for outcome in outcomes]


def _sample(task):
    """In a worker: one trip from one recording, or None if its goal is already the starting place."""
    (state, actions), seed, gaps, player = task
    emulator = workers.emulator()
    rng = np.random.default_rng(seed)
    start_step = int(rng.integers(0, len(actions) - gaps[0]))
    gap = int(rng.integers(gaps[0], min(gaps[1], len(actions) - start_step) + 1))
    emulator.load(state)
    for action in actions[:start_step]:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    start = emulator.save()
    for action in actions[start_step:start_step + gap]:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
    goal = emulator.memory()
    settings = Settings()
    if hand_made_place(emulator.memory(start), settings) == hand_made_place(goal, settings):
        return None
    return Trip(start, goal, 2 * gap, player, tuple(int(a) for a in actions[start_step:start_step + gap]),
                int(rng.integers(2**31)))


def _sample_chunk(tasks):
    return [_sample(task) for task in tasks]


def from_recordings(recordings, rng, count, gaps=(10, 30), player="learned", min_length=40):
    """
    Trips sampled from recordings, in parallel: start at a random step, aim at where the recording was
    10-30 steps later, with twice that many steps. Trips whose goal is already the starting place are skipped.
    """
    usable = [r for r in recordings if len(r[1]) >= min_length]
    trips = []
    while len(trips) < count:
        wanted = int((count - len(trips)) * 1.2) + workers.COUNT
        tasks = [(usable[i], int(seed), gaps, player)
                 for i, seed in zip(rng.integers(len(usable), size=wanted), rng.integers(2**31, size=wanted))]
        trips += [trip for chunk in workers.map(_sample_chunk, workers.split(tasks)) for trip in chunk if trip]
    return trips[:count]


def summary(outcomes):
    reached = np.mean([o.reached for o in outcomes])
    closed = np.mean([(o.start_distance - o.closest_distance) / o.start_distance
                      for o in outcomes if o.start_distance > 0])
    return f"reached the goal in {reached:.0%} of {len(outcomes)} trips; closed {closed:.0%} of the distance"
