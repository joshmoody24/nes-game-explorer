"""
Things random play spends (ammo, health; in other games bombs, keys, rupees), found without knowing the
game, for hoarding (see explorers.Settings.keep_more). A spendable byte, over recorded random play:
- drops in small steps far more often than it rises (counters and timers rise and wrap),
- changes rarely (a level that holds for long stretches; per-object bytes tick constantly),
- survives scene changes (steps where much of memory changes at once, like a room loading),
- depends on what I do: when a state holds some of it, two random plays from there leave different amounts
  (a timer wouldn't), judged only where there is some to spend.
"""

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from experience import build

PLAY_STEPS = 120
CONTROL_STARTS = 6000


def _paired(task):
    """In a worker: for each state, (which memory bytes it holds some of, which end up different after two
    random plays)."""
    states, seed = task
    emulator, rng = workers.emulator(), np.random.default_rng(seed)
    held, differs = [], []
    for state in states:
        held.append(emulator.memory(state) > 0)
        ends = []
        for _ in range(2):
            emulator.load(state)
            for _ in range(PLAY_STEPS):
                emulator.run(ACTIONS[int(rng.integers(len(ACTIONS)))], FRAMES_PER_STEP)
            ends.append(emulator.memory())
        differs.append(ends[0] != ends[1])
    return held, differs


def scores(emulator, recordings):
    """(score per memory byte, higher = more like a spendable, 0 = not; largest value seen per byte)."""
    replayed = build(emulator, recordings[:3000], np.arange(len(emulator.memory())))
    changes = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0)
                              for r in range(len(replayed.start_states))])
    decreases, increases = (changes < 0).mean(0), (changes > 0).mean(0)
    small = (np.abs(changes) <= 16).mean(0)
    changed_share = (changes != 0).mean(1)
    scene_changes = changed_share >= np.quantile(changed_share, 0.995)
    reset_rate = (changes[scene_changes] != 0).mean(0)
    rare = (decreases + increases) < 0.002
    # Many starts: a resource may be held in only a few percent of them, and control is judged only there.
    starts = [state for state, _ in recordings[:CONTROL_STARTS]]
    chunks = [starts[i:i + 50] for i in range(0, len(starts), 50)]
    results = list(workers.map(_paired, [(c, k) for k, c in enumerate(chunks)]))
    held = np.array([h for result, _ in results for h in result])
    differs = np.array([d for _, result in results for d in result])
    controllable = (held & differs).sum(0) / np.maximum(held.sum(0), 1)
    # Gates: mostly decreasing, in small steps, rarely, surviving scene changes. Ranked by control: the most
    # spendable things are the ones my actions most reliably change when I have some.
    gates = (decreases > 1e-4) & (decreases > 2 * increases) & (small > 0.999) & (reset_rate < 0.05) & rare
    score = np.where(gates, controllable, 0)
    return score, replayed.observations.max(0)


def detect(emulator, recordings, count=4, seed=0):
    """The count most spendable-looking bytes, as "index:scale" terms for Settings.keep_more."""
    recordings = list(recordings)
    np.random.default_rng(seed).shuffle(recordings)
    score, largest = scores(emulator, recordings)
    return [f"{b}:{max(1, int(largest[b]))}" for b in np.argsort(-score)[:count] if score[b] > 0]
