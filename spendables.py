"""
Things random play spends (ammo, health; in other games bombs, keys, rupees), found without knowing the
game, for hoarding (see explorers.Settings.keep_more). A spendable byte, over recorded random play:
- drops in small steps far more often than it rises (counters and timers rise and wrap),
- changes rarely (a level that holds for long stretches; per-object bytes tick constantly),
- survives scene changes (steps where much of memory changes at once, like a room loading),
- depends on what I do: two random plays from one state leave it different (a timer wouldn't).
"""

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from experience import build

PLAY_STEPS = 120


def _paired(task):
    """In a worker: for each state, which memory bytes end up different after two random plays."""
    states, seed = task
    emulator, rng = workers.emulator(), np.random.default_rng(seed)
    differs = []
    for state in states:
        ends = []
        for _ in range(2):
            emulator.load(state)
            for _ in range(PLAY_STEPS):
                emulator.run(ACTIONS[int(rng.integers(len(ACTIONS)))], FRAMES_PER_STEP)
            ends.append(emulator.memory())
        differs.append(ends[0] != ends[1])
    return differs


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
    starts = [state for state, _ in recordings[:2000]]
    chunks = [starts[i:i + 50] for i in range(0, len(starts), 50)]
    controllable = np.mean([d for result in workers.map(_paired, [(c, k) for k, c in enumerate(chunks)])
                            for d in result], 0)
    score = np.where((decreases > 1e-4) & (small > 0.999) & (reset_rate < 0.05) & rare & (controllable >= 0.02),
                     decreases / (increases + 1e-4), 0)
    return score, replayed.observations.max(0)


def detect(emulator, recordings, count=4):
    """The count most spendable-looking bytes, as "index:scale" terms for Settings.keep_more."""
    score, largest = scores(emulator, recordings)
    return [f"{b}:{max(1, int(largest[b]))}" for b in np.argsort(-score)[:count] if score[b] > 0]
