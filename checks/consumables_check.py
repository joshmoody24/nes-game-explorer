"""
Can "consumables" (ammo, health; in Zelda bombs, keys, hearts) be found without knowing the game? Ranks
memory bytes by how much more often random play decreases them than increases them, and shows where the
known Metroid resource bytes rank.

Usage: python -m checks.consumables_check <run name>
"""

import os
import sys
from pathlib import Path

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, at

PLAY_STEPS = 120
NAMED = {"missiles": at(CURRENT_MISSILES), "energy (tens)": 0x107, "energy (units)": 0x106}


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


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    np.random.default_rng(0).shuffle(recordings)
    replayed = build(emulator, recordings[:3000], np.arange(len(emulator.memory())))
    changes = np.concatenate([np.diff(replayed.recording(r)[0].astype(np.int16), axis=0)
                              for r in range(len(replayed.start_states))])
    decreases, increases = (changes < 0).mean(0), (changes > 0).mean(0)
    # A consumable drops in small steps far more often than it rises; counters and timers rise and wrap.
    small = (np.abs(changes) <= 16).mean(0)
    # It also survives scene changes (steps where a large share of memory changes at once, e.g. a room
    # loading), which reset per-scene bytes like enemy health.
    changed_share = (changes != 0).mean(1)
    scene_changes = changed_share >= np.quantile(changed_share, 0.995)
    reset_rate = (changes[scene_changes] != 0).mean(0)
    print(f"{scene_changes.sum()} scene-change steps (>= {changed_share[scene_changes].min():.0%} of bytes changed)")
    # And a resource level holds for long stretches: counters and per-object bytes tick far more often.
    rare = (decreases + increases) < 0.002
    score = np.where((decreases > 1e-4) & (small > 0.999) & (reset_rate < 0.05) & rare, decreases / (increases + 1e-4), 0)
    # A resource depends on what I do: two different plays from one state leave it different; a timer doesn't.
    starts = [state for state, _ in recordings[:2000]]
    chunks = [starts[i:i + 50] for i in range(0, len(starts), 50)]
    controllable = np.mean([d for result in workers.map(_paired, [(c, k) for k, c in enumerate(chunks)]) for d in result], 0)
    print("before requiring dependence on play: missiles rank", int(np.flatnonzero(np.argsort(-score) == 4217)[0]),
          "energy rank", int(np.flatnonzero(np.argsort(-score) == 263)[0]))
    score = np.where(controllable >= 0.02, score, 0)
    ranked = np.argsort(-score)
    names = {index: name for name, index in NAMED.items()}
    print("top 20 (index: decreases/increases per step):")
    print("  " + ", ".join(f"{b:#06x}{'=' + names[b] if b in names else ''}: {decreases[b]:.4f}/{increases[b]:.4f}, "
                           f"differs between plays {controllable[b]:.0%}" for b in ranked[:20]))
    picks = [int(b) for b in ranked[:12] if score[b] > 0]
    print("increases per step, rarest first: " + ", ".join(f"{b}: {increases[b]:.2e}" for b in sorted(ranked[:4], key=lambda b: increases[b])))
    print("for --keep-more: " + ",".join(f"{b}:{max(1, int(replayed.observations[:, b].max()))}" for b in picks))
    for name, index in NAMED.items():
        print(f"  {name:>15} ({index:#06x}): rank {int(np.flatnonzero(ranked == index)[0])}, "
              f"decreases {decreases[index]:.4f}, increases {increases[index]:.4f} per step, "
              f"changes at scene changes {reset_rate[index]:.1%}")


if __name__ == "__main__":
    main()
