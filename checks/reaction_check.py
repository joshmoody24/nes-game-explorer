"""
Where does the world react? Replays recorded segments with every memory byte, finds steps where a
rarely-changing byte takes a value it never had earlier in the data, and counts those reactions per location
(place without inventory). Shows where the Long Beam door room ranks (diagnostic only).

Usage: python -m checks.reaction_check <run name> [recordings]
"""

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM, Settings, hand_made_place

RARE_CHANGE_RATE = 0.002


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    np.random.default_rng(0).shuffle(recordings)
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    replayed = build(emulator, recordings[:count], np.arange(len(emulator.memory())))
    steps = [replayed.recording(r)[0] for r in range(len(replayed.start_states))]
    change_rate = np.concatenate([np.diff(s.astype(np.int16), axis=0) != 0 for s in steps]).mean(0)
    rare = change_rate < RARE_CHANGE_RATE
    seen = np.zeros((replayed.observations.shape[1], 256), bool)
    for s in steps:
        seen[np.arange(s.shape[1]), s[0]] = True
    reactions = Counter()
    settings = Settings()
    for s in steps:
        for t in range(1, len(s)):
            changed = np.flatnonzero(rare & (s[t] != s[t - 1]))
            new = [b for b in changed if not seen[b, s[t, b]]]
            for b in changed:
                seen[b, s[t, b]] = True
            if new:
                reactions[hand_made_place(s[t], settings)[:5]] += len(new)
    ranked = [loc for loc, _ in reactions.most_common()]
    door = [k for k, loc in enumerate(ranked) if loc[:3] == (0, 5, 8)]
    print(f"{sum(reactions.values())} reactions at {len(ranked)} locations; door-room ranks {door[:5]}; "
          f"top 8: {[(loc[:3], reactions[loc]) for loc in ranked[:8]]}")


if __name__ == "__main__":
    main()
