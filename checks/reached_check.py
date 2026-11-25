"""
Could a generic test replace hand-made place equality for "reached the waypoint"? On held-out recordings,
compares pairs (state, later state) labeled same place / not, scored by the learned distance and by the
share of chosen bytes that match. Reports AUC and the accuracy of the best threshold.

Usage: python -m checks.reached_check <run name> [teacher cache file]
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

import distance
from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM, Settings, hand_made_place


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives - 1) / 2) / (positives * (len(labels) - positives))


def best_accuracy(scores, labels):
    return max(np.mean((scores >= threshold) == labels) for threshold in np.quantile(scores, np.linspace(0, 1, 201)))


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / sys.argv[1])
    np.random.default_rng(0).shuffle(recordings)
    teacher = sys.argv[2] if len(sys.argv) > 2 else "cache/teacher-v2-policy-data-20000-20000-20000-4.pkl"
    params, byte_indices, _, _ = pickle.loads(Path(teacher).read_bytes())
    judge = distance.Distance(params, byte_indices)
    held_out = build(emulator, recordings[60000:61000], np.arange(len(emulator.memory())))
    rng = np.random.default_rng(1)
    firsts, seconds = [], []
    for _ in range(20000):
        r = int(rng.integers(len(held_out.start_states)))
        memories = held_out.recording(r)[0]
        t = int(rng.integers(len(memories)))
        gap = int(rng.integers(0, 41))
        firsts.append(memories[t])
        seconds.append(memories[min(t + gap, len(memories) - 1)])
    firsts, seconds = np.stack(firsts), np.stack(seconds)
    settings = Settings()
    labels = np.array([hand_made_place(a, settings) == hand_made_place(b, settings) for a, b in zip(firsts, seconds)])
    scores = {"learned distance": -np.concatenate([judge(firsts[i:i + 1000], seconds[i:i + 1000])
                                                   for i in range(0, len(firsts), 1000)]),
              "share of bytes matching": (firsts[:, byte_indices] == seconds[:, byte_indices]).mean(1)}
    print(f"{labels.mean():.0%} of pairs are the same hand-made place")
    for name, score in scores.items():
        print(f"{name:>24}: AUC {auc(score, labels):.3f}, best-threshold accuracy {best_accuracy(score, labels):.1%}")


if __name__ == "__main__":
    main()
