"""
Drives as senses: can a simple learned model predict which exploration segments succeed (find a new place)
from the drive signals of the visit they started from? Features per visit (from the run's final archive, an
approximation): how often chosen, holdings, danger, surprises, deaths and successes so far (as rates), path
length. Fits logistic regression on earlier segments, tests on later ones (AUC), and prints the learned weight
of each (standardized) drive.

Usage: python -m checks.drive_learning_check <run dir under runs/> [<run dir> ...]
"""

import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

import danger
from emulator import Emulator
from explorers import ROM

NAMES = ["rarely chosen", "holdings", "danger", "surprise rate", "death rate", "success rate", "path length"]


def features(emulator, saved):
    chosen = np.array(saved["times_chosen"], float)
    held = np.array([sum(h) if len(h) else 0 for h in saved.get("holdings", [()] * len(chosen))], float)
    dangers = np.array(saved.get("dangers") or [danger.score("cache/danger-model.pkl", emulator.memory(s))
                                                 for s in saved["states"]], float)
    tries = 1 + chosen
    per_try = lambda key: np.array(saved.get(key) or [0] * len(chosen), float) / tries
    lengths = np.array([len(p) if p else 0 for p in saved["paths"]], float)
    return np.stack([1 / np.sqrt(tries), held, dangers[:len(chosen)], per_try("surprises"), per_try("deaths"),
                     per_try("productive"), lengths], 1)


def fit(x, y, steps=3000, rate=0.1):
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        w -= rate * (x.T @ (p - y) / len(y) + 1e-3 * w)
        b -= rate * (p - y).mean()
    return w, b


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives - 1) / 2) / (positives * (len(labels) - positives))


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    xs, ys = [], []
    for run in sys.argv[1:]:
        run_dir = Path("runs") / run
        saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
        segments = np.load(run_dir / "segments.npz")
        visit_features = features(emulator, saved)
        starts = segments["start"]
        keep = starts < len(visit_features)
        xs.append(visit_features[starts[keep]])
        ys.append(segments["new_places"][keep] > 0)
    x, y = np.concatenate(xs), np.concatenate(ys)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    split = int(len(y) * 0.8)
    w, b = fit(x[:split], y[:split].astype(float))
    print(f"{len(y):,} segments, {y.mean():.1%} found something new; held-out AUC {auc(x[split:] @ w + b, y[split:]):.3f}")
    print("learned weights (standardized): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, w)))


if __name__ == "__main__":
    main()
