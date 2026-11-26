"""
Does a chooser trained on *downstream* outcomes learn that power matters? For each logged decision, the
downstream outcome is how much the visits it created later grew (1 + descendants each, summed, from the
archive's family tree). Fits the same logistic chooser on "found something new now" and on "led to a large
lineage later" (top share of decisions), and prints both sets of weights side by side.

Usage: python -m checks.downstream_check <run dir under runs/> [<run dir> ...] [--top 0.02]
"""

import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

from checks.drive_learning_check import auc, fit
from checks.value_check import descendants


def main():
    top = float(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0.02
    runs = [a for a in sys.argv[1:] if not a.startswith("--") and not a.replace(".", "").isdigit() and not a.endswith(".pkl")]
    xs, now, later = [], [], []
    for run in runs:
        run_dir = Path("runs") / run
        d = np.load(run_dir / "decisions.npz")
        saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
        size = descendants(saved["parents"])
        offsets, created = d["created_offsets"], d["created"]
        growth = np.array([sum(1 + size[c] for c in created[offsets[k]:offsets[k + 1]] if c < len(size))
                           for k in range(len(offsets) - 1)], float)
        xs.append(d["signals"].astype(float))
        now.append(d["found"].astype(bool))
        later.append(growth)
        names = d["names"]
    raw = np.concatenate(xs)
    x = (raw - raw.mean(0)) / (raw.std(0) + 1e-9)
    growth = np.concatenate(later)
    labels = {"found something now": np.concatenate(now),
              f"lineage in the top {top:.0%} later": growth >= np.quantile(growth[growth > 0], 1 - top) if (growth > 0).any() else growth > 0}
    split = int(len(x) * 0.8)
    print(f"{len(x):,} decisions; {np.mean(growth > 0):.2%} created any visit")
    for name, y in labels.items():
        w, b = fit(x[:split], y[:split].astype(float))
        print(f"{name:>28} ({y.mean():.2%}): held-out AUC {auc(x[split:] @ w + b, y[split:]):.3f}; weights " +
              ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, w)))
    if "--save" in sys.argv:
        from go_explore import fit_chooser
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        out.write_bytes(pickle.dumps(fit_chooser(raw, labels[list(labels)[1]].astype(float), steps=3000)))
        print(f"saved the long-range chooser -> {out}")


if __name__ == "__main__":
    main()
