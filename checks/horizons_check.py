"""
Multi-horizon chooser, offline: does combining a chooser trained on immediate outcomes with one trained on
long-range outcomes predict long-range success better than either alone? Uses decisions logged with the
visits they created (see checks.downstream_check). Combination: sum of the two standardized logits, plus a
learned mix fitted on held-in data.

Usage: python -m checks.horizons_check <run dir under runs/> [<run dir> ...] [--top 0.02]
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
    runs = [a for a in sys.argv[1:] if not a.startswith("--") and not a.replace(".", "").isdigit()]
    xs, now, growths = [], [], []
    for run in runs:
        run_dir = Path("runs") / run
        d = np.load(run_dir / "decisions.npz")
        saved = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
        size = descendants(saved["parents"])
        offsets, created = d["created_offsets"], d["created"]
        growths.append(np.array([sum(1 + size[c] for c in created[offsets[k]:offsets[k + 1]] if c < len(size))
                                 for k in range(len(offsets) - 1)], float))
        xs.append(d["signals"].astype(float))
        now.append(d["found"].astype(float))
    x = np.concatenate(xs)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    now, growth = np.concatenate(now), np.concatenate(growths)
    later = (growth >= np.quantile(growth[growth > 0], 1 - top)).astype(float)
    split = int(len(x) * 0.8)
    w_now, b_now = fit(x[:split], now[:split])
    w_later, b_later = fit(x[:split], later[:split])
    standardize = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    s_now, s_later = standardize(x @ w_now + b_now), standardize(x @ w_later + b_later)
    mix_w, mix_b = fit(np.stack([s_now[:split], s_later[:split]], 1), later[:split])
    test = slice(split, None)
    print(f"{len(x):,} decisions; long-range label = top {top:.0%} lineage growth ({later.mean():.2%})")
    for name, score in [("immediate chooser", s_now), ("downstream chooser", s_later),
                        ("sum of both", s_now + s_later),
                        (f"learned mix ({mix_w[0]:+.2f} now, {mix_w[1]:+.2f} later)",
                         np.stack([s_now, s_later], 1) @ mix_w)]:
        print(f"  {name:>38}: held-out AUC for long-range success {auc(score[test], later[test].astype(bool)):.3f}")


if __name__ == "__main__":
    main()
