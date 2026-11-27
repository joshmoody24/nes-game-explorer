"""Camera moves measured from pixels (1-D row/column profile matching) vs the hardware scroll change.
Usage: python -m checks.pixel_scroll_check <checkpoint.pkl.z>"""
import os, pickle, sys, zlib
from collections import Counter
from pathlib import Path
import numpy as np
from controls import FRAMES_PER_STEP, actions
from emulator import Emulator
from explorers import ROM, SCROLL_RANGE
e = Emulator(os.environ['QUICKNES_CORE'], ROM)
s = pickle.loads(zlib.decompress(Path(sys.argv[1]).read_bytes()))['states']
rng = np.random.default_rng(0); buttons = actions(True)
def shift(a, b, most=16):
    """The shift d (pixels) that best lines up profile b with profile a moved by d."""
    a, b = a - a.mean(), b - b.mean()
    best = min(range(-most, most + 1), key=lambda d: np.mean((a[max(0, d):len(a) + min(0, d)] - b[max(0, -d):len(b) + min(0, -d)]) ** 2))
    return best
results = Counter(); n = 0
for i in rng.choice(len(s), 150, replace=False):
    e.load(s[i]); e.run(0, 1, picture=True); prev_g = e.gray(1); prev_s = e.scroll(e.save()); a = rng.integers(len(buttons))
    for _ in range(40):
        if rng.random() > 0.9: a = rng.integers(len(buttons))
        e.run(buttons[a], FRAMES_PER_STEP, picture=True)
        g = e.gray(1); sc = e.scroll(e.save())
        true = tuple((c - p + r // 2) % r - r // 2 for p, c, r in zip(prev_s, sc, SCROLL_RANGE))
        body = slice(40, None)  # skip the top rows where a status bar usually sits
        est = (shift(prev_g[body].mean(axis=0), g[body].mean(axis=0)), shift(prev_g[body].mean(axis=1), g[body].mean(axis=1)))
        if true != (0, 0) and max(abs(t) for t in true) <= 16 and False:
            results['shown'] += 1; print('true', true, 'measured', est)
        if max(abs(t) for t in true) <= 16:
            moving = true != (0, 0)
            results[(moving, est == true)] += 1; results[(moving, all(abs(x - t) <= 1 for x, t in zip(est, true)))  , 'within1'] += 0
            results[('within 1px', moving)] += all(abs(x - t) <= 1 for x, t in zip(est, true))
            results[('count', moving)] += 1
        prev_g, prev_s = g, sc
for moving in (False, True):
    total = results[('count', moving)]
    print(f"{'camera moving' if moving else 'camera still '}: {total} steps, exact {results[(moving, True)] / total:.0%}, within 1 px {results[('within 1px', moving)] / total:.0%}")
