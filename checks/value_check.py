"""
Does a value learned from exploration's own family tree recognize what makes a state good? Trains, from memory,
log(1 + number of descendants) of each kept visit (how much new territory followed from it), then asks the
model counterfactuals: the same visits with a byte edited low vs high (e.g. missiles 0 vs 5), near a chosen
room and everywhere. Diagnostic (the room and byte are named by hand only to read the result).

Usage: python -m checks.value_check <run name> <byte> <low> <high> [area,y,x] [--per-try]
"""

import os
import pickle
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from emulator import Emulator
from experience import choose_bytes, valid_recordings
from explorers import ROM
from graph import location
from training import init_mlp, mlp


def descendants(parents):
    """Number of descendants of every visit in the archive's family tree (cycles from slot reuse are cut)."""
    children = defaultdict(list)
    for child, parent in enumerate(parents):
        if 0 <= parent != child:
            children[parent].append(child)
    counts = np.zeros(len(parents), int)
    seen = set()
    for root in range(len(parents)):
        if root in seen:
            continue
        stack, order = [root], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            stack.extend(children[node])
        for node in reversed(order):
            counts[node] = sum(1 + counts[c] for c in children[node])
    return counts


def main():
    run, byte, low, high = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    room = tuple(int(x) for x in sys.argv[5].split(",")) if len(sys.argv) > 5 and "," in sys.argv[5] else None
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress((Path("runs") / run / "checkpoint.pkl.z").read_bytes()))
    memories = np.stack([emulator.memory(s) for s in saved["states"]])
    if "--per-try" in sys.argv:
        # Direct discoveries per try: children found / times explored from the visit (not confounded by age).
        children = np.bincount([p for p in saved["parents"] if p >= 0], minlength=len(saved["parents"]))
        targets = np.log1p(children / (1 + np.array(saved["times_chosen"]))).astype(np.float32)
    else:
        targets = np.log1p(descendants(saved["parents"])).astype(np.float32)
    byte_indices = np.union1d(choose_bytes(emulator, valid_recordings(Path("runs") / run)[:1000]), [byte])
    x = memories[:, byte_indices].astype(np.float32) / 255
    rng = np.random.default_rng(0)
    test = rng.random(len(x)) < 0.2
    params = init_mlp(jax.random.key(0), (x.shape[1], 256, 128, 1))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    xs, ys = jnp.asarray(x[~test]), jnp.asarray(targets[~test])

    @jax.jit
    def step(params, opt_state, key):
        idx = jax.random.randint(key, (1024,), 0, len(ys))
        loss = lambda p: jnp.mean((mlp(p, xs[idx])[:, 0] - ys[idx]) ** 2)
        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, value

    for k in jax.random.split(jax.random.key(1), 4000):
        params, opt_state, _ = step(params, opt_state, k)
    predict = lambda a: np.asarray(mlp(params, jnp.asarray(a)))[:, 0]
    pred = predict(x[test])
    print(f"{len(x)} visits; held-out correlation of predicted vs actual target: "
          f"{np.corrcoef(pred, targets[test])[0, 1]:.2f}")
    column = int(np.flatnonzero(byte_indices == byte)[0])
    groups = {"everywhere": np.arange(len(x))}
    if room:
        groups[f"room {room}"] = np.array([i for i, m in enumerate(memories) if location(m)[:3] == room])
    for name, idx in groups.items():
        if len(idx) == 0:
            print(f"  {name}: no visits")
            continue
        lo, hi = x[idx].copy(), x[idx].copy()
        lo[:, column], hi[:, column] = low / 255, high / 255
        gain = predict(hi) - predict(lo)
        print(f"  {name} ({len(idx)} visits): value with byte {byte} = {high} minus = {low}: mean {gain.mean():+.3f}, "
              f"higher for {np.mean(gain > 0):.0%} of visits")


if __name__ == "__main__":
    main()
