"""
Steady-state training speed of the goal policy on synthetic data shaped like the real thing, so speed
changes can be compared in seconds. Compile time is excluded (one warm-up call).

Usage: python -m checks.train_speed_check [batch] [updates per call] [--queue: don't wait between calls] [--fp32]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import policy
from policy import OPTIMIZER, _train_chunk, init_params

WIDTH, RECORDINGS, CALLS = 3080, 4000, 10


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    if "--fp32" in sys.argv:
        policy.TRAINING_DTYPE = jnp.float32
    updates = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rng = np.random.default_rng(0)
    lengths = rng.integers(40, 100, RECORDINGS)
    action_offsets = np.concatenate([[0], np.cumsum(lengths)])
    observation_offsets = np.concatenate([[0], np.cumsum(lengths + 1)])
    data = tuple(map(jnp.asarray, (rng.integers(0, 256, (observation_offsets[-1], WIDTH), dtype=np.uint8),
                                   rng.integers(0, 25, action_offsets[-1], dtype=np.uint8),
                                   observation_offsets, action_offsets)))
    params = init_params(jax.random.key(0), 2 * WIDTH, 25)
    state = OPTIMIZER.init(params)
    key = jax.random.key(1)
    params, state, values = _train_chunk(params, state, key, data, updates, batch)
    values.block_until_ready()
    started = time.perf_counter()
    for call in range(CALLS):
        params, state, values = _train_chunk(params, state, jax.random.fold_in(key, call), data, updates, batch)
        if "--queue" not in sys.argv:
            values.block_until_ready()
    values.block_until_ready()
    rate = CALLS * updates / (time.perf_counter() - started)
    print(f"batch {batch}, {updates} updates per call: {rate:,.0f} updates/s "
          f"({rate * batch:,.0f} examples/s); PyTorch reference: 767 updates/s at batch 512", flush=True)


if __name__ == "__main__":
    main()
