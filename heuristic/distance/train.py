"""
Trains an encoder RAM -> embedding so that embedding distance predicts log(1 + steps apart) for two states of the
same replayed trajectory. Training runs only (see common.TEST_PREFIXES). Saves model.npz.
Usage: JAX_PLATFORMS=cpu python heuristic/distance/train.py [steps]
"""
import pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, optax
from common import runs, load, change_rate, EVERY

# Bytes changing between most consecutive samples (16 frames apart) behave like timers, frame counters or RNG.
MAX_CHANGE_RATE = 0.7
DIMENSIONS = 32
HIDDEN = (512, 256)
BATCH = 1024


def choose_bytes(memory, trajectory):
    keep = (memory != memory[0]).any(0) & (change_rate(memory, trajectory) <= MAX_CHANGE_RATE)
    chosen = np.where(keep)[0]
    mode = np.array([np.bincount(memory[:, i], minlength=256).argmax() for i in chosen], np.uint8)
    return chosen, mode


def features(raw, mode):
    """raw: selected bytes (uint8). Each byte gives its value and whether it differs from its usual value."""
    return jnp.concatenate([raw / 255.0, (raw != mode).astype(jnp.float32)], -1)


def encode(params, raw, mode):
    x = features(raw, mode)
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def predicted_distance(params, mode, a, b):
    za, zb = encode(params, a, mode), encode(params, b, mode)
    return jnp.sqrt(((za - zb) ** 2).sum(-1) + 1e-6)


def init(key, inputs):
    sizes = (inputs, *HIDDEN, DIMENSIONS)
    keys = jax.random.split(key, len(sizes))
    return [(jax.random.normal(k, (m, n)) * np.sqrt(1 / m), jnp.zeros(n)) for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


def sample_pairs(rng, step, starts, ends, count):
    """Anchor anywhere; partner in the same trajectory, half at a log-uniform gap (near pairs), half uniform (far)."""
    i = rng.integers(0, len(step), count)
    lo, hi = starts[i], ends[i]
    length = hi - lo
    gap = np.where(rng.random(count) < 0.5, np.exp(rng.uniform(0, np.log(length), count)) - 1, rng.uniform(0, length, count)).astype(int)
    j = np.clip(i + np.where(rng.random(count) < 0.5, gap, -gap), lo, hi - 1)
    return i, j, np.log1p(np.abs(step[i] - step[j]))


def trajectory_bounds(trajectory):
    change = np.r_[0, np.where(trajectory[1:] != trajectory[:-1])[0] + 1, len(trajectory)]
    lengths = np.diff(change)
    return np.repeat(change[:-1], lengths), np.repeat(change[1:], lengths)


def stack(paths):
    data = [load(p) for p in paths]
    memory = np.concatenate([d["memory"] for d in data])
    trajectory = np.concatenate([d["trajectory"] + 1000 * k for k, d in enumerate(data)])
    step = np.concatenate([d["step"] for d in data])
    return memory, trajectory, step


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    memory, trajectory, step = stack(runs(test=False))
    chosen, mode = choose_bytes(memory, trajectory)
    raw = jnp.asarray(memory[:, chosen])
    starts, ends = trajectory_bounds(trajectory)
    print(f"{len(memory)} training states, {len(chosen)} bytes chosen ({(chosen < 2048).sum()} system RAM)", flush=True)

    params = init(jax.random.PRNGKey(0), 2 * len(chosen))
    schedule = optax.cosine_decay_schedule(1e-3, steps)
    optimizer = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    mode_j = jnp.asarray(mode)

    @jax.jit
    def update(params, opt_state, i, j, target):
        def loss(params):
            return ((predicted_distance(params, mode_j, raw[i], raw[j]) - target) ** 2).mean()
        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    rng = np.random.default_rng(0)
    began, recent = time.time(), []
    for k in range(steps):
        i, j, target = sample_pairs(rng, step, starts, ends, BATCH)
        params, opt_state, value = update(params, opt_state, i, j, jnp.asarray(target, jnp.float32))
        recent.append(float(value))
        if (k + 1) % 250 == 0:
            print(f"step {k + 1}: loss {np.mean(recent):.3f} ({time.time() - began:.0f}s)", flush=True)
            recent = []
    np.savez(Path(__file__).parent / "model.npz", chosen=chosen, mode=mode,
             **{f"w{n}": np.asarray(w) for n, (w, _) in enumerate(params)}, **{f"b{n}": np.asarray(b) for n, (_, b) in enumerate(params)})
    print(f"trained in {time.time() - began:.0f}s")
