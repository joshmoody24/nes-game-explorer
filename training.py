"""
Shared pieces for training small networks on recorded experience on the accelerator: MLPs, sampling
(observation, later observation, action) triples on the device, and a training loop that runs many
updates per compiled call so the accelerator never waits on Python.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import device

CPU = jax.devices("cpu")[0]
# Matrix multiplies run in this type for training and accumulate in float32; parameters stay float32.
# bfloat16 is ~1.5x faster on the GPU at no loss in quality, but CPUs emulate it slowly, so acting uses float32.
TRAINING_DTYPE = jnp.bfloat16
OPTIMIZER = optax.adamw(3e-4, weight_decay=1e-4)


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (a, b)) / jnp.sqrt(a), "b": jnp.zeros(b)}
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp(params, x, dtype=jnp.float32):
    """ReLU MLP whose matrix multiplies run in dtype, accumulating in float32."""
    x = x.astype(dtype)
    for i, layer in enumerate(params):
        x = jnp.dot(x, layer["w"].astype(dtype), preferred_element_type=jnp.float32) + layer["b"]
        if i < len(params) - 1:
            x = jax.nn.relu(x).astype(dtype)
    return x


def device_data(experience):
    """The arrays training samples from, on the accelerator."""
    return (device.to_device(experience.observations), jnp.asarray(experience.actions),
            jnp.asarray(experience.observation_offsets), jnp.asarray(experience.action_offsets))


def sample_on_device(key, data, size, max_gap=32, geometric_mean=None, with_next=False):
    """
    Random (observation, goal later in the same recording, action) triples, drawn on the accelerator.
    The goal is 1..max_gap steps ahead, uniformly, or geometrically with the given mean (near futures
    likelier), truncated at the recording's end. with_next adds the observation right after the action.
    """
    observations, actions, observation_offsets, action_offsets = data
    step_key, gap_key = jax.random.split(key)
    steps = jax.random.randint(step_key, (size,), 0, len(actions))
    recording = jnp.searchsorted(action_offsets, steps, side="right") - 1
    t = steps - action_offsets[recording]
    remaining = jnp.minimum(max_gap, action_offsets[recording + 1] - action_offsets[recording] - t)
    if geometric_mean is None:
        gap = 1 + (jax.random.uniform(gap_key, (size,)) * remaining).astype(jnp.int32)
    else:
        gap = jnp.minimum(jax.random.geometric(gap_key, 1 / geometric_mean, (size,)), remaining).astype(jnp.int32)
    base = observation_offsets[recording] + t
    if with_next:
        return observations[base], observations[base + gap], actions[steps], observations[base + 1]
    return observations[base], observations[base + gap], actions[steps]


@partial(jax.jit, static_argnames=["loss", "updates", "batch", "sampler"], donate_argnums=(1, 2))
def _train_chunk(loss, params, optimizer_state, key, data, updates, batch, sampler):
    def update(carry, key):
        params, optimizer_state = carry
        value, grads = jax.value_and_grad(loss)(params, *sampler(key, data, batch))
        changes, optimizer_state = OPTIMIZER.update(grads, optimizer_state, params)
        return (optax.apply_updates(params, changes), optimizer_state), value

    (params, optimizer_state), values = jax.lax.scan(update, (params, optimizer_state), jax.random.split(key, updates))
    return params, optimizer_state, values


def fit(loss, params, data, key, updates, batch, sampler=sample_on_device, chunk=1000, log=print):
    """
    Minimizes loss(params, observations, goals, labels) on batches drawn by sampler(key, data, batch), on
    the default device (the GPU when there is one), from data already there (e.g. device_data(experience)).
    loss and sampler must be module-level functions.
    """
    optimizer_state = OPTIMIZER.init(params)
    for done in range(0, updates, chunk):
        key, chunk_key = jax.random.split(key)
        params, optimizer_state, values = _train_chunk(loss, params, optimizer_state, chunk_key, data,
                                                        min(chunk, updates - done), batch, sampler)
        log(f"update {done + len(values)}: loss {float(values[-1]):.3f}")
    return params


@dataclass
class CpuModel:
    """
    A trained model evaluated on the CPU, in batches: model(memories, goal_memories) -> one result per row,
    from full-memory snapshots. Subclasses set `evaluate`, a jitted (params, observations, goals) -> results.
    The CPU, because decisions come a few at a time during play, too few for a GPU's launch overhead.
    """
    params: object
    byte_indices: np.ndarray

    def __post_init__(self):
        self.params = jax.device_put(self.params, CPU)

    def host_params(self):
        """Parameters as plain NumPy arrays, for sending to other processes."""
        return jax.tree.map(np.asarray, self.params)

    def __call__(self, memories, goal_memories):
        with jax.default_device(CPU):
            return np.asarray(self.evaluate(self.params, jnp.asarray(memories[:, self.byte_indices]),
                                            jnp.asarray(goal_memories[:, self.byte_indices])))
