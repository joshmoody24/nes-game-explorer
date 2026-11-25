"""
A goal-reaching player trained with hindsight: from (current observation, goal observation) predict the
button pressed. Any recorded step is a correct example of how to reach wherever that recording actually
went a few steps later, so random-play recordings become supervised data. Game-agnostic.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import device
from controls import ACTIONS

MAX_GAP = 32  # goals are 1..MAX_GAP steps after the observation
HIDDEN = (512, 256)


def init_params(key, inputs, outputs):
    sizes = (inputs, *HIDDEN, outputs)
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (a, b)) / jnp.sqrt(a), "b": jnp.zeros(b)}
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


# Matrix multiplies run in `dtype` and accumulate in float32; parameters stay float32. bfloat16 is
# ~1.5x faster to train on the GPU at no loss in quality, but CPUs emulate it slowly, so acting uses float32.
TRAINING_DTYPE = jnp.bfloat16


def features(observations, goals, dtype):
    """
    Observation and goal as floats in [0, 1], plus whether each byte already matches the goal. (A
    goal-minus-observation input would add nothing, since the first layer is linear; "matches" isn't
    linear, and many bytes are IDs, where 0x37 and 0x38 mean unrelated things.)
    """
    matches = observations == goals
    return jnp.concatenate([observations.astype(dtype) / 255, goals.astype(dtype) / 255, matches.astype(dtype)], -1)


def dense(x, layer, dtype):
    return jnp.dot(x, layer["w"].astype(dtype), preferred_element_type=jnp.float32) + layer["b"]


def logits(params, observations, goals, dtype=jnp.float32):
    x = features(observations, goals, dtype)
    for layer in params[:-1]:
        x = jax.nn.relu(dense(x, layer, dtype)).astype(dtype)
    return dense(x, params[-1], dtype)


def loss(params, observations, goals, actions, dtype=jnp.float32):
    return optax.softmax_cross_entropy_with_integer_labels(logits(params, observations, goals, dtype), actions).mean()


def sample_on_device(key, data, size):
    """Random (observation, goal 1..MAX_GAP steps later in the same recording, action) triples, drawn on
    the accelerator from data already there (see device_data)."""
    observations, actions, observation_offsets, action_offsets = data
    step_key, gap_key = jax.random.split(key)
    steps = jax.random.randint(step_key, (size,), 0, len(actions))
    recording = jnp.searchsorted(action_offsets, steps, side="right") - 1
    t = steps - action_offsets[recording]
    remaining = action_offsets[recording + 1] - action_offsets[recording] - t
    gap = 1 + (jax.random.uniform(gap_key, (size,)) * jnp.minimum(MAX_GAP, remaining)).astype(jnp.int32)
    base = observation_offsets[recording] + t
    return observations[base], observations[base + gap], actions[steps]


OPTIMIZER = optax.adamw(3e-4, weight_decay=1e-4)


@partial(jax.jit, static_argnames=["updates", "batch"], donate_argnums=(0, 1))
def _train_chunk(params, optimizer_state, key, data, updates, batch):
    """Many updates in one compiled call, so the accelerator never waits on Python."""
    def update(carry, key):
        params, optimizer_state = carry
        value, grads = jax.value_and_grad(loss)(params, *sample_on_device(key, data, batch), TRAINING_DTYPE)
        changes, optimizer_state = OPTIMIZER.update(grads, optimizer_state, params)
        return (optax.apply_updates(params, changes), optimizer_state), value

    (params, optimizer_state), values = jax.lax.scan(update, (params, optimizer_state), jax.random.split(key, updates))
    return params, optimizer_state, values



CPU = jax.devices("cpu")[0]


@dataclass
class GoalPolicy:
    """Acts on the CPU: one decision at a time is too small for a GPU, where launch overhead dominates."""
    params: list
    byte_indices: np.ndarray

    def __post_init__(self):
        self.params = jax.device_put(self.params, CPU)

    def host_params(self):
        """Parameters as plain NumPy arrays, for sending to other processes."""
        return jax.tree.map(np.asarray, self.params)

    def action_logits(self, observations, goals):
        with jax.default_device(CPU):
            return np.asarray(_logits(self.params, jnp.asarray(observations), jnp.asarray(goals)))

    def act(self, memory, goal_memory, rng, temperature=1.0):
        """A button index for the full-memory current state and goal. Temperature 0 picks the most likely."""
        scores = self.action_logits(memory[self.byte_indices][None], goal_memory[self.byte_indices][None])[0]
        if temperature == 0:
            return int(scores.argmax())
        probabilities = np.exp((scores - scores.max()) / temperature)
        return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))


_logits = jax.jit(logits)


def device_data(experience):
    """The arrays training samples from, on the accelerator."""
    return (device.to_device(experience.observations), jnp.asarray(experience.actions),
            jnp.asarray(experience.observation_offsets), jnp.asarray(experience.action_offsets))


@partial(jax.jit, static_argnames=["size", "shuffle_goals"])
def _sampled_loss(params, key, data, size, shuffle_goals):
    sample_key, shuffle_key = jax.random.split(key)
    observations, goals, actions = sample_on_device(sample_key, data, size)
    if shuffle_goals:
        goals = jax.random.permutation(shuffle_key, goals)
    return loss(params, observations, goals, actions)


def mean_loss(params, experience, key, shuffle_goals=False, batches=5, size=2048):
    """Average loss over random batches. shuffle_goals pairs each observation with someone else's goal."""
    data = device_data(experience)
    return float(np.mean([_sampled_loss(params, k, data, size, shuffle_goals) for k in jax.random.split(key, batches)]))


def train(experience, key, updates=5000, batch=512, chunk=1000, log=print):
    """Trains on the default device (the GPU when there is one), with the whole dataset copied there once."""
    init_key, key = jax.random.split(key)
    params = init_params(init_key, 3 * len(experience.byte_indices), len(ACTIONS))
    optimizer_state = OPTIMIZER.init(params)
    data = device_data(experience)
    for done in range(0, updates, chunk):
        key, chunk_key = jax.random.split(key)
        params, optimizer_state, values = _train_chunk(params, optimizer_state, chunk_key, data,
                                                        min(chunk, updates - done), batch)
        log(f"update {done + len(values)}: loss {float(values[-1]):.3f}")
    return GoalPolicy(params, experience.byte_indices)
