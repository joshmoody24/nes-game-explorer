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

from controls import ACTIONS

MAX_GAP = 32  # goals are 1..MAX_GAP steps after the observation
HIDDEN = (512, 256)


def init_params(key, inputs, outputs):
    sizes = (inputs, *HIDDEN, outputs)
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (a, b)) / jnp.sqrt(a), "b": jnp.zeros(b)}
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def features(observations, goals):
    """Observation, goal, and their difference, as floats in [-1, 1]."""
    current, target = observations.astype(jnp.float32) / 255, goals.astype(jnp.float32) / 255
    return jnp.concatenate([current, target, target - current], -1)


def logits(params, observations, goals):
    x = features(observations, goals)
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def loss(params, observations, goals, actions):
    return optax.softmax_cross_entropy_with_integer_labels(logits(params, observations, goals), actions).mean()


def sample_batch(rng, experience, size):
    """Random (observation, goal 1..MAX_GAP steps later in the same recording, action) triples."""
    steps = rng.integers(0, len(experience.actions), size)
    recording = np.searchsorted(experience.action_offsets, steps, side="right") - 1
    t = steps - experience.action_offsets[recording]
    remaining = experience.action_offsets[recording + 1] - experience.action_offsets[recording] - t
    gap = 1 + (rng.random(size) * np.minimum(MAX_GAP, remaining)).astype(int)
    base = experience.observation_offsets[recording] + t
    return experience.observations[base], experience.observations[base + gap], experience.actions[steps]


OPTIMIZER = optax.adamw(3e-4, weight_decay=1e-4)


@partial(jax.jit, donate_argnums=(0, 1))
def _update(params, optimizer_state, observations, goals, actions):
    value, grads = jax.value_and_grad(loss)(params, observations, goals, actions)
    updates, optimizer_state = OPTIMIZER.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), optimizer_state, value



@dataclass
class GoalPolicy:
    params: list
    byte_indices: np.ndarray

    def action_logits(self, observations, goals):
        return np.asarray(logits(self.params, jnp.asarray(observations), jnp.asarray(goals)))

    def act(self, memory, goal_memory, rng, temperature=1.0):
        """A button index for the full-memory current state and goal. Temperature 0 picks the most likely."""
        scores = self.action_logits(memory[self.byte_indices][None], goal_memory[self.byte_indices][None])[0]
        if temperature == 0:
            return int(scores.argmax())
        probabilities = np.exp((scores - scores.max()) / temperature)
        return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))


def train(experience, key, updates=5000, batch=512, log=print):
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    params = init_params(key, 3 * len(experience.byte_indices), len(ACTIONS))
    optimizer_state = OPTIMIZER.init(params)
    for update in range(updates):
        params, optimizer_state, value = _update(params, optimizer_state, *map(jnp.asarray, sample_batch(rng, experience, batch)))
        if update % 1000 == 0 or update == updates - 1:
            log(f"update {update}: loss {float(value):.3f}")
    return GoalPolicy(params, experience.byte_indices)
