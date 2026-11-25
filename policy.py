"""
A goal-reaching player trained with hindsight: from (current observation, goal observation) predict the
button pressed. Any recorded step is a correct example of how to reach wherever that recording actually
went a few steps later, so random-play recordings become supervised data. Game-agnostic.

Its ceiling: only ~7.5 of the 25 button combos differ in effect at a given moment, so "which button did
random play press?" is mostly unanswerable, and imitating it gives a blurry player (see critic.py).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from controls import ACTIONS
from training import CPU, TRAINING_DTYPE, device_data, fit, host_params, init_mlp, mlp, sample_on_device

HIDDEN = (512, 256)


def features(observations, goals):
    """
    Observation and goal as floats in [0, 1], plus whether each byte already matches the goal. (A
    goal-minus-observation input would add nothing, since the first layer is linear; "matches" isn't
    linear, and many bytes are IDs, where 0x37 and 0x38 mean unrelated things.)
    """
    return jnp.concatenate([observations / 255, goals / 255, observations == goals], -1)


def logits(params, observations, goals, dtype=jnp.float32):
    return mlp(params, features(observations, goals), dtype)


def loss(params, observations, goals, actions):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits(params, observations, goals, TRAINING_DTYPE), actions).mean()


_logits = jax.jit(logits)


@dataclass
class GoalPolicy:
    """Acts on the CPU: one decision at a time is too small for a GPU, where launch overhead dominates."""
    params: list
    byte_indices: np.ndarray

    def __post_init__(self):
        self.params = jax.device_put(self.params, CPU)

    def host_params(self):
        return host_params(self.params)

    def act(self, memory, goal_memory, rng, temperature=1.0):
        """A button index for the full-memory current state and goal. Temperature 0 picks the most likely."""
        with jax.default_device(CPU):
            scores = np.asarray(_logits(self.params, jnp.asarray(memory[self.byte_indices][None]),
                                        jnp.asarray(goal_memory[self.byte_indices][None])))[0]
        if temperature == 0:
            return int(scores.argmax())
        probabilities = np.exp((scores - scores.max()) / temperature)
        return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))


@partial(jax.jit, static_argnames=["size", "shuffle_goals"])
def _sampled_loss(params, key, data, size, shuffle_goals):
    sample_key, shuffle_key = jax.random.split(key)
    observations, goals, actions = sample_on_device(sample_key, data, size)
    if shuffle_goals:
        goals = jax.random.permutation(shuffle_key, goals)
    return optax.softmax_cross_entropy_with_integer_labels(logits(params, observations, goals), actions).mean()


def mean_loss(params, experience, key, shuffle_goals=False, batches=5, size=2048):
    """Average loss over random batches. shuffle_goals pairs each observation with someone else's goal."""
    data = device_data(experience)
    return float(np.mean([_sampled_loss(params, k, data, size, shuffle_goals) for k in jax.random.split(key, batches)]))


def train(experience, key, updates=5000, batch=512, log=print):
    init_key, key = jax.random.split(key)
    params = init_mlp(init_key, (3 * len(experience.byte_indices), *HIDDEN, len(ACTIONS)))
    return GoalPolicy(fit(loss, params, experience, key, updates, batch, log=log), experience.byte_indices)
