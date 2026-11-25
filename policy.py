"""
A goal-reaching player trained with hindsight: from (current observation, goal observation) predict the
button pressed. Any recorded step is a correct example of how to reach wherever that recording actually
went a few steps later, so random-play recordings become supervised data. Game-agnostic.

Its ceiling: only ~7.5 of the 25 button combos differ in effect at a given moment, so "which button did
random play press?" is mostly unanswerable, and imitating it gives a blurry player (see critic.py).
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from controls import ACTIONS
from training import TRAINING_DTYPE, CpuModel, device_data, fit, init_mlp, mlp, sample_on_device

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


class GoalPolicy(CpuModel):
    """policy(memories, goal_memories) -> [rows, buttons] logits."""
    evaluate = staticmethod(jax.jit(logits))


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


def value_loss(params, observations, goals, targets):
    """Regression onto a teacher's score for every button."""
    return jnp.mean(jnp.square(logits(params, observations, goals, TRAINING_DTYPE) - targets))


def sample_aimed(key, data, size):
    """(observation, the goal its recording aimed for, the teacher's target there) from aimed recordings."""
    observations, labels, observation_offsets, label_offsets, goals = data
    steps = jax.random.randint(key, (size,), 0, len(labels))
    recording = jnp.searchsorted(label_offsets, steps, side="right") - 1
    return observations[observation_offsets[recording] + steps - label_offsets[recording]], goals[recording], labels[steps]


def train_aimed(experience, targets, goals, key, updates=5000, batch=512, loss=loss, log=print):
    """
    Imitates a teacher: experience replays the teacher's trips (buttons as pressed, sticky noise included);
    targets holds, for each step, the button the teacher picked (with loss) or its score for every button
    (with value_loss); goals [recordings, memory bytes] is what each trip aimed for.
    """
    init_key, key = jax.random.split(key)
    params = init_mlp(init_key, (3 * len(experience.byte_indices), *HIDDEN, len(ACTIONS)))
    observations, _, observation_offsets, action_offsets = device_data(experience)
    data = (observations, jnp.asarray(targets), observation_offsets, action_offsets,
            jnp.asarray(goals[:, experience.byte_indices]))
    return GoalPolicy(fit(loss, params, data, key, updates, batch, sampler=sample_aimed, log=log),
                      experience.byte_indices)


def train(experience, key, updates=5000, batch=512, log=print):
    init_key, key = jax.random.split(key)
    params = init_mlp(init_key, (3 * len(experience.byte_indices), *HIDDEN, len(ACTIONS)))
    return GoalPolicy(fit(loss, params, device_data(experience), key, updates, batch, log=log), experience.byte_indices)
