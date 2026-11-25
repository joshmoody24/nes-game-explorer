"""
A learned distance between game states: how many steps apart are state s and goal g? Trained on pairs from
recordings (0-64 steps apart in one recording, plus pairs from different recordings labeled "far"), as a
classification over coarse step buckets. Game-agnostic: the same raw-memory inputs as the policy.
With emulator lookahead it judges which button gets closest to a goal (see trips._searched).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from policy import features
from training import TRAINING_DTYPE, CpuModel, fit, init_mlp, mlp

HIDDEN = (512, 256)
MAX_GAP = 64
# Bucket upper edges in steps; the last bucket is "far" (different recordings, or beyond MAX_GAP).
EDGES = np.array([0, 1, 3, 7, 15, 31, 64])
LOG_CENTERS = jnp.log1p(jnp.array([0, 1, 2.5, 5.5, 11.5, 23.5, 48, 128]))
FAR_SHARE = 0.25


def bucket(gaps):
    return jnp.searchsorted(jnp.asarray(EDGES), gaps)


def sample_pairs(key, data, size):
    """(observation, goal, bucket): goals 0..MAX_GAP steps ahead in the same recording, or from another one."""
    observations, actions, observation_offsets, action_offsets = data
    step_key, gap_key, far_key, other_key = jax.random.split(key, 4)
    steps = jax.random.randint(step_key, (size,), 0, len(actions))
    recording = jnp.searchsorted(action_offsets, steps, side="right") - 1
    t = steps - action_offsets[recording]
    remaining = jnp.minimum(MAX_GAP, action_offsets[recording + 1] - action_offsets[recording] - t)
    gap = (jax.random.uniform(gap_key, (size,)) * (remaining + 1)).astype(jnp.int32)
    base = observation_offsets[recording] + t
    far = jax.random.uniform(far_key, (size,)) < FAR_SHARE
    other = jax.random.randint(other_key, (size,), 0, len(observations))
    goals = jnp.where(far[:, None], observations[other], observations[base + gap])
    return observations[base], goals, jnp.where(far, len(EDGES), bucket(gap))


def logits(params, observations, goals, dtype=jnp.float32):
    return mlp(params, features(observations, goals), dtype)


def loss(params, observations, goals, buckets):
    return optax.softmax_cross_entropy_with_integer_labels(logits(params, observations, goals, TRAINING_DTYPE), buckets).mean()


@jax.jit
def expected_log_steps(params, observations, goals):
    return jax.nn.softmax(logits(params, observations, goals)) @ LOG_CENTERS


class Distance(CpuModel):
    """distance(memories, goal_memories) -> expected log(1 + steps) from each state to its goal."""
    evaluate = staticmethod(expected_log_steps)


def train(experience, key, updates=20000, batch=1024, log=print):
    init_key, key = jax.random.split(key)
    params = init_mlp(init_key, (3 * len(experience.byte_indices), *HIDDEN, len(EDGES) + 1))
    return Distance(fit(loss, params, experience, key, updates, batch, sampler=sample_pairs, log=log),
                    experience.byte_indices)
