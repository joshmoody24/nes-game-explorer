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

import policy
from controls import ACTIONS
from policy import features
from training import TRAINING_DTYPE, CpuModel, device_data, fit, init_mlp, mlp, sample_on_device

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
    return Distance(fit(loss, params, device_data(experience), key, updates, batch, sampler=sample_pairs, log=log),
                    experience.byte_indices)


# Advantage-weighted imitation (Peng et al., "Advantage-Weighted Regression"): hindsight examples count
# more when their button closed more of the distance to the goal than the batch usually does, so the player
# learns what works best rather than whatever was done.
ADVANTAGE_TEMPERATURE = 1.0
MOST_WEIGHT = 20.0


def sample_weighted(key, data, size):
    *experience_data, judge_params = data
    observations, goals, actions, nexts = sample_on_device(key, experience_data, size, with_next=True)
    progress = expected_log_steps(judge_params, observations, goals) - expected_log_steps(judge_params, nexts, goals)
    advantage = (progress - progress.mean()) / (progress.std() + 1e-6)
    weights = jnp.minimum(jnp.exp(advantage / ADVANTAGE_TEMPERATURE), MOST_WEIGHT)
    return observations, goals, actions, jax.lax.stop_gradient(weights / weights.mean())


def weighted_loss(params, observations, goals, actions, weights):
    return jnp.mean(weights * optax.softmax_cross_entropy_with_integer_labels(
        policy.logits(params, observations, goals, TRAINING_DTYPE), actions))


def train_weighted_player(experience, judge, key, updates=30000, batch=512, log=print):
    """A goal-reaching player (policy.GoalPolicy) trained with advantage weights from this distance."""
    init_key, key = jax.random.split(key)
    params = init_mlp(init_key, (3 * len(experience.byte_indices), *policy.HIDDEN, len(ACTIONS)))
    data = (*device_data(experience), jax.device_put(judge.host_params()))
    return policy.GoalPolicy(fit(weighted_loss, params, data, key, updates, batch, sampler=sample_weighted, log=log),
                             experience.byte_indices)
