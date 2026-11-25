"""
A contrastive goal-conditioned critic (Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL"):
scores how likely pressing button a in state s leads to state g in the near future, as the dot product of
an embedding of (s, a) and an embedding of g. Trained by classification on recordings: each (s, a) must
pick out its real future among the other futures in the batch. Acting picks the highest-scoring button,
rather than imitating whichever interchangeable button random play happened to press. Game-agnostic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from controls import ACTIONS
from training import TRAINING_DTYPE, CpuModel, device_data, fit, init_mlp, mlp, sample_on_device

HIDDEN = (512, 256)
EMBEDDING = 64
FUTURE_MEAN = 16  # goals are drawn geometrically: near futures likelier, mean ~16 steps


def embed_state_action(params, observations, actions, dtype=jnp.float32):
    inputs = jnp.concatenate([observations / 255, jax.nn.one_hot(actions, len(ACTIONS))], -1)
    return mlp(params["state_action"], inputs, dtype)


def embed_goal(params, goals, dtype=jnp.float32):
    return mlp(params["goal"], goals / 255, dtype)


def loss(params, observations, goals, actions):
    """InfoNCE: row i's real future is goal i; the batch's other goals are the negatives."""
    scores = embed_state_action(params, observations, actions, TRAINING_DTYPE) @ embed_goal(params, goals, TRAINING_DTYPE).T
    return optax.softmax_cross_entropy_with_integer_labels(scores, jnp.arange(len(scores))).mean()


def sample(key, data, size):
    return sample_on_device(key, data, size, max_gap=64, geometric_mean=FUTURE_MEAN)


@jax.jit
def button_scores(params, observations, goals):
    """[rows, buttons]: how well each button's predicted future matches the goal."""
    rows = len(observations)
    pairs = embed_state_action(params, jnp.repeat(observations, len(ACTIONS), 0),
                               jnp.tile(jnp.arange(len(ACTIONS)), rows))
    return jnp.einsum("rbe,re->rb", pairs.reshape(rows, len(ACTIONS), -1), embed_goal(params, goals))


class Critic(CpuModel):
    """critic(memories, goal_memories) -> [rows, buttons] scores."""
    evaluate = staticmethod(button_scores)


def train(experience, key, updates=20000, batch=1024, log=print):
    width = len(experience.byte_indices)
    state_key, goal_key, key = jax.random.split(key, 3)
    params = {"state_action": init_mlp(state_key, (width + len(ACTIONS), *HIDDEN, EMBEDDING)),
              "goal": init_mlp(goal_key, (width, *HIDDEN, EMBEDDING))}
    return Critic(fit(loss, params, device_data(experience), key, updates, batch, sampler=sample, log=log), experience.byte_indices)
