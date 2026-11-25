"""
A contrastive goal-conditioned critic (Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL"):
scores how likely pressing button a in state s leads to state g in the near future, as the dot product of
an embedding of (s, a) and an embedding of g. Trained by classification on recordings: each (s, a) must
pick out its real future among the other futures in the batch. Acting picks the highest-scoring button,
rather than imitating whichever interchangeable button random play happened to press. Game-agnostic.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from controls import ACTIONS
from training import CPU, TRAINING_DTYPE, fit, host_params, init_mlp, mlp, sample_on_device

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
def _button_scores(params, observation, goal):
    observations = jnp.repeat(observation[None], len(ACTIONS), 0)
    return embed_state_action(params, observations, jnp.arange(len(ACTIONS))) @ embed_goal(params, goal[None])[0]


@dataclass
class Critic:
    """Acts on the CPU, like GoalPolicy."""
    params: dict
    byte_indices: np.ndarray

    def __post_init__(self):
        self.params = jax.device_put(self.params, CPU)

    def host_params(self):
        return host_params(self.params)

    def act(self, memory, goal_memory, rng, temperature=0.0):
        """The button whose predicted future best matches the goal. Temperature > 0 samples instead."""
        with jax.default_device(CPU):
            scores = np.asarray(_button_scores(self.params, jnp.asarray(memory[self.byte_indices]),
                                               jnp.asarray(goal_memory[self.byte_indices])))
        if temperature == 0:
            return int(scores.argmax())
        probabilities = np.exp((scores - scores.max()) / temperature)
        return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))


def train(experience, key, updates=20000, batch=1024, log=print):
    width = len(experience.byte_indices)
    state_key, goal_key, key = jax.random.split(key, 3)
    params = {"state_action": init_mlp(state_key, (width + len(ACTIONS), *HIDDEN, EMBEDDING)),
              "goal": init_mlp(goal_key, (width, *HIDDEN, EMBEDDING))}
    return Critic(fit(loss, params, experience, key, updates, batch, sampler=sample, log=log), experience.byte_indices)
