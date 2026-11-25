"""
A learned sense of danger: from raw memory, predict whether random play from a state soon ends with my
buttons barely mattering (dead / game over). Deaths are common in random play, so unlike rare
discoveries this signal is dense enough to learn from. Game-agnostic: inputs are the memory bytes that
vary across states, scaled to [0, 1].
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from empowerment import NO_CONTROL_BITS, controllable_bits, random_play

HIDDEN = 128
TRAINING_STEPS = 2000


def lost_control(emulator, state, rng, steps=60):
    """One sample: does a random-play replay of this many steps end somewhere my buttons barely affect?"""
    random_play(emulator, state, rng, steps)
    return bool(controllable_bits(emulator, emulator.save(), rng, 2, 6).sum() < NO_CONTROL_BITS)


def init_params(key, inputs):
    first, second, third = jax.random.split(key, 3)

    def dense(key, fan_in, fan_out):
        return {"w": jax.random.normal(key, (fan_in, fan_out)) / jnp.sqrt(fan_in), "b": jnp.zeros(fan_out)}

    return [dense(first, inputs, HIDDEN), dense(second, HIDDEN, HIDDEN), dense(third, HIDDEN, 1)]


def logit(params, inputs):
    x = inputs
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return (x @ params[-1]["w"] + params[-1]["b"])[..., 0]


def loss(params, inputs, labels):
    return optax.sigmoid_binary_cross_entropy(logit(params, inputs), labels).mean()


@partial(jax.jit, static_argnames=["steps"])
def _fit(params, inputs, labels, steps):
    optimizer = optax.adamw(1e-3, weight_decay=1e-2)

    def step(carry, _):
        params, state = carry
        value, grads = jax.value_and_grad(loss)(params, inputs, labels)
        updates, state = optimizer.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), value

    (params, _), losses = jax.lax.scan(step, (params, optimizer.init(params)), None, length=steps)
    return params, losses


@dataclass
class DangerModel:
    params: list
    byte_indices: np.ndarray

    def danger(self, memories):
        inputs = jnp.asarray(memories[..., self.byte_indices], dtype=jnp.float32) / 255
        return np.asarray(jax.nn.sigmoid(logit(self.params, inputs)))


def fit(memories, labels, key, steps=TRAINING_STEPS):
    """memories: [states, bytes] uint8; labels: [states] share of replays that lost control."""
    byte_indices = np.flatnonzero((memories != memories[0]).any(0))
    inputs = jnp.asarray(memories[:, byte_indices], dtype=jnp.float32) / 255
    params, losses = _fit(init_params(key, len(byte_indices)), inputs, jnp.asarray(labels, jnp.float32), steps)
    return DangerModel(params, byte_indices), np.asarray(losses)
