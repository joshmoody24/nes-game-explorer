"""
Promise: how many new places a trip from a saved state is expected to find, predicted from raw
memory. Game-agnostic: the input is every memory byte that varies across the saved states, scaled to
[0, 1], and nothing is known about what any byte means.
"""

import pickle
import zlib
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

HIDDEN = 128
TRAINING_STEPS = 3000


@dataclass
class TripRecord:
    """Per saved place that was explored from: its memory, how many trips started there, and what they found."""
    memories: np.ndarray  # [places, memory bytes] uint8
    trips: np.ndarray  # [places]
    discoveries: np.ndarray  # [places]


def trip_record(run_dir, emulator):
    archive = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))
    segments = np.load(run_dir / "segments.npz")
    starts = np.unique(segments["start"])
    trips = np.bincount(segments["start"], minlength=len(archive["states"]))[starts]
    discoveries = np.bincount(segments["start"], weights=segments["new_places"],
                              minlength=len(archive["states"]))[starts]
    memories = np.stack([emulator.memory(archive["states"][i]) for i in starts])
    return TripRecord(memories, trips, discoveries)


def init_params(key, inputs):
    first, second, third = jax.random.split(key, 3)

    def dense(key, fan_in, fan_out):
        return {"w": jax.random.normal(key, (fan_in, fan_out)) / jnp.sqrt(fan_in), "b": jnp.zeros(fan_out)}

    return [dense(first, inputs, HIDDEN), dense(second, HIDDEN, HIDDEN), dense(third, HIDDEN, 1)]


def log_rate(params, inputs):
    """Log of expected discoveries per trip."""
    x = inputs
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return (x @ params[-1]["w"] + params[-1]["b"])[..., 0]


def poisson_loss(params, inputs, trips, discoveries):
    """Discoveries from a place ~ Poisson(trips x rate). Negative log-likelihood, dropping constants."""
    predicted = log_rate(params, inputs)
    return jnp.mean(trips * jnp.exp(predicted) - discoveries * predicted)


@dataclass
class PromiseModel:
    params: list
    byte_indices: np.ndarray  # which memory bytes the model reads

    def inputs(self, memories):
        return jnp.asarray(memories[..., self.byte_indices], dtype=jnp.float32) / 255

    def rate(self, memories):
        return np.asarray(jnp.exp(log_rate(self.params, self.inputs(memories))))


@partial(jax.jit, static_argnames=["steps"])
def _fit(params, inputs, trips, discoveries, steps):
    optimizer = optax.adam(1e-3)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(poisson_loss)(params, inputs, trips, discoveries)
        updates, state = optimizer.update(grads, state)
        return (optax.apply_updates(params, updates), state), loss

    (params, _), losses = jax.lax.scan(step, (params, optimizer.init(params)), None, length=steps)
    return params, losses


def fit(record, key, steps=TRAINING_STEPS):
    byte_indices = np.flatnonzero((record.memories != record.memories[0]).any(0))
    model = PromiseModel(None, byte_indices)
    inputs = model.inputs(record.memories)
    params, losses = _fit(init_params(key, len(byte_indices)), inputs, jnp.asarray(record.trips, jnp.float32),
                          jnp.asarray(record.discoveries, jnp.float32), steps)
    model.params = params
    return model, np.asarray(losses)


def deviance_explained(model, record):
    """Share of the Poisson deviance a constant rate leaves that the model removes (1 is perfect, 0 no better)."""
    def deviance(rate):
        expected = record.trips * rate
        observed = record.discoveries
        ratio = np.where(observed > 0, observed / np.maximum(expected, 1e-9), 1)
        return 2 * np.sum(np.where(observed > 0, observed * np.log(ratio), 0) - (observed - expected))

    constant = record.discoveries.sum() / record.trips.sum()
    return 1 - deviance(model.rate(record.memories)) / deviance(constant)


def subset(record, mask):
    return TripRecord(record.memories[mask], record.trips[mask], record.discoveries[mask])
