"""Filling big arrays on the accelerator piece by piece, so no side ever holds two full copies."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, donate_argnums=0)
def _write_rows(buffer, rows, start):
    return jax.lax.dynamic_update_slice(buffer, rows, (start,) + (0,) * (rows.ndim - 1))


def zeros(shape, dtype=jnp.uint8):
    return jnp.zeros(shape, dtype)


def write_rows(buffer, rows, start):
    """buffer with rows written from row start on, updated in place (the old buffer is consumed)."""
    return _write_rows(buffer, jnp.asarray(rows), start)


def to_device(array, rows_per_copy=250_000):
    """A host array copied to the accelerator in pieces. Arrays already there are returned as they are."""
    if isinstance(array, jax.Array):
        return array
    buffer = jnp.zeros(array.shape, array.dtype)
    for start in range(0, len(array), rows_per_copy):
        buffer = write_rows(buffer, array[start:start + rows_per_copy], start)
    return buffer
