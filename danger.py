"""
A learned danger score: from memory alone, the chance of taking damage within the next few steps under random
play. Trained on recordings (see checks.careful_check --save); evaluated with NumPy so exploration's main
process can score each kept visit cheaply.
"""

import pickle
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def load(path):
    params, byte_indices = pickle.loads(open(path, "rb").read())
    return [(np.asarray(layer["w"], np.float32), np.asarray(layer["b"], np.float32)) for layer in params], byte_indices


def score(path, memory):
    """Predicted chance of damage soon, from a full memory snapshot."""
    layers, byte_indices = load(path)
    x = np.asarray(memory)[byte_indices].astype(np.float32) / 255.0
    for k, (w, b) in enumerate(layers):
        x = x @ w + b
        if k < len(layers) - 1:
            x = np.maximum(x, 0)
    return float(1 / (1 + np.exp(-x[0])))
