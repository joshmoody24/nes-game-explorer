"""
Do pixels make learning to play faster? Trains the same small network on RAM, on a tiny grayscale screenshot,
and on both, to predict two things that matter for play: taking damage within DAMAGE_HORIZON steps, and changing
room within ROOM_HORIZON steps. Learning curves (held-out AUC vs number of training recordings) show whether
pixels reach the same accuracy with less data. Labels read Metroid bytes only to judge.

Usage: python -m checks.pixel_learning_check <run>[+<run>...] [recordings]
"""

import os
import random
import sys
from multiprocessing import get_context
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from checks.drive_learning_check import auc
from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from experience import choose_bytes, valid_recordings
from explorers import ROM
from metroid import MAP_X, MAP_Y, area_of, energy_of
from training import init_mlp, mlp

CONV = os.environ.get("CONV") == "1"


def emulator_frame_shape():
    return 224, 240  # QuickNES crops 8 pixels off each edge

DAMAGE_HORIZON, ROOM_HORIZON = 10, 50
SHRINK = int(os.environ.get("SHRINK", 8))


def _replay(task):
    recordings, byte_indices = task
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    out = []
    for start, actions in recordings:
        emulator.load(start)
        ram, pixels, energy, room = [], [], [], []
        for a in actions:
            emulator.run(ACTIONS[a], FRAMES_PER_STEP - 1)
            gray = emulator.frame(ACTIONS[a]).mean(2)
            h, w = gray.shape
            pixels.append(gray[:h - h % SHRINK, :w - w % SHRINK]
                          .reshape(h // SHRINK, SHRINK, w // SHRINK, SHRINK).mean((1, 3)).astype(np.uint8).ravel())
            memory = emulator.memory()
            ram.append(memory[byte_indices])
            energy.append(energy_of(memory))
            room.append((area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X])))
        energy = np.array(energy)
        hurt = np.array([energy[t + 1:t + 1 + DAMAGE_HORIZON].min(initial=energy[t]) < energy[t]
                         for t in range(len(energy))])
        moved = np.array([any(r != room[t] for r in room[t + 1:t + 1 + ROOM_HORIZON]) for t in range(len(room))])
        out.append((np.stack(ram), np.stack(pixels), hurt, moved))
    return out


def conv_features(params, pixels, shape):
    """Two strided 3x3 convolutions (reused across the screen), flattened."""
    h = pixels.reshape(-1, 1, *shape)
    for kernel in params:
        h = jax.nn.relu(jax.lax.conv_general_dilated(h, kernel, (2, 2), "SAME"))
    return h.reshape(len(pixels), -1)


def train(x, y, key, steps=1500, batch=512, pixel_shape=None, ram_width=0):
    """MLP on x; with pixel_shape, the columns after ram_width go through conv_features first."""
    if pixel_shape:
        k1, k2, key = jax.random.split(key, 3)
        kernels = [jax.random.normal(k1, (16, 1, 3, 3)) * 0.3, jax.random.normal(k2, (16, 16, 3, 3)) * 0.1]
        width = ram_width + 16 * (-(-pixel_shape[0] // 4)) * (-(-pixel_shape[1] // 4))
        params = (kernels, init_mlp(key, (width, 256, 64, 1)))
        net = lambda p, v: mlp(p[1], jnp.concatenate([v[:, :ram_width], conv_features(p[0], v[:, ram_width:], pixel_shape)], 1))
    else:
        params = init_mlp(key, (x.shape[1], 256, 64, 1))
        net = mlp
    return _fit(net, params, x, y, key, steps, batch)


def _fit(net, params, x, y, key, steps, batch, chunk=4096):
    optimizer = optax.adam(1e-3)
    state = optimizer.init(params)
    x, y = jnp.asarray(x), jnp.asarray(y, jnp.float32)  # arguments, not captured: captured data compiles in

    @jax.jit
    def step(params, state, key, x, y):
        rows = jax.random.randint(key, (batch,), 0, len(y))
        loss = lambda p: optax.sigmoid_binary_cross_entropy(net(p, x[rows] / 255.0)[:, 0], y[rows]).mean()
        grads = jax.grad(loss)(params)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state

    for k in jax.random.split(key, steps):
        params, state = step(params, state, k, x, y)
    score = jax.jit(lambda p, v: net(p, v / 255.0)[:, 0])
    return lambda inputs: np.concatenate([np.asarray(score(params, jnp.asarray(inputs[i:i + chunk], jnp.float32)))
                                          for i in range(0, len(inputs), chunk)])


def main():
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in sys.argv[1].split("+") for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    recordings = recordings[:count]
    byte_indices = choose_bytes(emulator, recordings[:500])
    chunks = [(recordings[k::4], byte_indices) for k in range(4)]
    with get_context("spawn").Pool(4) as pool:
        replayed = [r for part in pool.map(_replay, chunks) for r in part]
    random.shuffle(replayed)
    test, training = replayed[:len(replayed) // 5], replayed[len(replayed) // 5:]
    join = lambda part, k: np.concatenate([r[k] for r in part])
    inputs = {"RAM": 0, "pixels": 1, "both": None}
    get = lambda part, name: (np.concatenate([join(part, 0), join(part, 1)], 1) if inputs[name] is None
                              else join(part, inputs[name]))
    print(f"{len(replayed)} recordings, {len(join(replayed, 2)):,} steps; RAM {len(byte_indices)} bytes, "
          f"pixels {join(replayed, 1).shape[1]}; damage soon {join(replayed, 2).mean():.1%}, "
          f"new room soon {join(replayed, 3).mean():.1%}", flush=True)
    height, width = emulator_frame_shape()
    pixel_shape = (height // SHRINK, width // SHRINK)
    for target, k in [("damage soon", 2), ("new room soon", 3)]:
        for fraction in [0.02, 0.1, 0.3, 1.0]:
            part = training[:max(2, int(len(training) * fraction))]
            cells = []
            for name in inputs:
                conv = CONV and name != "RAM"
                predict = train(get(part, name), join(part, k), jax.random.key(0),
                                pixel_shape=pixel_shape if conv else None,
                                ram_width=len(byte_indices) if name == "both" else 0)
                cells.append(f"{name} {auc(predict(get(test, name)), join(test, k)):.3f}")
            print(f"{target}, {len(part)} training recordings: " + ", ".join(cells), flush=True)


if __name__ == "__main__":
    main()
