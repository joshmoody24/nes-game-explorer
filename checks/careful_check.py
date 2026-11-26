"""
Can a small network learn "careful play" from recordings? Predicts, from memory and the button pressed,
whether energy drops (damage) and whether missiles drop (spending) within the next HORIZON steps, and
compares with the same model blind to the button: if the button adds little, a careful explorer can't steer
with it. Held-out recordings, AUC. Energy and missile bytes are named by hand here (labels only).

Usage: python -m checks.careful_check <run name>[+<run name>...] [recordings] [--save <path>]
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from controls import ACTIONS
from emulator import Emulator
from experience import build, choose_bytes, valid_recordings
from explorers import ROM
from metroid import CURRENT_MISSILES, at, energy_of
from training import init_mlp, mlp

HORIZON = 10


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    positives = labels.sum()
    return (ranks[labels].sum() - positives * (positives - 1) / 2) / (positives * (len(labels) - positives))


def examples(experience, byte_indices):
    """(inputs without the button, button, [damage, spend] labels) for every step with HORIZON steps after it."""
    column = {b: k for k, b in enumerate(byte_indices)}
    states, buttons, labels = [], [], []
    for r in range(len(experience.start_states)):
        observations, actions = experience.recording(r)
        if len(actions) <= HORIZON:
            continue
        energy = np.array([10 * (o[column[0x107]] & 0xF) + (o[column[0x106]] >> 4) for o in observations])
        missiles = observations[:, column[at(CURRENT_MISSILES)]].astype(int)
        n = len(actions) - HORIZON
        states.append(observations[:n])
        buttons.append(actions[:n])
        future = np.arange(n)[:, None] + np.arange(1, HORIZON + 1)
        labels.append(np.stack([(energy[future].min(1) < energy[:n]), (missiles[future].min(1) < missiles[:n])], 1))
    return np.concatenate(states), np.concatenate(buttons), np.concatenate(labels)


def train(states, buttons, labels, use_button, key, steps=4000, batch=2048, keep_params=False):
    width = states.shape[1] + len(ACTIONS)
    params = init_mlp(key, (width, 256, 128, 2))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    states, buttons, labels = jnp.asarray(states), jnp.asarray(buttons), jnp.asarray(labels, jnp.float32)

    def features(s, b):
        onehot = jax.nn.one_hot(b, len(ACTIONS)) * (1.0 if use_button else 0.0)
        return jnp.concatenate([s / 255.0, onehot], -1)

    @jax.jit
    def step(params, opt_state, key):
        idx = jax.random.randint(key, (batch,), 0, len(labels))
        loss = lambda p: optax.sigmoid_binary_cross_entropy(mlp(p, features(states[idx], buttons[idx])), labels[idx]).mean()
        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, value

    for k in jax.random.split(key, steps):
        params, opt_state, _ = step(params, opt_state, k)
    if keep_params:
        return params
    return lambda s, b: np.asarray(jax.nn.sigmoid(mlp(params, features(jnp.asarray(s), jnp.asarray(b)))))


def main():
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in sys.argv[1].split("+") for r in valid_recordings(Path("runs") / run)]
    np.random.default_rng(0).shuffle(recordings)
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    byte_indices = np.union1d(choose_bytes(emulator, recordings[:1000]), [0x106, 0x107, at(CURRENT_MISSILES)])
    split = int(count * 0.8)
    train_data = examples(build(emulator, recordings[:split], byte_indices), byte_indices)
    test_states, test_buttons, test_labels = examples(build(emulator, recordings[split:count], byte_indices), byte_indices)
    print(f"{len(train_data[2]):,} training steps; rates: damage {train_data[2][:, 0].mean():.1%}, "
          f"missile spending {train_data[2][:, 1].mean():.1%}", flush=True)
    if "--save" in sys.argv:
        path = sys.argv[sys.argv.index("--save") + 1]
        params = train(*train_data, False, jax.random.key(0), keep_params=True)
        two_heads = jax.tree.map(np.asarray, params)
        damage_only = two_heads[:-1] + [{"w": two_heads[-1]["w"][:, :1], "b": two_heads[-1]["b"][:1]}]
        # The button inputs are zeroed when blind to the button, so drop their rows from the first layer.
        damage_only[0] = {"w": damage_only[0]["w"][:len(byte_indices)], "b": damage_only[0]["b"]}
        Path(path).write_bytes(__import__("pickle").dumps((damage_only, byte_indices)))
        print(f"saved the damage model to {path}")
        return
    for use_button in [False, True]:
        predict = train(*train_data, use_button, jax.random.key(0))
        scores = np.concatenate([predict(test_states[i:i + 8192], test_buttons[i:i + 8192])
                                 for i in range(0, len(test_states), 8192)])
        print(f"{'state + button' if use_button else 'state only':>15}: held-out AUC damage "
              f"{auc(scores[:, 0], test_labels[:, 0]):.3f}, missile spending {auc(scores[:, 1], test_labels[:, 1]):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
