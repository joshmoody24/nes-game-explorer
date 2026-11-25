"""
How well does the player predict the button pressed, as a function of how far ahead the goal is? At a gap
of 1 the button is nearly determined by what changed, so a sound pipeline should be close to perfect
there; a steep fall with the gap means the training target is mostly noise rather than a bug.

Usage: python -m checks.gap_accuracy_check <run name> [--train N] [--updates N]
"""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from emulator import Emulator
from experience import build, valid_recordings
from explorers import ROM
from policy import logits, train

GAPS = [1, 2, 4, 8, 16, 32]


def fixed_gap_batch(rng, experience, gap, size):
    """(observation, observation gap steps later, button pressed) from recordings long enough."""
    lengths = np.diff(experience.action_offsets)
    recording = rng.choice(np.flatnonzero(lengths >= gap), size)
    t = (rng.random(size) * (lengths[recording] - gap + 1)).astype(int)
    base = experience.observation_offsets[recording] + t
    return (experience.observations[base], experience.observations[base + gap],
            experience.actions[experience.action_offsets[recording] + t])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--updates", type=int, default=20000)
    args = parser.parse_args()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = valid_recordings(Path("runs") / args.run)
    np.random.default_rng(0).shuffle(recordings)
    experience = build(emulator, recordings[:args.train], on_device=True)
    policy = train(experience, jax.random.key(0), updates=args.updates, log=lambda _: None)
    held_out = build(emulator, recordings[args.train:args.train + 2000], experience.byte_indices)
    rng = np.random.default_rng(1)
    for gap in GAPS:
        observations, goals, actions = fixed_gap_batch(rng, held_out, gap, 8192)
        scores = np.asarray(logits(policy.params, jnp.asarray(observations), jnp.asarray(goals)))
        probabilities = np.exp(scores - scores.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        print(f"gap {gap:>2}: top button right {np.mean(scores.argmax(1) == actions):.0%}, "
              f"in top 3 {np.mean((np.argsort(-scores, 1)[:, :3] == actions[:, None]).any(1)):.0%}, "
              f"probability given to the true button {np.mean(probabilities[np.arange(len(actions)), actions]):.2f} "
              f"(guessing: {1 / scores.shape[1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
