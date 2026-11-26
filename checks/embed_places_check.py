"""
Generic places from a learned representation: trains an encoder from RAM so that states a few steps apart in a
recording embed close together and states from different recordings don't (temporal contrastive learning),
then groups library states with k-means and compares the groups with hand-made rooms: purity (a group's main
room is the state's room) and groups per room. Hand-made rooms only judge the result.

Usage: python -m checks.embed_places_check <run>[+<run>...] [states]
"""

import os
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import library
from emulator import Emulator
from experience import build, choose_bytes, valid_recordings
from explorers import ROM
from training import init_mlp, mlp

EMBEDDING, MAX_GAP = 32, 8


def train_encoder(experience, key, steps=3000, batch=1024):
    observations = jnp.asarray(experience.observations)
    offsets = jnp.asarray(experience.observation_offsets)
    lengths = jnp.asarray(np.diff(experience.observation_offsets))
    params = init_mlp(key, (observations.shape[1], 256, 128, EMBEDDING))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    embed = lambda p, o: mlp(p, o / 255.0)

    @jax.jit
    def step(params, opt_state, key):
        k1, k2, k3 = jax.random.split(key, 3)
        rec = jax.random.randint(k1, (batch,), 0, len(lengths))
        t = (jax.random.uniform(k2, (batch,)) * lengths[rec]).astype(jnp.int32)
        gap = 1 + (jax.random.uniform(k3, (batch,)) * MAX_GAP).astype(jnp.int32)
        anchor = observations[offsets[rec] + t]
        positive = observations[offsets[rec] + jnp.minimum(t + gap, lengths[rec] - 1)]

        def loss(p):
            a, b = embed(p, anchor), embed(p, positive)
            a = a / (jnp.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
            b = b / (jnp.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
            logits = a @ b.T / 0.1
            return optax.softmax_cross_entropy_with_integer_labels(logits, jnp.arange(batch)).mean()

        value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, value

    for k in jax.random.split(key, steps):
        params, opt_state, value = step(params, opt_state, k)
    print(f"encoder trained, final contrastive loss {float(value):.3f}", flush=True)
    return lambda o: np.asarray(embed(params, jnp.asarray(o)))


def kmeans(x, k, rng, iterations=25):
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iterations):
        labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), 1)
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centers[c] = members.mean(0)
    return labels


def main():
    runs = sys.argv[1].split("+")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    recordings = [r for run in runs for r in valid_recordings(Path("runs") / run)]
    random.seed(0)
    random.shuffle(recordings)
    byte_indices = choose_bytes(emulator, recordings[:1000])
    experience = build(emulator, recordings[:6000], byte_indices)
    encode = train_encoder(experience, jax.random.key(0))
    picks = random.sample(library.entries(), count)
    by_run = defaultdict(list)
    for e in picks:
        by_run[e["run"]].append(e)
    memories, rooms = [], []
    for run, group in by_run.items():
        saved = pickle.loads(zlib.decompress((Path(run) / "checkpoint.pkl.z").read_bytes()))
        for e in group:
            memories.append(emulator.memory(saved["states"][e["slot"]])[byte_indices])
            rooms.append(e["room"])
    z = encode(np.stack(memories))
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
    rng = np.random.default_rng(0)
    for k in [130, 400, 1000]:
        labels = kmeans(z, k, rng)
        rooms_of, places_of_room = defaultdict(Counter), defaultdict(set)
        for label, room in zip(labels, rooms):
            rooms_of[label][room] += 1
            places_of_room[room].add(label)
        purity = sum(c.most_common(1)[0][1] for c in rooms_of.values()) / len(labels)
        print(f"k={k}: {len(rooms_of)} learned places for {len(places_of_room)} rooms; purity {purity:.0%}; "
              f"median learned places per room {np.median([len(p) for p in places_of_room.values()]):.0f}", flush=True)


if __name__ == "__main__":
    main()
