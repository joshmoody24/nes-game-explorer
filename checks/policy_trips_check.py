"""
Test A: can the hindsight goal-reaching player make short trips without save states? From points along
held-out recordings, the goal is where that recording was 10-30 steps later. Each trip gets 2x that many
steps, with sticky-action noise (25% of steps repeat the previous button instead). Places are used only
to judge success.

Usage: python -m checks.policy_trips_check <run name> [--train N] [--trials N] [--updates N] [--batch N]
"""

import argparse
import os
import time
from pathlib import Path

import jax
import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP
from emulator import Emulator
from experience import build, replay_memories, valid_recordings
from explorers import Settings, hand_made_place
from policy import train

STICKY = 0.25


def distance(place, goal_place):
    """Evaluation only: map cells apart, plus position blocks apart within a cell (8 blocks ~ one cell)."""
    return (abs(place[1] - goal_place[1]) + abs(place[2] - goal_place[2])
            + (abs(place[3] - goal_place[3]) + abs(place[4] - goal_place[4])) / 8)


def trip(emulator, start, goal_place, budget, choose, rng):
    """Runs choose(memory, rng) -> action for up to budget steps from start.
    Returns (reached goal_place, starting distance, closest distance)."""
    emulator.load(start)
    previous = 0
    settings = Settings()
    first = closest = distance(hand_made_place(emulator.memory(), settings), goal_place)
    for _ in range(budget):
        action = choose(emulator.memory(), rng)
        if rng.random() < STICKY:
            action = previous
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        previous = action
        place = hand_made_place(emulator.memory(), settings)
        closest = min(closest, distance(place, goal_place))
        if place == goal_place:
            return True, first, 0.0
    return False, first, closest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--train", type=int, default=4000, help="training recordings")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    run_dir = Path("runs") / args.run
    training_count, trials = args.train, args.trials
    emulator = Emulator(os.environ["QUICKNES_CORE"], "roms/metroid.nes")
    recordings = valid_recordings(run_dir)
    rng = np.random.default_rng(0)
    rng.shuffle(recordings)
    held_out = [r for r in recordings[training_count:] if len(r[1]) >= 40]
    started = time.perf_counter()
    experience = build(emulator, recordings[:training_count])
    built = time.perf_counter()
    print(f"{len(recordings)} usable recordings; training on {training_count} "
          f"({len(experience.actions)} steps, {len(experience.byte_indices)} varying bytes); "
          f"{len(held_out)} held out")
    policy = train(experience, jax.random.key(0), updates=args.updates, batch=args.batch)
    trained = time.perf_counter()
    from policy import loss, sample_batch
    import jax.numpy as jnp
    held_out_experience = build(emulator, held_out[:500], experience.byte_indices)
    batch_rng = np.random.default_rng(1)
    for name, data in [("training", experience), ("held-out", held_out_experience)]:
        values = [float(loss(policy.params, *map(jnp.asarray, sample_batch(batch_rng, data, 2048)))) for _ in range(5)]
        print(f"loss on {name} recordings: {np.mean(values):.3f} (3.22 = guessing among {len(ACTIONS)} buttons)")
        shuffled = []
        for _ in range(5):
            observations, goals, actions = sample_batch(batch_rng, data, 2048)
            shuffled.append(float(loss(policy.params, jnp.asarray(observations),
                                       jnp.asarray(goals[batch_rng.permutation(len(goals))]), jnp.asarray(actions))))
        print(f"  same, with each goal swapped for a random other one: {np.mean(shuffled):.3f}")
    print(f"timing: building data {built - started:.0f}s ({len(experience.actions) / (built - started):,.0f} steps/s), "
          f"training {trained - built:.0f}s ({args.updates / (trained - built):.0f} updates/s)")

    settings = Settings()
    outcomes = {name: [] for name in ["learned, sampled", "learned, temperature 0.3", "learned, most likely button",
                                       "random buttons", "replay the recorded buttons"]}
    decision_time, decisions = 0.0, 0
    for i in range(trials):
        state, actions = held_out[i % len(held_out)]
        start_step = int(rng.integers(0, len(actions) - 10))
        gap = int(rng.integers(10, min(30, len(actions) - start_step) + 1))
        memories = replay_memories(emulator, state, actions[:start_step + gap])
        goal_memory = memories[start_step + gap]
        goal_place = hand_made_place(goal_memory, settings)
        if hand_made_place(memories[start_step], settings) == goal_place:
            continue
        emulator.load(state)
        for action in actions[:start_step]:
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        start = emulator.save()
        recorded = list(actions[start_step:start_step + gap])
        def timed(temperature):
            def choose(memory, rng):
                nonlocal decision_time, decisions
                before = time.perf_counter()
                action = policy.act(memory, goal_memory, rng, temperature)
                decision_time += time.perf_counter() - before
                decisions += 1
                return action
            return choose

        players = {
            "learned, sampled": timed(1.0),
            "learned, temperature 0.3": timed(0.3),
            "learned, most likely button": timed(0),
            "random buttons": lambda memory, rng: int(rng.integers(len(ACTIONS))),
            "replay the recorded buttons": lambda memory, rng, steps=iter(recorded): next(steps, 0),
        }
        for name, choose in players.items():
            outcomes[name].append(trip(emulator, start, goal_place, 2 * gap, choose, np.random.default_rng([i, 7])))
    print(f"timing: one learned decision {decision_time / decisions * 1e6:.0f} us "
          f"(an emulator step of {FRAMES_PER_STEP} frames is ~80 us)")
    for name, results in outcomes.items():
        reached = np.array([r[0] for r in results])
        progress = np.array([(r[1] - r[2]) / r[1] for r in results if r[1] > 0])
        print(f"{name:>28}: reached the goal in {reached.mean():.0%} of {len(results)} trips; "
              f"closed {np.mean(progress):.0%} of the starting distance on average")


if __name__ == "__main__":
    main()
