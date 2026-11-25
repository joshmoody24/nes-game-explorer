"""
Training data from recorded segments. Emulation is deterministic, so a recording (start state + button
indices) is replayed to regenerate the memory at every step; nothing per-frame is stored on disk.
Game-agnostic: an observation is the memory bytes that vary across recordings, nothing named.
"""

import pickle
import zlib
from dataclasses import dataclass

import numpy as np

from controls import ACTIONS, FRAMES_PER_STEP


@dataclass
class Experience:
    """Recordings flattened: recording r has observations[observation_offsets[r]:observation_offsets[r + 1]]
    (one more than its actions) and actions[action_offsets[r]:action_offsets[r + 1]]."""
    byte_indices: np.ndarray  # which memory bytes an observation holds
    observations: np.ndarray  # [rows, len(byte_indices)] uint8
    actions: np.ndarray  # [steps] uint8, indices into controls.ACTIONS
    observation_offsets: np.ndarray
    action_offsets: np.ndarray
    start_states: list  # save state each recording starts from, for evaluation

    def recording(self, r):
        return (self.observations[self.observation_offsets[r]:self.observation_offsets[r + 1]],
                self.actions[self.action_offsets[r]:self.action_offsets[r + 1]])


def replay_memories(emulator, state, actions):
    """[len(actions) + 1, memory bytes]: memory before any button and after each step."""
    emulator.load(state)
    memories = [emulator.memory()]
    for action in actions:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        memories.append(emulator.memory())
    return np.stack(memories)


def valid_recordings(run_dir):
    """(start state, actions) for every recording whose start state still matches its checksum."""
    states = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))["states"]
    # Read each array once: indexing an npz file re-loads the whole array, and slices keep it alive.
    with np.load(run_dir / "trajectories.npz") as recorded:
        starts, checksums = recorded["start"], recorded["start_checksum"]
        actions, offsets = recorded["actions"], recorded["offsets"]
    state_checksums = [zlib.crc32(state) for state in states]
    return [(states[start], actions[offsets[i]:offsets[i + 1]])
            for i, (start, checksum) in enumerate(zip(starts, checksums))
            if state_checksums[start] == checksum and offsets[i + 1] > offsets[i]]


INPUT_ECHO_ACCURACY = 0.9


def input_echo_bytes(replayed, recordings):
    """
    Bytes whose value right after a step nearly determines the button pressed in it: the game's copy of
    the controller. A goal containing them tells a learner "press what was pressed then" instead of how to
    get there, so they're left out of observations.
    """
    after = np.concatenate([memories[1:] for memories in replayed])
    actions = np.concatenate([np.asarray(a, np.int64) for _, a in recordings])
    accuracy = np.array([np.bincount(after[:, byte].astype(np.int64) * len(ACTIONS) + actions,
                                     minlength=256 * len(ACTIONS)).reshape(256, len(ACTIONS)).max(1).sum()
                         for byte in range(after.shape[1])]) / len(actions)
    return np.flatnonzero(accuracy >= INPUT_ECHO_ACCURACY)


def build(emulator, recordings, byte_indices=None):
    """Replays recordings into an Experience. byte_indices defaults to bytes that vary within them, minus
    bytes that just echo the buttons."""
    replayed = [replay_memories(emulator, state, actions) for state, actions in recordings]
    if byte_indices is None:
        sample = replayed[:500]
        stacked = np.concatenate(sample)
        varying = np.flatnonzero((stacked != stacked[0]).any(0))
        byte_indices = np.setdiff1d(varying, input_echo_bytes(sample, recordings[:500]))
    observations = np.concatenate([memories[:, byte_indices] for memories in replayed])
    observation_offsets = np.cumsum([0] + [len(m) for m in replayed])
    action_offsets = np.cumsum([0] + [len(actions) for _, actions in recordings])
    return Experience(byte_indices, observations, np.concatenate([a for _, a in recordings]),
                      observation_offsets, action_offsets, [state for state, _ in recordings])
