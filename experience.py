"""
Training data from recorded segments. Emulation is deterministic, so a recording (start state + button
indices) is replayed to regenerate the memory at every step; nothing per-frame is stored on disk.
Game-agnostic: an observation is the memory bytes that vary across recordings, nothing named.
"""

import pickle
import zlib
from dataclasses import dataclass

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP


@dataclass
class Experience:
    """Recordings flattened: recording r has observations[observation_offsets[r]:observation_offsets[r + 1]]
    (one more than its actions) and actions[action_offsets[r]:action_offsets[r + 1]]."""
    byte_indices: np.ndarray  # which memory bytes an observation holds
    observations: np.ndarray  # [rows, len(byte_indices)] uint8, on the host or the accelerator
    actions: np.ndarray  # [steps] uint8, indices into controls.ACTIONS
    observation_offsets: np.ndarray
    action_offsets: np.ndarray
    start_states: list  # save state each recording starts from, for evaluation

    def recording(self, r):
        return (self.observations[self.observation_offsets[r]:self.observation_offsets[r + 1]],
                self.actions[self.action_offsets[r]:self.action_offsets[r + 1]])


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


def choose_bytes(emulator, recordings, sample=500):
    """Memory bytes that vary within a sample of recordings, minus bytes that just echo the buttons."""
    every_byte = build(emulator, recordings[:sample], np.arange(len(emulator.memory())))
    replayed = [every_byte.recording(r)[0] for r in range(len(recordings[:sample]))]
    stacked = every_byte.observations
    varying = np.flatnonzero((stacked != stacked[0]).any(0))
    return np.setdiff1d(varying, input_echo_bytes(replayed, recordings[:sample]))


def _replay_selected(task):
    """In a worker: each recording's chosen bytes at every step, read straight from save states."""
    byte_indices, recordings = task
    emulator = workers.emulator()
    offsets = emulator.state_offsets(byte_indices)
    results = []
    for state, actions in recordings:
        emulator.load(state)
        rows = np.empty((len(actions) + 1, len(offsets)), np.uint8)
        rows[0] = np.frombuffer(state, np.uint8)[offsets]
        for i, action in enumerate(actions):
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            rows[i + 1] = np.frombuffer(emulator.save(), np.uint8)[offsets]
        results.append(rows)
    return results


def build(emulator, recordings, byte_indices=None, on_device=False):
    """
    Replays recordings into an Experience, in parallel worker processes, into one preallocated array.
    on_device puts that array on the accelerator and writes each worker's rows straight into it, so host
    memory stays small however much data there is.
    """
    if byte_indices is None:
        byte_indices = choose_bytes(emulator, recordings)
    observation_offsets = np.cumsum([0] + [len(actions) + 1 for _, actions in recordings])
    action_offsets = np.cumsum([0] + [len(actions) for _, actions in recordings])
    if on_device:
        import device  # only here: the replay workers import this module and must not load JAX
    shape = (observation_offsets[-1], len(byte_indices))
    observations = device.zeros(shape) if on_device else np.empty(shape, np.uint8)
    chunks = workers.split(recordings)
    firsts = np.cumsum([0] + [len(chunk) for chunk in chunks])
    for first, results in zip(firsts, workers.map(_replay_selected, [(byte_indices, chunk) for chunk in chunks])):
        if on_device:  # a chunk's recordings are consecutive, so their rows are one contiguous block
            observations = device.write_rows(observations, np.concatenate(results), observation_offsets[first])
        else:
            for r, rows in enumerate(results, start=first):
                observations[observation_offsets[r]:observation_offsets[r + 1]] = rows
    return Experience(byte_indices, observations, np.concatenate([a for _, a in recordings]),
                      observation_offsets, action_offsets, [state for state, _ in recordings])
