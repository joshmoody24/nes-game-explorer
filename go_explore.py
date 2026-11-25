"""
Go-Explore on Metroid: keep an archive with one save state per distinct place, repeatedly restore a
rarely-chosen one and press random buttons from there, and add any new places found.
"Place" here is hand-made from documented RAM addresses: the sanity baseline the learned versions
must approach.

Usage: python go_explore.py <run name> [--minutes M] [--workers N] [settings flags, see --help]
"""

import argparse
import dataclasses
import json
import os
import pickle
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emulator import Emulator
from explorers import ROM, Explorers, Settings, hand_made_place, score
from metroid import start_game
from progress import Progress

RUNS = Path("runs")
MAX_RUNS_GIGABYTES = 50
LOG_EVERY_SECONDS = 2
CHECKPOINT_EVERY_SECONDS = 300


@dataclass
class Archive:
    selection: str
    places: list = field(default_factory=list)
    states: list = field(default_factory=list)
    scores: list = field(default_factory=list)  # see explorers.score; the kept state has the best seen
    times_chosen: list = field(default_factory=list)
    discoveries: list = field(default_factory=list)  # new places found by segments started here
    parents: list = field(default_factory=list)  # archive index of the place whose segment found this one (-1: start)
    index: dict = field(default_factory=dict)  # place -> position in the lists above
    updates: list = field(default_factory=list)  # append-only (place, score) log that workers sync from

    def offer(self, place, state, score, parent=-1):
        """Keeps state if place is new or score beats the kept state's. Returns "new", "improved" or None."""
        if place in self.index:
            i = self.index[place]
            if score > self.scores[i]:
                self.states[i], self.scores[i] = state, score
                self.updates.append((place, score))
                return "improved"
            return None
        self.index[place] = len(self.places)
        self.places.append(place)
        self.states.append(state)
        self.scores.append(score)
        self.times_chosen.append(0)
        self.discoveries.append(0)
        self.parents.append(parent)
        self.updates.append((place, score))
        return "new"

    def choose(self, rng):
        chosen = np.array(self.times_chosen)
        weights = 1 / np.sqrt(1 + chosen)
        if self.selection in ("productive", "powerful"):
            weights *= 1 + np.array(self.discoveries) / (1 + chosen)
        if self.selection == "powerful":
            # Explore more from stronger states (safer, better armed), so new places are first reached
            # by healthy lineages. Scores are rescaled to 0-1 so any keep rule's scale works.
            scores = np.array(self.scores, dtype=float)
            spread = scores.max() - scores.min()
            weights *= 0.2 + ((scores - scores.min()) / spread if spread > 0 else 0)
        index = int(rng.choice(len(self.places), p=weights / weights.sum()))
        self.times_chosen[index] += 1
        return index


def stalled(history, now, window_seconds):
    """history: [(time, archive size)]. True once the archive grew by under 1% over the last window."""
    if window_seconds is None or now - history[0][0] < window_seconds:
        return False
    size_then = next(size for when, size in reversed(history) if now - when >= window_seconds)
    return history[-1][1] < size_then * 1.01


def directory_gigabytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def resumed_archive(run_name, settings, emulator):
    """An earlier run's saved states, re-sorted into places under the current settings."""
    saved = pickle.loads(zlib.decompress((RUNS / run_name / "checkpoint.pkl.z").read_bytes()))
    archive = Archive(settings.selection)
    for state in saved["states"]:
        memory = emulator.view(state)
        archive.offer(hand_made_place(memory, settings), state, score(memory, settings))
    return archive


class Recordings:
    """
    Button sequences of sampled segments, saved as trajectories.npz: for recording i, the segment started
    from archive index start[i] (whose state had checksum start_checksum[i]) and pressed
    actions[offsets[i]:offsets[i + 1]] (indices into controls.ACTIONS). Emulation is deterministic, so
    replaying those buttons from that state regenerates every frame.
    """

    def __init__(self):
        self.starts, self.checksums, self.actions, self.offsets = [], [], bytearray(), [0]

    def add(self, segment):
        self.starts.append(segment.start)
        self.checksums.append(segment.start_checksum)
        self.actions += segment.actions
        self.offsets.append(len(self.actions))

    def save(self, path):
        np.savez(path, start=np.array(self.starts), start_checksum=np.array(self.checksums, dtype=np.uint32),
                 actions=np.frombuffer(bytes(self.actions), np.uint8), offsets=np.array(self.offsets))


def checkpoint(run_dir, archive, segment_log, recordings):
    """
    Plain data only, so a checkpoint loads without this module. segments.npz has one row per segment:
    the archive index it started from, how many new places it added, and its frames.
    """
    temporary = run_dir / "checkpoint.tmp"
    temporary.write_bytes(zlib.compress(pickle.dumps(dataclasses.asdict(archive)), level=1))
    temporary.rename(run_dir / "checkpoint.pkl.z")
    np.savez(run_dir / "segments.npz", **{name: np.array(values) for name, values in segment_log.items()})
    recordings.save(run_dir / "trajectories.npz")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--minutes", type=float, default=4)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="start from this run's checkpointed archive")
    parser.add_argument("--record", type=float, default=0.1, metavar="FRACTION",
                        help="share of segments whose button sequences are saved as training data")
    parser.add_argument("--stop-after-stall", type=float, metavar="MINUTES",
                        help="also stop once the archive grew by less than 1%% over the last MINUTES")
    for setting in dataclasses.fields(Settings):
        flag = "--" + setting.name.replace("_", "-")
        if isinstance(setting.default, bool):
            parser.add_argument(flag, action="store_true")
        else:
            kind = int if setting.default is None else type(setting.default)  # the only optional setting is an int
            parser.add_argument(flag, type=kind, default=setting.default)
    args = parser.parse_args()
    settings = Settings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)})
    return args, settings


def prepare_run_dir(args, settings):
    RUNS.mkdir(exist_ok=True)
    used = directory_gigabytes(RUNS)
    if used > MAX_RUNS_GIGABYTES:
        raise SystemExit(f"runs/ holds {used:.1f} GB, over the {MAX_RUNS_GIGABYTES} GB cap; clean up old runs first")
    run_dir = RUNS / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"minutes": args.minutes, "workers": args.workers, "seed": args.seed, "resume": args.resume,
         "record": args.record,
         "stop_after_stall": args.stop_after_stall}
        | dataclasses.asdict(settings),
        indent=2))
    return run_dir


def main():
    args, settings = parse_arguments()
    run_dir = prepare_run_dir(args, settings)
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start = start_game(emulator)
    if args.resume:
        assert settings.keep in ("first", "more-missiles"), "resuming would mean replay-scoring every saved state"
        archive = resumed_archive(args.resume, settings, emulator)
    else:
        archive = Archive(settings.selection)
        memory = emulator.view(start)
        archive.offer(hand_made_place(memory, settings), start, score(memory, settings))
    progress = Progress()
    rng = np.random.default_rng(args.seed)
    frames = 0
    gaps = []
    segment_log = {"start": [], "new_places": [], "frames": []}
    recordings = Recordings()
    explorers = Explorers(args.workers, settings, args.seed)

    def assign(worker_index):
        index = archive.choose(rng)
        explorers.send(worker_index, index, archive.states[index], archive.updates)

    for worker_index in range(explorers.count):
        assign(worker_index)
    started = last_log = last_checkpoint = time.perf_counter()
    deadline = started + args.minutes * 60
    stall_window = args.stop_after_stall * 60 if args.stop_after_stall else None
    sizes = [(started, len(archive.places))]
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        while time.perf_counter() < deadline:
            for worker_index, segment in explorers.finished():
                frames += segment.frames
                progress.merge(segment.progress)
                gaps.extend(segment.gaps)
                outcomes = [archive.offer(place, state, state_score, segment.start)
                            for place, (state, state_score) in segment.found.items()]
                new_places = outcomes.count("new")
                # A stronger state for a known place (healed, rearmed) counts as progress, like a new place.
                archive.discoveries[segment.start] += new_places + outcomes.count("improved")
                if rng.random() < args.record:
                    recordings.add(segment)
                segment_log["start"].append(segment.start)
                segment_log["new_places"].append(new_places)
                segment_log["frames"].append(segment.frames)
                assign(worker_index)
            now = time.perf_counter()
            if now - last_log > LOG_EVERY_SECONDS:
                line = {"seconds": round(now - started, 1), "frames": frames, "places": len(archive.places),
                        "frames per second": round(frames / (now - started)),
                        "steps between discoveries (50/90/99%)": np.percentile(gaps, [50, 90, 99]).round().tolist()
                        if gaps else None} | progress.summary()
                metrics.write(json.dumps(line) + "\n")
                metrics.flush()
                last_log = now
                sizes.append((now, len(archive.places)))
                if stalled(sizes, now, stall_window):
                    print(f"stopping: archive grew under 1% in the last {args.stop_after_stall} minutes", flush=True)
                    break
            if now - last_checkpoint > CHECKPOINT_EVERY_SECONDS:
                checkpoint(run_dir, archive, segment_log, recordings)
                last_checkpoint = now
    explorers.close()
    checkpoint(run_dir, archive, segment_log, recordings)
    print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
