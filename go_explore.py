"""
Go-Explore on Metroid: keep an archive with one save state per distinct place, repeatedly restore a
rarely-chosen one and press random buttons from there, and add any new places found.
"Place" here is hand-made from documented RAM addresses: the sanity baseline the learned versions
must approach.

Usage: python go_explore.py <run name> [--minutes M] [--workers N] [settings flags, see --help]
"""

import argparse
import functools
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
from explorers import ROM, Explorers, Settings, beats, hand_made_place, holdings
import danger
from metroid import permanent_progress, start_game
from progress import Progress

RUNS = Path("runs")
UNKNOWN_PATH_LENGTH = 10**9  # ranks a resumed place, whose path is unknown, below any place with one
MAX_RUNS_GIGABYTES = 50
LOG_EVERY_SECONDS = 2
CHECKPOINT_EVERY_SECONDS = 300


@dataclass
class Archive:
    """
    Visits worth keeping: for each place, every visit that no other visit there beats (see explorers.beats).
    A visit that beats a kept one takes over its slot; any others it beats stay in the lists (others refer
    to them by position) but are no longer chosen.
    """
    places: list = field(default_factory=list)
    states: list = field(default_factory=list)
    times_chosen: list = field(default_factory=list)
    familiar: dict = field(default_factory=dict)  # location -> choices earlier runs spent there (--familiar)
    worn_rooms: frozenset = frozenset()  # rooms history wore out (--skip-worn): new saves there aren't kept
    parents: list = field(default_factory=list)  # archive index of the visit whose segment found this one (-1: start)
    # Buttons (indices into ACTIONS) from the start state to the kept state: every visit has a replayable
    # route. None for places resumed from an older run, which didn't keep paths.
    paths: list = field(default_factory=list)
    holdings: list = field(default_factory=list)  # see explorers.holdings
    active: list = field(default_factory=list)  # False once another visit to the same place beats it
    surprises: list = field(default_factory=list)  # new change patterns found by segments started from each visit
    deaths: list = field(default_factory=list)  # segments started from each visit that ended in game over
    surprise_weight: float = 0.0  # see explorers.Settings.surprise
    holdings_weight: float = 0.0  # see explorers.Settings.favor_holdings
    dangers: list = field(default_factory=list)  # see explorers.Settings.danger_model
    danger_weight: float = 0.0
    productive: list = field(default_factory=list)  # segments from each visit that found a new place
    mastery_weight: float = 0.0
    chooser_weight: float = 0.0  # see explorers.Settings.learned_chooser
    chooser: tuple = None  # (weights, bias, means, scales) of the learned chooser, once fit
    cached_weights: tuple = None  # (archive size, choice probabilities, uses) reused by choose
    recency_weight: float = 0.0  # see explorers.Settings.recency
    selection: str = "classic"  # see explorers.Settings.selection
    rng: object = None  # draws Thompson samples (set by main)
    progress: list = field(default_factory=list)  # how far along the guide route each visit is (0-1)
    # The locations each visit's lineage passed through, snapshotted when the visit was kept (parent links
    # can loop after slot reuse; these can't). Kept only when exploring with a refill (to pay back cheats).
    routes: list = field(default_factory=list)
    guide_weight: float = 0.0
    index: dict = field(default_factory=dict)  # place -> positions of its visits in the lists above
    place_numbers: dict = field(default_factory=dict)  # place -> a small number, for counting visits per place
    updates: list = field(default_factory=list)  # append-only (place, (holdings, path length)) log workers sync from

    def path_length(self, i):
        return UNKNOWN_PATH_LENGTH if self.paths[i] is None else len(self.paths[i])

    def offer(self, place, state, held, parent=-1, path=None):
        """Keeps the visit unless one already kept there beats it. Returns "new" (new place), "better" or None."""
        if self.worn_rooms and place[0] not in ("record", "climb") and tuple(place[:3]) in self.worn_rooms and place not in self.index:
            return None
        visit = (held, UNKNOWN_PATH_LENGTH if path is None else len(path))
        visits = self.index.get(place, [])
        if any(self.active[i] and beats((self.holdings[i], self.path_length(i)), visit) for i in visits):
            return None
        beaten = [i for i in visits if self.active[i] and beats(visit, (self.holdings[i], self.path_length(i)))]
        if beaten:
            # The new visit takes over the first beaten one's slot, so the lists don't fill with dead visits.
            slot, *others = beaten
            for i in others:
                self.active[i] = False
            self.states[slot], self.paths[slot], self.holdings[slot] = state, path, held
            self.parents[slot] = parent
            self.updates.append((place, visit))
            self.last_slot = slot
            return "better"
        self.place_numbers.setdefault(choice_group(place), len(self.place_numbers))
        self.index.setdefault(place, []).append(len(self.places))
        self.places.append(place)
        self.states.append(state)
        self.times_chosen.append(self.familiar.get(place[:LOCATION_SIZE], 0) if place[0] not in ("record", "climb") else 0)
        self.parents.append(parent)
        self.paths.append(path)
        self.holdings.append(held)
        self.active.append(True)
        self.last_slot = len(self.places) - 1
        self.surprises.append(0)
        self.deaths.append(0)
        self.productive.append(0)
        self.updates.append((place, visit))
        return "better" if visits else "new"

    SIGNALS = ["rarely chosen", "holdings", "danger", "surprise rate", "death rate", "success rate", "path length"]

    def signals(self, visits=None):
        """[visits, drives]: the drive signals of the given visits (default all) as they stand now (the learned
        chooser's senses)."""
        visits = range(len(self.places)) if visits is None else visits
        pick = lambda values, default=0.0: np.array([values[i] if i < len(values) else default for i in visits], float)
        tries = 1 + pick(self.times_chosen)
        held = np.array([sum(self.holdings[i]) for i in visits], float)
        lengths = np.array([self.path_length(i) for i in visits], float)
        return np.stack([1 / np.sqrt(tries), held, pick(self.dangers), pick(self.surprises) / tries,
                         pick(self.deaths) / tries, pick(self.productive) / tries, np.minimum(lengths, 1e5)], 1)

    def choose(self, rng):
        """
        A rarely chosen visit still worth keeping. Chooses among places, not visits (each place's weight split
        evenly between its kept visits), so places holding several kept visits don't slow coverage. Weights take
        time proportional to the archive, so they are reused for WEIGHTS_REUSE choices while it grows under 1%.
        """
        size, probabilities, uses = self.cached_weights or (0, None, WEIGHTS_REUSE)
        if uses >= WEIGHTS_REUSE or len(self.places) > size * 1.01:
            weights = self.weights()
            size, probabilities, uses = len(self.places), weights / weights.sum(), 0
        index = int(rng.choice(size, p=probabilities))
        self.cached_weights = (size, probabilities, uses + 1)
        self.times_chosen[index] += 1
        return index

    def sampled_rates(self, chosen):
        """One Thompson sample per visit of its chance that a try finds something new: Beta(1 + successes, prior
        failures + failures), where the prior has the strength of one success and the mean of the archive's rate
        ("thompson") or of the visit's room's rate, itself starting from the archive's ("thompson-room")."""
        found = np.array(self.productive, float)
        failed = np.maximum(chosen - found, 0)
        rate = (found.sum() + 1) / (chosen.sum() + 2)
        prior_failures = np.full(len(found), 1 / rate - 1)
        if self.selection == "thompson-room":
            rooms = {}
            room_of = np.array([rooms.setdefault(tuple(p[1][:3]) if p[0] == "record" else tuple(p[:3]), len(rooms))
                                for p in self.places])
            room_found, room_tries = np.bincount(room_of, found), np.bincount(room_of, chosen)
            room_rate = (room_found + 1) / (room_tries + 1 / rate)
            prior_failures = 1 / room_rate[room_of] - 1
        return self.rng.beta(1 + found, prior_failures + failed)

    def weights(self):
        active = np.array(self.active)
        place_of = np.array([self.place_numbers[choice_group(p)] for p in self.places]) if len(self.place_numbers) else None
        visits_per_place = np.bincount(place_of, weights=active)
        chosen = np.array(self.times_chosen)
        weights = (self.sampled_rates(chosen) if self.selection.startswith("thompson") else 1 / np.sqrt(1 + chosen)) * active
        weights = weights / np.maximum(visits_per_place[place_of], 1)
        if self.selection == "rooms":
            # Each room gets one share however many saves it holds, so a big explored area can't drown out the
            # few rooms where progress is happening; the share fades with the tries the room has had.
            rooms = {}
            room_of = np.array([rooms.setdefault(p[1] if p[0] in ("record", "climb") else tuple(p[:3]), len(rooms)) for p in self.places])
            room_total = np.bincount(room_of, weights=weights)
            room_tries = np.bincount(room_of, weights=chosen)
            weights = weights / np.maximum(room_total[room_of], 1e-12) / np.sqrt(1 + room_tries[room_of])
        # Visits whose segments keep surprising get chosen more; the bonus fades unless surprises continue.
        weights = weights * (1 + self.surprise_weight * np.log1p(np.array(self.surprises)) / (1 + chosen))
        if self.chooser_weight and self.chooser is not None:
            w, b, means, scales = self.chooser
            logit = ((self.signals() - means) / scales) @ w + b
            logit = (logit - logit.mean()) / (logit.std() + 1e-9)
            weights = weights * np.exp(np.clip(self.chooser_weight * logit, -20, 20))
        if self.guide_weight and self.progress:
            progress = np.array(self.progress + [0.0] * (len(weights) - len(self.progress)))
            weights = weights * (1 + self.guide_weight * progress)
        if self.mastery_weight:
            p = (np.array(self.productive) + 0.5) / (chosen + 1)
            weights = weights * (1 + self.mastery_weight * 4 * np.clip(p, 0, 1) * (1 - np.clip(p, 0, 1)))
        if self.danger_weight and self.dangers:
            danger = np.array(self.dangers)
            weights = weights / (1 + self.danger_weight * danger / max(float(np.median(danger)), 1e-6))
        if self.recency_weight:
            weights = weights * (1 + self.recency_weight * (np.arange(len(weights)) / max(len(weights) - 1, 1)) ** 4)
        if self.holdings_weight and self.holdings and len(self.holdings[0]):
            held = np.array(self.holdings, dtype=float).sum(1)
            weights = weights * (1 + self.holdings_weight * held / max(held.max(), 1e-9))
        return weights


CHOOSER_REFIT_EVERY, CHOOSER_WINDOW = 2000, 50000
WEIGHTS_REUSE = 200  # choices between recomputing choice weights (see Archive.choose)
GAIN_KEYS = ("items", "energy tanks", "max missiles", "bosses")  # progress fields --stop-at-gain watches
CHOOSER_LAG = 5000  # decisions a "later" label waits for its lineage to grow


@functools.lru_cache(maxsize=1)
def guide_route(path):
    """{location: progress 0-1} along a guide route (see explorers.Settings.guide)."""
    locations = pickle.loads(Path(path).read_bytes())
    return {tuple(loc): (k + 1) / len(locations) for k, loc in enumerate(locations)}


def choice_group(place):
    """What choose() splits weight over: a room's record places (Settings.novel_values) count as one place, so
    progress inside a fight is explored without crowding out the map."""
    return place[:2] if place[0] in ("record", "climb") else place


def place_location(place):
    """The location part of a hand-made place (area, room, position), as in graph.location."""
    return tuple(place[:5])


def lineage_sizes(parents):
    """Number of descendants of every visit (cycles from slot reuse are cut)."""
    from checks.value_check import descendants
    return descendants(parents)


def fit_chooser(x, y, steps=300, rate=0.2):
    """Logistic regression (standardized inputs) from drive signals to "found a new place"."""
    means, scales = x.mean(0), x.std(0) + 1e-9
    z = (x - means) / scales
    w, b = np.zeros(z.shape[1]), float(np.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        w -= rate * (z.T @ (p - y) / len(y) + 1e-3 * w)
        b -= rate * float((p - y).mean())
    return w, b, means, scales


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
    archive = Archive()
    # Keeps a visit's stored place when it isn't a hand-made one (e.g. a record place, see Settings.novel_values).
    places = saved.get("places") or [None] * len(saved["states"])
    paths = saved.get("paths") or [None] * len(saved["states"])
    for state, place, path in zip(saved["states"], places, paths):
        memory = emulator.view(state)
        archive.offer(place if place is not None and place[0] in ("record", "climb") else hand_made_place(memory, settings),
                      state, holdings(memory, settings), path=path)
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
    parser.add_argument("--skip-worn", action="store_true", help="don't keep new saves in rooms history wore out (more than the median room's history)")
    parser.add_argument("--familiar", default="", help="runs whose choices count as already spent here: novelty is judged against all earlier experience, not just this run")
    parser.add_argument("--start-state", help="a file holding a save state to explore from, instead of power-on")
    parser.add_argument("--stop-at-item", help="diagnostic: stop once progress shows this item (e.g. \"long beam\")")
    parser.add_argument("--stop-at-gain", action="store_true",
                        help="diagnostic: stop at the first new item, energy tank, missile capacity or boss")
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
         "start_state": args.start_state,
         "record": args.record,
         "stop_after_stall": args.stop_after_stall}
        | dataclasses.asdict(settings),
        indent=2))
    return run_dir


LOCATION_SIZE = 5  # a hand-made place starts with (area, map y, map x, block x, block y)


def worn_rooms(familiar, start_place):
    """Rooms where history spent more than the median room did; the start's room never counts as worn."""
    rooms = {}
    for location, n in familiar.items():
        rooms[location[:3]] = rooms.get(location[:3], 0) + n
    median = float(np.median(list(rooms.values())))
    return frozenset(r for r, n in rooms.items() if n > median and r != tuple(start_place[:3]))


def familiar_counts(run_names, start, emulator):
    """Choices earlier runs spent at each location, so this run's novelty is judged against earlier experience.
    Only runs that began with the same permanent progress count: a new item or a beaten boss can open up places
    that were worn out before."""
    progress = permanent_progress(emulator.view(start))
    counts = {}
    for name in filter(None, run_names.split(",")):
        run_start = Path(json.loads((Path("runs") / name / "config.json").read_text())["start_state"]).read_bytes()
        if permanent_progress(emulator.view(run_start)) != progress:
            continue
        saved = pickle.loads(zlib.decompress((Path("runs") / name / "checkpoint.pkl.z").read_bytes()))
        for place, chosen in zip(saved["places"], saved["times_chosen"]):
            if place[0] not in ("record", "climb"):
                counts[tuple(place[:LOCATION_SIZE])] = counts.get(tuple(place[:LOCATION_SIZE]), 0) + chosen
    return counts


def main():
    args, settings = parse_arguments()
    run_dir = prepare_run_dir(args, settings)
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start = Path(args.start_state).read_bytes() if args.start_state else start_game(emulator)
    if args.resume:
        archive = resumed_archive(args.resume, settings, emulator)
    else:
        archive = Archive(surprise_weight=settings.surprise, holdings_weight=settings.favor_holdings,
                          danger_weight=settings.avoid_danger if settings.danger_model else 0.0,
                          mastery_weight=settings.mastery, chooser_weight=settings.learned_chooser,
                          guide_weight=settings.guide_weight if settings.guide else 0.0, recency_weight=settings.recency,
                          selection=settings.selection, rng=np.random.default_rng(args.seed),
                          chooser=pickle.loads(Path(settings.chooser_file).read_bytes()) if settings.chooser_file else None)
        memory = emulator.view(start)
        archive.offer(hand_made_place(memory, settings), start, holdings(memory, settings), path=b"")
    archive.familiar = familiar_counts(args.familiar, start, emulator)
    if args.skip_worn and archive.familiar:
        archive.worn_rooms = worn_rooms(archive.familiar, hand_made_place(emulator.view(start), settings))
    progress = Progress()
    rng = np.random.default_rng(args.seed)
    frames = 0
    gaps = []
    segment_log = {"start": [], "new_places": [], "frames": []}
    recordings = Recordings()
    seen_patterns = set()  # every change pattern any segment has produced (see explorers.Settings.surprise)
    explorers = Explorers(args.workers, settings, args.seed)

    # The path to each worker's start state, as it was when assigned: the archive may replace that state
    # (with a shorter path) while the worker explores from it.
    start_paths = [None] * explorers.count

    decisions = {"signals": [], "found": [], "visit": [], "created": []}
    pending_signals = [None] * explorers.count

    def assign(worker_index):
        index = archive.choose(rng)
        # A copy: a row view would keep the whole signals table alive for every logged decision.
        pending_signals[worker_index] = archive.signals([index])[0]
        start_paths[worker_index] = archive.paths[index]
        explorers.send(worker_index, index, archive.states[index], archive.path_length(index), archive.updates,
                       archive.paths[index][-1] if archive.paths[index] else None)

    for worker_index in range(explorers.count):
        assign(worker_index)
    started = last_log = last_checkpoint = time.perf_counter()
    first_line = None  # the first metrics line, the baseline for --stop-at-gain
    deadline = started + args.minutes * 60
    stall_window = args.stop_after_stall * 60 if args.stop_after_stall else None
    sizes = [(started, len(archive.places))]
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        while time.perf_counter() < deadline:
            for worker_index, segment in explorers.finished():
                frames += segment.frames
                progress.merge(segment.progress)
                gaps.extend(segment.gaps)
                start_path = start_paths[worker_index]
                outcomes, created = [], []
                for place, (state, length, held) in segment.found.items():
                    outcomes.append(archive.offer(place, state, held, segment.start,
                                                  None if start_path is None else start_path + segment.actions[:length - len(start_path)]))
                    if outcomes[-1]:
                        created.append(archive.last_slot)
                        if settings.refill:
                            while len(archive.routes) < len(archive.states):
                                archive.routes.append(())
                            here = place_location(place)
                            parent_route = archive.routes[segment.start] if segment.start < len(archive.routes) else ()
                            archive.routes[archive.last_slot] = (parent_route if parent_route[-1:] == (here,)
                                                                 else parent_route + (here,))
                new_places = outcomes.count("new")
                if settings.danger_model:
                    while len(archive.dangers) < len(archive.states):
                        archive.dangers.append(0.0)
                    for place, (state, _, _) in segment.found.items():
                        for i in archive.index.get(place, []):
                            if archive.states[i] is state:
                                archive.dangers[i] = danger.score(settings.danger_model, emulator.memory(state))
                archive.surprises[segment.start] += len(segment.patterns - seen_patterns)
                archive.deaths[segment.start] += segment.died
                archive.productive[segment.start] += new_places > 0
                decisions["signals"].append(pending_signals[worker_index])
                decisions["found"].append(new_places > 0)
                if settings.guide:
                    route = guide_route(settings.guide)
                    while len(archive.progress) < len(archive.states):
                        archive.progress.append(0.0)
                    for slot in created:
                        here = place_location(archive.places[slot])
                        if here in route:
                            archive.progress[slot] = max(archive.progress[slot], route[here])
                decisions["visit"].append(segment.start)
                decisions["created"].append(created)
                if len(decisions["found"]) % CHOOSER_REFIT_EVERY == 0 and not settings.chooser_file:
                    if settings.chooser_target == "later":
                        old = len(decisions["found"]) - CHOOSER_LAG
                        if old > CHOOSER_REFIT_EVERY:
                            window = range(max(0, old - CHOOSER_WINDOW), old)
                            size = lineage_sizes(archive.parents)
                            growth = np.array([sum(1 + size[c] for c in decisions["created"][k]) for k in window], float)
                            positive = growth[growth > 0]
                            if len(positive):
                                labels = (growth >= np.quantile(positive, 0.98)).astype(float)
                                archive.chooser = fit_chooser(np.array([decisions["signals"][k] for k in window]), labels)
                    else:
                        archive.chooser = fit_chooser(np.array(decisions["signals"][-CHOOSER_WINDOW:]),
                                                      np.array(decisions["found"][-CHOOSER_WINDOW:], float))
                seen_patterns |= segment.patterns
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
                first_line = first_line or line
                if args.stop_at_gain and any(line[k] != first_line[k] for k in GAIN_KEYS):
                    print(f"stopping: an acquisition after {now - started:.0f}s", flush=True)
                    break
                if args.stop_at_item and any(args.stop_at_item in item for item in line["items"]):
                    print(f"stopping: got the {args.stop_at_item} after {now - started:.0f}s", flush=True)
                    break
                if stalled(sizes, now, stall_window):
                    print(f"stopping: archive grew under 1% in the last {args.stop_after_stall} minutes", flush=True)
                    break
            if now - last_checkpoint > CHECKPOINT_EVERY_SECONDS:
                checkpoint(run_dir, archive, segment_log, recordings)
                last_checkpoint = now
    explorers.close()
    checkpoint(run_dir, archive, segment_log, recordings)
    np.savez(run_dir / "decisions.npz", signals=np.array(decisions["signals"]), found=np.array(decisions["found"]),
             names=np.array(Archive.SIGNALS), visit=np.array(decisions["visit"]),
             created_offsets=np.cumsum([0] + [len(c) for c in decisions["created"]]),
             created=np.array([i for c in decisions["created"] for i in c], int))
    if progress.ever_up is not None:
        # Every bit ever seen turning on, and every bit ever seen turning off, over every step of the run.
        np.savez(run_dir / "bit_flips.npz", ever_up=progress.ever_up, ever_down=progress.ever_down)
    print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
