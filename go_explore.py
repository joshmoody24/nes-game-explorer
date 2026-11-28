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
import place_extras
from contingency import player_bytes, varied_saves, x_is_scrolled
from explorers import ROM, Explorers, Settings, beats, hand_made_place, holdings, place_of_state, player_step
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
    # Buttons (indices into controls.actions(Settings.turbo)) from the start state to the kept state: every visit
    # has a replayable route. None for places resumed from an older run, which didn't keep paths.
    paths: list = field(default_factory=list)
    holdings: list = field(default_factory=list)  # see explorers.holdings
    active: list = field(default_factory=list)  # False once another visit to the same place beats it
    deaths: list = field(default_factory=list)  # segments started from each visit that ended in game over
    productive: list = field(default_factory=list)  # segments from each visit that found a new place
    cached_weights: tuple = None  # (archive size, choice probabilities, uses) reused by choose
    index: dict = field(default_factory=dict)  # place -> positions of its visits in the lists above
    place_numbers: dict = field(default_factory=dict)  # place -> a small number, for counting visits per place
    place_of: list = field(default_factory=list)  # each visit's place_numbers entry, kept as visits are added
    room_first: bool = False  # attention split evenly over rooms (screen-sized dead-reckoning blocks) before places
    coarse_bias: float = 1.0  # with grid levels: level L's share of attention is proportional to this ** L
    place_productivity: float = 0.0  # like productivity, but each place's rate pooled over its saves
    productivity: float = 0.0  # weight saves by ((1 + saves their tries created) / (1 + times chosen)) ** this
    choice_power: float = 0.5  # a save's weight falls as 1 / (1 + times chosen) ** this (Go-Explore: 0.5)
    place_level: list = field(default_factory=list)  # each place number's grid level (Settings.place_levels), else 0
    positions: list = field(default_factory=list)  # each visit's exact dead-reckoned position (scroll places), or None
    children: list = field(default_factory=list)  # saves each visit's tries created
    fertility: float = 0.0  # --fertility: weight visits by (1 + children) ** fertility; 0 is off
    descendants: list = field(default_factory=list)  # all saves descended from each visit
    since_scenery_change: list = field(default_factory=list)  # generations since an ancestor had different scenery
    scenery: list = field(default_factory=list)  # each visit's tile bag (see tile_bag)
    descendant_weight: float = 0.0  # --descendants: weight by (1 + descendants) ** this; 0 is off
    scenery_recency: float = 0.0  # --scenery-recency: weight by 1 / (1 + since_scenery_change) ** this; 0 is off
    updates: list = field(default_factory=list)  # append-only (place, (holdings, path length)) log workers sync from

    def path_length(self, i):
        return UNKNOWN_PATH_LENGTH if self.paths[i] is None else len(self.paths[i])

    def offer(self, place, state, held, parent=-1, path=None, scenery=None, position=None):
        """Keeps the visit unless one already kept there beats it. Returns "new" (new place), "better" or None."""
        if self.worn_rooms and place[0] != "record" and tuple(place[:3]) in self.worn_rooms and place not in self.index:
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
            self.positions[slot] = position
            self.parents[slot] = parent
            if parent >= 0:
                self.children[parent] += 1
            self.updates.append((place, visit))
            return "better"
        number = self.place_numbers.setdefault(choice_group(place), len(self.place_numbers))
        if number == len(self.place_level):
            self.place_level.append(place[1] if place[0] == "level" else 0)
        self.place_of.append(number)
        self.index.setdefault(place, []).append(len(self.places))
        self.places.append(place)
        self.states.append(state)
        self.positions.append(position)
        self.times_chosen.append(self.familiar.get(place[:LOCATION_SIZE], 0) if place[0] != "record" else 0)
        self.parents.append(parent)
        self.children.append(0)
        self.descendants.append(0)
        self.scenery.append(scenery)
        self.since_scenery_change.append(self.since_scenery_change[parent] + 1 if parent >= 0 and scenery is not None and scenery == self.scenery[parent] else 0)
        ancestor, seen = parent, set()
        while ancestor >= 0 and ancestor not in seen:  # a replaced visit can take a descendant as parent: loops
            seen.add(ancestor)
            self.descendants[ancestor] += 1
            ancestor = self.parents[ancestor]
        if parent >= 0:
            self.children[parent] += 1
        self.paths.append(path)
        self.holdings.append(held)
        self.active.append(True)
        self.deaths.append(0)
        self.productive.append(0)
        self.updates.append((place, visit))
        return "better" if visits else "new"

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

    def weights(self):
        active = np.array(self.active)
        place_of = np.array(self.place_of)
        visits_per_place = np.bincount(place_of, weights=active)
        weights = active / (1 + np.array(self.times_chosen)) ** self.choice_power / np.maximum(visits_per_place[place_of], 1)
        if any(self.place_level):
            # Every grid size at once: each level gets an equal share, split over that level's places.
            level = np.array(self.place_level)
            share = self.coarse_bias ** np.arange(level.max() + 1)  # each level's share of attention, coarser levels more
            weights = weights * (share / share.sum() / np.bincount(level))[level[place_of]]
        # Stepping stones to a breakthrough are saves whose tries keep creating saves (FINDINGS 2026-11-27).
        if self.productivity:
            # Hill-climb what pays off: a save whose tries keep creating new saves stays hot; one tried often for nothing cools.
            weights = weights * ((1 + np.array(self.children)) / (1 + np.array(self.times_chosen))) ** self.productivity
        if self.room_first and any(p is not None for p in self.positions):
            # Room-first attention: every room (screen-sized block of the dead-reckoned position) gets an equal share, split
            # over its places, so a room with many places doesn't outvote a fresh room with a few.
            rooms = {}
            room_of_place = np.zeros(len(visits_per_place), np.int64)
            for visit, number in enumerate(self.place_of):
                position = self.positions[visit]
                room_of_place[number] = rooms.setdefault(tuple(p // 256 for p in position) if position is not None else ("none",), len(rooms))
            places_in_room = np.bincount(room_of_place, minlength=len(rooms))
            weights = weights / places_in_room[room_of_place[place_of]]
        if self.place_productivity:
            # The same, pooled per place: with many saves per place, one save's own record is too few tries to mean much.
            made = np.bincount(place_of, weights=np.array(self.children), minlength=len(visits_per_place))
            tried = np.bincount(place_of, weights=np.array(self.times_chosen), minlength=len(visits_per_place))
            weights = weights * ((1 + made) / (1 + tried))[place_of] ** self.place_productivity
        if self.fertility:
            weights = weights * (1 + np.array(self.children)) ** self.fertility
        if self.descendant_weight:
            weights = weights * (1 + np.array(self.descendants)) ** self.descendant_weight
        if self.scenery_recency:
            # Saves that just crossed into new scenery were the likeliest to lead on (FINDINGS 2026-11-27, breadth test).
            weights = weights / (1 + np.array(self.since_scenery_change)) ** self.scenery_recency
        return weights


WEIGHTS_REUSE = 200  # choices between recomputing choice weights (see Archive.choose)
GAIN_KEYS = ("items", "energy tanks", "max missiles", "bosses")  # progress fields --stop-at-gain watches


def tile_bag(nametable):
    """Which background tiles fill a good part of the screen (32+ cells across both name tables, attribute bytes
    left out): steady while scrolling within a scene, different in new scenery."""
    cells = np.frombuffer(nametable, np.uint8)[np.r_[0:960, 1024:1984]]
    return frozenset(np.flatnonzero(np.bincount(cells, minlength=256) >= 32).tolist())


def choice_group(place):
    """What choose() splits weight over: a room's record places (Settings.novel_values) count as one place, so
    progress inside a fight is explored without crowding out the map."""
    return place[:2] if place[0] == "record" else place


def stalled(history, now, window_seconds):
    """history: [(time, archive size)]. True once the archive grew by under 1% over the last window."""
    if window_seconds is None or now - history[0][0] < window_seconds:
        return False
    size_then = next(size for when, size in reversed(history) if now - when >= window_seconds)
    return history[-1][1] < size_then * 1.01


def directory_gigabytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def resumed_archive(run_name, settings, emulator, first=0):
    """An earlier run's saved states from index first on, re-sorted into places under the current settings."""
    saved = pickle.loads(zlib.decompress((RUNS / run_name / "checkpoint.pkl.z").read_bytes()))
    archive = Archive()
    # Keeps a visit's stored place when it isn't a hand-made one (e.g. a record place, see Settings.novel_values).
    places = saved.get("places") or [None] * len(saved["states"])
    paths = saved.get("paths") or [None] * len(saved["states"])
    for state, place, path in list(zip(saved["states"], places, paths))[first:]:
        memory = emulator.view(state)
        # Record, screen and scroll places are kept as stored: their history (cells, distance travelled) isn't in the state.
        archive.offer(place if place is not None and place[0] in ("record", "screen", "scroll", "level", "radius", "split", "group", "compass", "ruler") else place_of_state(emulator, state, settings),
                      state, holdings(memory, settings), path=path)
    return archive


class Recordings:
    """
    Button sequences of sampled segments, saved as trajectories.npz: for recording i, the segment started
    from archive index start[i] (whose state had checksum start_checksum[i]) and pressed
    actions[offsets[i]:offsets[i + 1]] (indices into controls.actions(Settings.turbo)). Emulation is
    deterministic, so replaying those buttons from that state regenerates every frame.
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
    parser.add_argument("--resume-after", type=int, default=0, help="with --resume, keep only saves from this index on (e.g. those a stalled chunk found)")
    parser.add_argument("--resume", help="start from this run's checkpointed archive")
    parser.add_argument("--skip-worn", action="store_true", help="don't keep new saves in rooms history wore out (more than the median room's history)")
    parser.add_argument("--stop-when", help="a bench.py wall name: stop as soon as a new save passes that wall's success test")
    parser.add_argument("--descendants", type=float, default=0.0, help="weight visits by (1 + saves descended from them) ** this; 0 is off")
    parser.add_argument("--scenery-recency", type=float, default=0.0, help="weight visits by 1 / (1 + generations since their scenery changed) ** this; 0 is off")
    parser.add_argument("--room-first", action="store_true", help="split attention evenly over rooms first, then over places in each room")
    parser.add_argument("--coarse-bias", type=float, default=1.0, help="with grid levels: level L gets attention in proportion to this ** L (1 = equal)")
    parser.add_argument("--place-productivity", type=float, default=0.0, help="weight places by their saves' pooled rate of creating new saves ** this")
    parser.add_argument("--productivity", type=float, default=0.0, help="weight saves by their rate of creating new saves ** this; 0 is off")
    parser.add_argument("--choice-power", type=float, default=0.5, help="weight saves by 1 / (1 + times chosen) ** this")
    parser.add_argument("--fertility", type=float, default=0.0, help="weight visits by (1 + saves their tries created) ** this; 0 is off")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many emulated frames (a budget that means the same on any machine); 0 = no limit")
    parser.add_argument("--segments", type=int, default=0, help="stop after this many tries (for exact before/after comparisons); 0 = no limit")
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
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=setting.default)
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


def location_tries(name, emulator):
    """(permanent progress at the run's start, {location: tries}) for one earlier run, cached next to its archive
    because decompressing a whole archive for these few numbers took seconds per run."""
    run = Path("runs") / name
    cache, checkpoint = run / "location_tries.pkl", run / "checkpoint.pkl.z"
    if cache.exists() and cache.stat().st_mtime >= checkpoint.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    run_start = Path(json.loads((run / "config.json").read_text())["start_state"]).read_bytes()
    saved = pickle.loads(zlib.decompress(checkpoint.read_bytes()))
    counts = {}
    for place, chosen in zip(saved["places"], saved["times_chosen"]):
        if place[0] != "record":
            counts[tuple(place[:LOCATION_SIZE])] = counts.get(tuple(place[:LOCATION_SIZE]), 0) + chosen
    result = (permanent_progress(emulator.view(run_start)), counts)
    cache.write_bytes(pickle.dumps(result))
    return result


def familiar_counts(run_names, start, emulator):
    """Choices earlier runs spent at each location, so this run's novelty is judged against earlier experience.
    Only runs that began with the same permanent progress count: a new item or a beaten boss can open up places
    that were worn out before."""
    progress = permanent_progress(emulator.view(start))
    counts = {}
    for name in filter(None, run_names.split(",")):
        run_progress, run_counts = location_tries(name, emulator)
        if run_progress == progress:
            for location, n in run_counts.items():
                counts[location] = counts.get(location, 0) + n
    return counts


def main():
    args, settings = parse_arguments()
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    start = Path(args.start_state).read_bytes() if args.start_state else start_game(emulator)
    if player_step(settings) and not settings.player_bytes:
        found = player_bytes(emulator, start, np.random.default_rng(args.seed), settings.turbo, settings.player_per_axis)
        settings = dataclasses.replace(settings, player_bytes=",".join(map(str, found)))
        if settings.player_y_sprite or settings.player_world or settings.compass or settings.ruler:
            saves = varied_saves(emulator, start, np.random.default_rng(args.seed + 1), 60, settings.turbo)
            settings = dataclasses.replace(settings, player_x_scrolled=x_is_scrolled(emulator, saves, found[0]))
            print(f"player x byte counts on the background: {settings.player_x_scrolled}", flush=True)
        print(f"player bytes (by contingency): {settings.player_bytes}", flush=True)
    if settings.place_extras:
        share = np.load("cache/poke_share.npy") if settings.byte_picker == "poke" else None
        learned = place_extras.pick(emulator, start, np.random.default_rng(args.seed), settings, share)
        settings = dataclasses.replace(settings, reference_state=args.start_state or "", **learned)
        print(f"place add-ons {settings.place_extras}: {learned}", flush=True)
    run_dir = prepare_run_dir(args, settings)
    if args.resume:
        archive = resumed_archive(args.resume, settings, emulator, args.resume_after)
    else:
        archive = Archive()
        memory = emulator.view(start)
        archive.offer(place_of_state(emulator, start, settings), start, holdings(memory, settings), path=b"")
    archive.familiar = familiar_counts(args.familiar, start, emulator)
    archive.fertility = args.fertility
    archive.choice_power = args.choice_power
    archive.productivity = args.productivity
    archive.place_productivity = args.place_productivity
    archive.coarse_bias = args.coarse_bias
    archive.room_first = args.room_first
    archive.descendant_weight = args.descendants
    archive.scenery_recency = args.scenery_recency
    if args.skip_worn and archive.familiar:
        archive.worn_rooms = worn_rooms(archive.familiar, hand_made_place(emulator.view(start), settings))
    progress = Progress()
    rng = np.random.default_rng(args.seed)
    frames = 0
    gaps = []
    segment_log = {"start": [], "new_places": [], "frames": []}
    recordings = Recordings()
    explorers = Explorers(args.workers, settings, args.seed)

    # The path to each worker's start state, as it was when assigned: the archive may replace that state
    # (with a shorter path) while the worker explores from it.
    start_paths = [None] * explorers.count

    def assign(worker_index):
        index = archive.choose(rng)
        start_paths[worker_index] = archive.paths[index]
        explorers.send(worker_index, index, archive.states[index], archive.path_length(index), archive.places[index],
                       archive.positions[index], archive.updates)

    for worker_index in range(explorers.count):
        assign(worker_index)
    started = last_log = last_checkpoint = time.perf_counter()
    first_line = None  # the first metrics line, the baseline for --stop-at-gain
    deadline = started + args.minutes * 60
    segments_done = frames_done = 0
    stop_test = None
    if args.stop_when:
        from bench import WALLS
        stop_test = WALLS[args.stop_when][1]
    stall_window = args.stop_after_stall * 60 if args.stop_after_stall else None
    sizes = [(started, len(archive.places))]
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        while time.perf_counter() < deadline:
            for worker_index, segment in explorers.finished():
                segments_done += 1
                frames_done += segment.frames
                if args.max_frames and frames_done >= args.max_frames:
                    deadline = time.perf_counter()
                    break
                if args.segments and segments_done >= args.segments:
                    deadline = time.perf_counter()
                    break
                frames += segment.frames
                progress.merge(segment.progress)
                gaps.extend(segment.gaps)
                start_path = start_paths[worker_index]
                outcomes = []
                for place, (state, length, held, position) in segment.found.items():
                    outcomes.append(archive.offer(place, state, held, segment.start, scenery=tile_bag(emulator.nametable(state)) if archive.scenery_recency else None, path=
                                                  None if start_path is None else start_path + segment.actions[:length - len(start_path)], position=position))
                    if outcomes[-1]:
                        if stop_test and stop_test(emulator.view(state)):
                            print(f"stopping: --stop-when {args.stop_when} passed after {time.perf_counter() - started:.0f}s", flush=True)
                            deadline = time.perf_counter()
                new_places = outcomes.count("new")
                archive.deaths[segment.start] += segment.died
                archive.productive[segment.start] += new_places > 0
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
    if progress.ever_up is not None:
        # Every bit ever seen turning on, and every bit ever seen turning off, over every step of the run.
        np.savez(run_dir / "bit_flips.npz", ever_up=progress.ever_up, ever_down=progress.ever_down)
    if first_line is not None:
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
