"""
Worker processes that press random buttons from saved states, one emulator each.

Each worker keeps its own copy of the known places and the path length of the state kept for each, so
it can end a segment that has stopped finding anything new and send back only states that are new or
reached by a shorter path. The main process keeps an append-only log of (place, path length) updates,
so keeping a worker in sync just means sending the updates logged since its last task.
"""

import os
import zlib
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from controls import FRAMES_PER_STEP, TURBO_FROM_ENVIRONMENT, actions
from emulator import Emulator
from metroid import ENERGY_TANKS, GAME_MODE, GAME_OVER, ITEMS, MAP_X, MAP_Y, MAX_MISSILES, ROOM_X, ROOM_Y, area_of, at
from place_extras import Extras
from progress import Progress

ROM = "roms/metroid.nes"
PROGRESS_SAMPLE_EVERY = 25


@dataclass(frozen=True)
class Settings:
    steps_per_segment: int = 100
    turbo: bool = TURBO_FROM_ENVIRONMENT  # also offer A/B pressed every step instead of held (see controls.actions)
    # End a segment after this many steps in a row without a new place. None never ends early.
    stale_steps: int | None = None
    # Go-Explore's random exploration repeats the previous action most of the time, so it makes
    # sustained moves (a full jump, a run across a room) instead of jittering in place.
    repeat_probability: float = 0.9
    # Each segment lowers its repeat probability by a random amount up to this, so some segments jitter: tapping
    # a button many times in a row (firing several shots) is near impossible when presses are long holds.
    repeat_spread: float = 0.0
    # Hoarding: comma-separated memory indices (memory() numbering, optionally "index:scale") of things random
    # play spends. Each place keeps every visit that no other visit beats on all of them (see beats), so
    # nothing is traded for anything else; a state holding more can become one holding less, not back. Empty
    # keeps just the shortest path.
    keep_more: str = ""
    block_size: int = 32  # pixels per side of the position blocks that split a room into places
    # Diagnostic focus: map cells "y,x;y,x;..." a try must stay in (leaving ends it like a death). Empty: off.
    stay: str = ""
    # Add a fingerprint of the background tile layout to places, so scenery damage (e.g. a barrier being shot
    # down) counts as progress. Pair with stay: scrolling also changes it.
    place_nametable: bool = False
    # Keep a save whenever a rarely-changing byte sets a new record low or high in this room (e.g. a boss's hit
    # points dropping, a hit counter rising), as its own place: progress that rooms and positions can't see. A byte
    # that has set records both ways in a room is not a progress counter; one counts only from its PROGRESS_RECORDS-th
    # one-way record on.
    novel_values: bool = False
    # What a place is: "hand" (Metroid's map and inventory bytes, see hand_made_place), or a game-agnostic scheme:
    # "screen:W,H,D[,N]", the picture shrunk to W x H cells of D gray levels (Go-Explore's domain-agnostic cell),
    # together with the N different cells seen before it: look-alike screens are told apart by where play came from.
    # "scroll:B": dead reckoning. The camera's moves (hardware scroll, see Emulator.scroll) added up since the start,
    # in B-pixel blocks; B must divide 32 so a place plus the current scroll pins down the exact position again.
    # "scroll:B,P" adds where the player is on screen: the controller-driven bytes (player_bytes) in steps of P.
    place_scheme: str = "hand"
    # memory() indices of the player's position, comma-separated; go_explore fills it in (see contingency.py).
    player_bytes: str = ""
    player_per_axis: int = 2  # how many controller-driven bytes per direction contingency picks (copies add places)
    # Height from the player's sprite instead of the up-down byte: the visible hardware sprite nearest the player's x
    # byte gives y. The jump test that finds the up-down byte is weak (a jump comes back down); left-right is strong.
    player_y_sprite: bool = False
    player_x_scrolled: bool = False  # the x byte counts on the background, so screen x = x byte - scroll x (go_explore checks)
    # Experimental add-ons to scroll places, comma-separated (see place_extras.py), and what they learned at run start.
    place_extras: str = ""
    byte_picker: str = "sticky"
    extra_bytes: str = ""
    common_tiles: str = ""
    reference_state: str = ""  # file of the run's start save (the fuzzy add-on hashes RAM relative to it)
    # Every grid size at once: >1 files each save under this many scroll places, the block (and player step) doubling
    # each level ("level", L, ...). A save is new if new at any level; go_explore shares attention evenly across levels.
    place_levels: int = 1
    level_player: bool = True  # with place_levels: the player step doubles per level too (False keeps it fixed)
    # Every grid shape at once: "B1xP1,B2xP2,..." files each save under every listed grid (scroll block B px, player
    # step P px), attention shared evenly across grids like place_levels. Overrides place_levels when set.
    place_grids: str = ""
    # Dead reckoning from pixels instead of the NES scroll registers: each step, slide this frame's row and column
    # brightness profiles against the last frame's to see how the background moved (98% exact vs the hardware in
    # checks/pixel_scroll_check.py). A step whose best match is poor counts as a cut (no travel).
    pixel_scroll: bool = False
    # Record places (novel_values) group by the screen-sized dead-reckoning block instead of the place itself.
    screen_rooms: bool = False
    # One measurement per axis: place position = the player's world position (dead-reckoned camera + the player's offset
    # on screen), with no separate player part, so the camera and a player byte never count the same move twice.
    player_world: bool = False
    # Compass champions: each room (screen-sized block of the player's world position) has 8 places, one per compass
    # direction; a save's holding there is how far it got that way, so the keep rule keeps the one that went furthest.
    compass: bool = False
    # Nearest-center places (k-center): >0 makes a place the nearest known center within this many pixels (L-infinity
    # over scroll position and player bytes), else the save starts a new center. No grid, so no cell edges.
    place_radius: int = 0
    # Split a place when it earns it (crowding version): >0 starts each place as a coarse scroll block (8x the scheme's
    # block) and splits it into finer ones once play has landed in it on this many steps, down to the scheme's block.
    place_split: int = 0
    # Stitched world map: after each camera jump, look the background layout up in a map of layouts seen so far and put
    # the position back where that layout was first seen (shifted by the scroll difference), so every path into a room
    # agrees on where it is. Off: positions only stay continuous.
    stitch: bool = False
    # Mutual-reachability places: 32-pixel scroll blocks merge into one group once play has moved both ways between
    # them; a place is (group, player part). One-way drops and locked doors stay group borders.
    place_groups: bool = False
    hand_inventory: bool = True  # diagnostic: False drops items, tanks and missiles from hand-made places


def location(memory, settings):
    """Which room and roughly where in it."""
    return (area_of(memory), int(memory[MAP_Y]), int(memory[MAP_X]),
            int(memory[ROOM_X]) // settings.block_size, int(memory[ROOM_Y]) // settings.block_size)


def hand_made_place(memory, settings):
    """Where (see location) and progress counters."""
    if not settings.hand_inventory:
        return location(memory, settings)
    return (*location(memory, settings),
            int(memory[at(ITEMS)]), int(memory[at(ENERGY_TANKS)]), int(memory[at(MAX_MISSILES)]) // 5)


def screen_cell(gray, width, height, levels):
    """gray (brightness 0-1, see Emulator.gray) averaged over a width x height grid, each cell cut to levels steps."""
    rows, columns = gray.shape[0] // height, gray.shape[1] // width
    small = gray[:rows * height, :columns * width].reshape(height, rows, width, columns).mean(axis=(1, 3))
    return np.minimum(small * levels, levels - 1).astype(np.uint8).tobytes()


def screen_scheme(settings):
    """(width, height, levels, cells of context) of a screen place scheme."""
    width, height, levels, *context = (int(v) for v in settings.place_scheme.removeprefix("screen:").split(","))
    return width, height, levels, (context or [0])[0]


def screen_place(history, gray, settings):
    """A screen place, and the history of different cells it extends (see Settings.place_scheme)."""
    width, height, levels, context = screen_scheme(settings)
    cell = screen_cell(gray, width, height, levels)
    history = history if history and history[-1] == cell else (history + (cell,))[-(context + 1):]
    return ("screen", history), history


def uses_picture(settings):
    return settings.place_scheme.startswith("screen:") or settings.pixel_scroll


PIXEL_SHIFT_MOST, STATUS_ROWS, POOR_MATCH = 16, 40, 0.5  # status bars usually sit in the top rows; they don't scroll


def profile_shift(a, b):
    """(shift in pixels that best lines up profile b with profile a, how poor that best match is: 0 perfect, 1 none)."""
    a, b = a - a.mean(), b - b.mean()
    errors = [np.mean((a[max(0, d):len(a) + min(0, d)] - b[max(0, -d):len(b) + min(0, -d)]) ** 2)
              for d in range(-PIXEL_SHIFT_MOST, PIXEL_SHIFT_MOST + 1)]
    best = int(np.argmin(errors))
    return best - PIXEL_SHIFT_MOST, errors[best] / max(np.mean(a ** 2) + np.mean(b ** 2), 1e-9)


def pixel_move(previous_gray, gray):
    """How far the background moved between two frames (x, y), or (0, 0) for a cut (no good match)."""
    old, new = previous_gray[STATUS_ROWS:], gray[STATUS_ROWS:]
    (dx, poor_x), (dy, poor_y) = profile_shift(old.mean(axis=0), new.mean(axis=0)), profile_shift(old.mean(axis=1), new.mean(axis=1))
    return (0, 0) if max(poor_x, poor_y) > POOR_MATCH else (dx, dy)


SCROLL_RANGE = (512, 480)  # the picture chip scrolls over two name tables each way: 2 x 256 wide, 2 x 240 tall


def scroll_block(settings):
    return int(settings.place_scheme.removeprefix("scroll:").split(",")[0])


def player_step(settings):
    """P of a "scroll:B,P" scheme, or 0 when places don't include the player."""
    parts = settings.place_scheme.removeprefix("scroll:").split(",")
    return int(parts[1]) if len(parts) > 1 else 0


def player_values(emulator, state, memory, settings):
    """The player's position values: the player bytes, or with player_y_sprite (x byte, y of the nearest sprite)."""
    values = [int(memory[int(b)]) for b in settings.player_bytes.split(",") if b]
    if settings.player_y_sprite and values:
        sprites = np.frombuffer(emulator.block(state, "SPRT"), np.uint8).reshape(64, 4)
        sprites = sprites[sprites[:, 0] < 0xEF]
        screen_x = (values[0] - emulator.scroll(state)[0]) % 256 if settings.player_x_scrolled else values[0]
        height = int(sprites[np.abs(sprites[:, 3].astype(int) - screen_x).argmin(), 0]) if len(sprites) else 0
        values = [values[0], height]
    return tuple(values)


COMPASS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def world_position(camera, scroll, memory, settings):
    """The player's position in the world: the dead-reckoned camera plus where the player is on screen."""
    on_screen = [int(memory[int(b)]) for b in settings.player_bytes.split(",") if b][:2]
    if settings.player_x_scrolled:
        on_screen = [(v - s) % r for v, s, r in zip(on_screen, scroll, (256, 240))]
    return tuple(c + o for c, o in zip(camera, on_screen + [0] * (2 - len(on_screen))))


def player_part(values, settings):
    step = player_step(settings)
    return tuple(v // step for v in values) if step else ()


def travelled(start_place, scroll, settings):
    """Where dead reckoning starts a try: the exact position its start place stands for (see Settings.place_scheme)."""
    block = scroll_block(settings)
    if start_place[0] == "level":
        block, start_place = block << start_place[1], start_place[1:]
    elif start_place[0] != "scroll":
        return scroll
    return tuple(b * block + s % block for b, s in zip(start_place[1:], scroll))


CUT = 16  # pixels: a camera move bigger than this in one step is a jump (a new room loading), not travel


def moved(position, previous_scroll, scroll, block):
    """position plus the camera's move between two scroll readings, each axis taken the short way round. A jump
    (see CUT) isn't travel: it moves position only by the jump's remainder modulo block, which keeps position and scroll
    in step modulo block (travelled relies on that) and keeps position continuous across a room change."""
    def step(p, q, s, r):
        move = (s - q + r // 2) % r - r // 2
        return p + (move if abs(move) <= CUT else (move + block // 2) % block - block // 2)
    return tuple(step(p, q, s, r) for p, q, s, r in zip(position, previous_scroll, scroll, SCROLL_RANGE))


def scroll_place(position, settings):
    block = scroll_block(settings)
    return ("scroll",) + tuple(p // block for p in position)


def place_of_state(emulator, state, settings):
    """The place of a stored state with no history; a screen scheme draws one more frame from it to see the picture."""
    if settings.place_scheme.startswith("scroll:"):
        extras = Extras(emulator, settings, Path(settings.reference_state).read_bytes() if settings.reference_state else state)
        return (scroll_place(emulator.scroll(state), settings) + player_part(player_values(emulator, state, emulator.view(state), settings), settings)
                + extras.part(state, emulator.view(state), extras.start(("start",))))
    if not uses_picture(settings):
        return hand_made_place(emulator.view(state), settings)
    emulator.load(state)
    emulator.run(0, 1, picture=True)
    return screen_place((), emulator.gray(), settings)[0]


def room_of(place):
    """The coarse part of a place: what record places (novel_values) and layer rules count as one room."""
    if place[0] == "record":
        return place[1]
    if place[0] == "compass":
        return place[:3]
    return place if place[0] in ("screen", "scroll", "level", "radius", "split", "group") else tuple(place[:3])


@dataclass
class Segment:
    """What one stretch of random buttons found: {place: (state, path length from the start, holdings)} for
    places that were new, or reached by a visit no kept visit beats (see beats)."""
    start: int  # archive index it started from
    found: dict
    progress: Progress
    frames: int
    gaps: list  # for each new place, steps since the previous new place (or since the segment began)
    actions: bytes  # index into controls.actions(Settings.turbo) for every step, so it can be replayed exactly
    start_checksum: int  # zlib.crc32 of the start state; a replay is only valid from a state that matches
    died: bool = False  # ended in game over


def holdings(memory, settings):
    """What a state holds of each hoarded byte (see keep_more), divided by its scale."""
    return tuple(int(memory[int(index)]) / float(scale or 1)
                 for index, _, scale in (item.partition(":") for item in settings.keep_more.split(",") if item and item != "none"))


def beats(a, b):
    """
    Whether visit a = (holdings, path length) makes visit b unnecessary: a holds at least as much of
    everything, and either more of something or the same with a path no longer.
    """
    (held_a, length_a), (held_b, length_b) = a, b
    return all(x >= y for x, y in zip(held_a, held_b)) and (held_a != held_b or length_a <= length_b)


RARE_CHANGE_RATE = 0.002  # bytes changing on fewer steps than this count as rarely changing (see novel_values)
WARMUP_STEPS = 2000  # steps a worker watches before it trusts its change rates
_change_counts, _steps_watched = None, 0  # per worker process: how often each memory byte changed
_records = {}  # per worker process: (room, byte) -> (lowest, highest, went down, went up, records), for novel_values
PROGRESS_RECORDS = 3  # one-way records a byte must set in a room before its records count (junk bytes stop sooner)


_layouts = {}  # per worker process: background layout checksum -> (position, scroll) when first seen, for Settings.stitch


def stitched(position, scroll, nametable, block):
    """Where position belongs after a camera jump: where this layout was first seen, moved by the scroll since then,
    kept in step with scroll modulo block (travelled relies on that)."""
    key = zlib.crc32(nametable)
    if key not in _layouts:
        _layouts[key] = (position, scroll)
        return position
    seen_position, seen_scroll = _layouts[key]
    from_seen = moved(seen_position, seen_scroll, scroll, block)
    # The position nearest the remembered one that has the same remainder modulo block as the current one.
    return tuple(f + (p - f + block // 2) % block - block // 2 for f, p in zip(from_seen, position))


_moves, _group_of = set(), {}  # per worker process: block-to-block moves seen, and union-find parents of blocks


def group(block):
    while _group_of.get(block, block) != block:
        _group_of[block] = _group_of.get(_group_of[block], _group_of[block])
        block = _group_of[block]
    return block


def saw_move(a, b):
    """Records a move between blocks; once both directions have been seen, the two blocks' groups merge."""
    if a != b:
        _moves.add((a, b))
        if (b, a) in _moves:
            ra, rb = group(a), group(b)
            if ra != rb:
                _group_of[max(ra, rb)] = min(ra, rb)


SPLIT_LEVELS = 4  # a split place's block is the scheme's block times 1, 2, 4 or 8
_crowding = {}  # per worker process: steps landed in each split cell (level, coordinates), for Settings.place_split


def split_cell(position, values, settings, level):
    player = tuple(v // (player_step(settings) << level) for v in values)
    return ("split", level) + tuple(p // (scroll_block(settings) << level) for p in position) + player


def grids(settings):
    """The (scroll block, player step) of each grid a save is filed under (see place_grids and place_levels)."""
    if settings.place_grids:
        return [tuple(int(v) for v in grid.split("x")) for grid in settings.place_grids.split(",")]
    return [(scroll_block(settings) << level, player_step(settings) << (level if settings.level_player else 0))
            for level in range(settings.place_levels)]


def ancestors(place, extra):
    """The coarser cells (level, coordinates) a split place sits inside, itself included."""
    level, cell = place[1], place[2:len(place) - len(extra)]
    return [(level + up,) + tuple(c >> up for c in cell) for up in range(SPLIT_LEVELS - level)]


def split_place(position, values, settings, crowding):
    """The finest cell, coarse to fine, whose parent has earned a split (play landed there Settings.place_split times)."""
    level = SPLIT_LEVELS - 1
    while level > 0 and crowding.get(split_cell(position, values, settings, level)[1:], 0) >= settings.place_split:
        level -= 1
    return split_cell(position, values, settings, level)


def nearest_center(feature, centers, radius):
    """Index of the nearest center within radius (L-infinity), or None. centers: [n, d] array."""
    if not len(centers):
        return None
    distance = np.abs(centers - feature).max(axis=1)
    best = int(distance.argmin())
    return best if distance[best] <= radius else None


def explore(emulator, rng, settings, known, start, start_state, start_path_length, start_place, start_position=None):
    """known: {place: [(holdings, path length) of every visit the archive kept there]}."""
    global _change_counts, _steps_watched, _records
    died = False
    emulator.load(start_state)
    stay = {tuple(int(v) for v in cell.split(",")) for cell in settings.stay.split(";") if cell}
    rare_values, changed = [], None
    found, progress, gaps = {}, Progress(), []
    previous_memory = None
    buttons = actions(settings.turbo)
    action = rng.integers(len(buttons))
    stale = 0
    history = start_place[1] if start_place[0] == "screen" else ()
    scrolling = settings.place_scheme.startswith("scroll:")
    if scrolling:
        previous_scroll = emulator.scroll(start_state)
        position = start_position if start_position is not None else travelled(start_place, previous_scroll, settings)
        if settings.place_radius:
            width = 2 + len([b for b in settings.player_bytes.split(",") if b])  # scroll x, y, then the player bytes
            keys = list(dict.fromkeys(p[:1 + width] for p in known if p[0] == "radius"))
            centers = np.array([k[1:] for k in keys], np.int64).reshape(len(keys), width)
        carry = _extras(emulator, settings).start(start_place)
        last_block = tuple(p // 32 for p in position)
        previous_gray = None
    pressed = bytearray()
    repeat_probability = settings.repeat_probability - rng.random() * settings.repeat_spread
    for step in range(settings.steps_per_segment):
        if rng.random() > repeat_probability:
            action = rng.integers(len(buttons))
        emulator.run(buttons[action], FRAMES_PER_STEP, picture=uses_picture(settings))
        pressed.append(action)
        state = emulator.save()
        memory = emulator.view(state)
        full_memory = emulator.memory(state)
        if previous_memory is not None:
            progress.see_change(previous_memory, full_memory)
            if settings.novel_values:
                changed_before = changed if changed is not None else np.zeros(len(full_memory), bool)
                changed = full_memory != previous_memory
                _change_counts = changed.astype(np.int64) if _change_counts is None else _change_counts + changed
                _steps_watched += 1
                if _steps_watched > WARMUP_STEPS:
                    rare = np.flatnonzero(changed & (_change_counts < RARE_CHANGE_RATE * _steps_watched))
                    # Only isolated changes: a hit lowers hit points in one jump; moving things change step after step.
                    rare_values = [(int(b), int(full_memory[b])) for b in rare if not changed_before[b]]
        previous_memory = full_memory
        if memory[GAME_MODE] == GAME_OVER or (stay and (int(memory[MAP_Y]), int(memory[MAP_X])) not in stay):
            died = True
            break
        if scrolling:
            if settings.pixel_scroll:
                gray = emulator.gray(1)
                move = pixel_move(previous_gray, gray) if previous_gray is not None else (0, 0)
                position, previous_gray = tuple(p + m for p, m in zip(position, move)), gray
            scroll = emulator.scroll(state)
            jump = settings.stitch and any(min(abs(a - b), r - abs(a - b)) > CUT for a, b, r in zip(previous_scroll, scroll, SCROLL_RANGE))
            if not settings.pixel_scroll:
                position = moved(position, previous_scroll, scroll, scroll_block(settings))
            previous_scroll = scroll
            if jump:
                position = stitched(position, scroll, emulator.nametable(state), scroll_block(settings))
            probe_block = tuple(p // 32 for p in position)  # probe add-ons rerun only when this coarse block changes
            extra = _extras(emulator, settings).part(state, memory, carry, probe_block, position)
            values = player_values(emulator, state, memory, settings)
            if settings.player_world:
                values, at = (), world_position(position, emulator.scroll(state), memory, settings)
            else:
                at = position
            place = scroll_place(at, settings) + player_part(values, settings) + extra
            if settings.place_groups:
                block = tuple(p // 32 for p in position)
                saw_move(last_block, block)
                last_block = block
                place = ("group", group(block)) + player_part(values, settings) + extra
            if settings.place_split:
                place = split_place(position, values, settings, _crowding) + extra
                for ancestor in ancestors(place, extra):
                    _crowding[ancestor] = _crowding.get(ancestor, 0) + 1
            if settings.place_radius:
                feature = np.array(tuple(position) + values, np.int64)
                nearest = nearest_center(feature, centers, settings.place_radius)
                if nearest is None:
                    keys.append(("radius",) + tuple(int(v) for v in feature))
                    centers = np.vstack([centers, feature])
                    nearest = len(keys) - 1
                place = keys[nearest] + extra
            levels = [("level", level) + tuple(p // block for p in at) + tuple(v // step for v in values) + extra
                      for level, (block, step) in enumerate(grids(settings))] if len(grids(settings)) > 1 else [place]
        elif uses_picture(settings):
            place, history = screen_place(history, emulator.gray(), settings)
        else:
            place = hand_made_place(memory, settings)
        place += (zlib.crc32(emulator.nametable(state)),) if settings.place_nametable else ()
        length = start_path_length + step + 1
        held = holdings(memory, settings)
        offers = [(offered, held) for offered in (levels if scrolling else [place])]
        if scrolling and settings.compass:
            world = world_position(position, emulator.scroll(state), memory, settings)
            room = tuple(w // 256 for w in world)
            offers = [(("compass",) + room + (d,), held + (dx * world[0] + dy * world[1],)) for d, (dx, dy) in enumerate(COMPASS)]
        new = False
        for offered, offered_held in offers:
            new = new or (offered not in known and offered not in found)
            rivals = known.get(offered, []) + ([(found[offered][2], found[offered][1])] if offered in found else [])
            if not any(beats(rival, (offered_held, length)) for rival in rivals):
                found[offered] = (state, length, offered_held, tuple(position) if scrolling else None)
        if new:
            gaps.append(stale + 1)
            stale = 0
        else:
            stale += 1
            if settings.stale_steps is not None and stale >= settings.stale_steps:
                break
        if settings.novel_values:
            # Records need a room-sized grouping to pile up (PROGRESS_RECORDS one-way records per room). For dead-reckoning
            # places that's the screen-sized block of the position, not the (much finer) place itself.
            room = ("screen",) + tuple(p // 256 for p in position) if scrolling and settings.screen_rooms else room_of(place)
            for b, value in rare_values:
                low, high, went_down, went_up, count = _records.get((room, b), (value, value, False, False, 0))
                record = value < low or value > high
                went_down, went_up = went_down or value < low, went_up or value > high
                count += record
                # A save matching the record is offered too, under the same place, so the keep rule (beats) swaps in
                # one that got there holding more: the cheapest way to each level of progress.
                at_record = value <= low if went_down else value >= high
                if at_record and went_down != went_up and count >= PROGRESS_RECORDS:
                    value_place = ("record", room, b, value)
                    rivals = known.get(value_place, []) + ([(found[value_place][2], found[value_place][1])] if value_place in found else [])
                    if not any(beats(rival, (held, length)) for rival in rivals):
                        found[value_place] = (state, length, held, tuple(position) if scrolling else None)
                _records[(room, b)] = (min(low, value), max(high, value), went_down, went_up, count)
            rare_values = []
        if step % PROGRESS_SAMPLE_EVERY == 0:
            progress.see(emulator.memory(state), emulator.nametable(state))
    probe_frames = carry["probe_frames"] if scrolling else 0  # probes are play too: they count toward the budget
    return Segment(start, found, progress, (step + 1) * FRAMES_PER_STEP + probe_frames, gaps, bytes(pressed), zlib.crc32(start_state),
                   died)


_worker_extras = None  # per worker process: the place add-ons (they precompute weights once)


def _extras(emulator, settings):
    global _worker_extras
    if _worker_extras is None:
        reference = Path(settings.reference_state).read_bytes() if settings.reference_state else emulator.save()
        _worker_extras = Extras(emulator, settings, reference)
    return _worker_extras


def worker(connection, settings, seed, index):
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    rng = np.random.default_rng([seed, index])  # by worker index, not pid, so a run can be reproduced exactly
    known = {}
    while (task := connection.recv()) is not None:
        start, start_state, start_path_length, start_place, start_position, updates = task
        for place, visit in updates:
            known.setdefault(place, []).append(visit)
        connection.send(explore(emulator, rng, settings, known, start, start_state, start_path_length, start_place, start_position))


class Explorers:
    def __init__(self, count, settings, seed):
        context = get_context("spawn")
        pipes = [context.Pipe() for _ in range(count)]
        self._connections = [main_end for main_end, _ in pipes]
        self._processes = [context.Process(target=worker, args=(worker_end, settings, seed, index), daemon=True)
                           for index, (_, worker_end) in enumerate(pipes)]
        for process in self._processes:
            process.start()
        self._updates_sent = [0] * count
        self.count = count

    def send(self, worker_index, start, start_state, start_path_length, start_place, start_position, updates):
        """updates is the main process's full, append-only log of (place, (holdings, path length)) updates."""
        self._connections[worker_index].send((start, start_state, start_path_length, start_place, start_position,
                                              updates[self._updates_sent[worker_index]:]))
        self._updates_sent[worker_index] = len(updates)

    def finished(self):
        """Blocks until at least one worker is done. Returns [(worker index, Segment)]."""
        ready = wait(self._connections)
        return [(self._connections.index(connection), connection.recv()) for connection in ready]

    def close(self):
        """Ends the workers at once: their unfinished tries are thrown away, and waiting for each to finish cost
        about a minute per run."""
        for process in self._processes:
            process.terminate()
        for process in self._processes:
            process.join()
