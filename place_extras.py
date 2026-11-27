"""
Game-agnostic add-ons to scroll places (Settings.place_extras), one experiment each. A place becomes
(scroll place, player part, extra 1, extra 2, ...). Anything an add-on needs to learn about the game first (which bytes
to add, which sprite tiles are always on screen) comes from a short random wander at the start of a run (see pick).

  bytes  the values of chosen RAM bytes (Settings.extra_bytes), picked by Settings.byte_picker:
           sticky  bytes holding bits that get set and never cleared along the wander (items, flags)
           cuts    bytes that change mostly when the camera jumps (a room change): a room number (per-game file
                   cache/cut_bytes.json from long wanders over many saves; a run's own start rarely crosses a door)
           poke    bytes that change the future when altered, and change rarely (per-game file cache/poke_share.npy
                   from checks/poke_check.py: ~12M frames once per game)
  cast   which sprite tiles are on screen, minus the ones that almost always are (the player's): who's in the room
  area   a checksum of the loaded tile graphics and the mapper's bank registers: the area's hardware fingerprint
  fuzzy  8 bits of a random-projection hash of all RAM minus the run's start RAM: near-equal memories collide
  burst  which tiles fill the background just after the last camera jump: what the game drew for this room
  patch  the background tiles around the player (player_bytes as screen x, y): the local landmark
Probe add-ons run short button tests from the current save whenever the scroll block changes (results carry forward
until the next change); every probe frame counts toward the run's frame budget (carry["probe_frames"]):
  afford what the buttons do here: for each of 6 held presses (left, right, up, down, jump, fire; 20 frames), whether
         the player or camera moved and whether the camera jumped. Like tapping around with a cane.
  reach  which 64-pixel blocks 4 short random walks (60 frames) end in: two saves in one room reach the same blocks
  oneway how many camera jumps so far were one-way: after a jump, 6 random walks (60 frames) never bring the camera
         back near where it was
"""
import json, zlib
from pathlib import Path

import numpy as np

from controls import FRAMES_PER_STEP, actions
from emulator import button_mask

CUT = 16
FUZZY_BITS = 4
WANDER_WALKS, WANDER_STEPS = 20, 150


def oam(emulator, state):
    sprites = np.frombuffer(emulator.block(state, "SPRT"), np.uint8).reshape(64, 4)
    return sprites[sprites[:, 0] < 0xEF]


def tile_bag(nametable):
    cells = np.frombuffer(nametable, np.uint8)[np.r_[0:960, 1024:1984]]
    return frozenset(np.flatnonzero(np.bincount(cells, minlength=256) >= 32).tolist())


def jumped(previous_scroll, scroll):
    return previous_scroll is not None and any(min(abs(a - b), r - abs(a - b)) > CUT
                                               for a, b, r in zip(previous_scroll, scroll, (512, 480)))


def wander(emulator, start_state, rng, turbo):
    """[(memory, scroll, visible sprites)] per step of WANDER_WALKS random walks from the start."""
    buttons, walks = actions(turbo), []
    for _ in range(WANDER_WALKS):
        emulator.load(start_state)
        action, walk = rng.integers(len(buttons)), []
        for _ in range(WANDER_STEPS):
            if rng.random() > 0.95:
                action = rng.integers(len(buttons))
            emulator.run(buttons[action], FRAMES_PER_STEP)
            state = emulator.save()
            walk.append((emulator.memory(state), emulator.scroll(state), oam(emulator, state)))
        walks.append(walk)
    return walks


def sticky_bytes(walks, most=16):
    set_ever, cleared_ever = 0, 0
    for walk in walks:
        for (a, _, _), (b, _, _) in zip(walk, walk[1:]):
            set_ever = set_ever | (~a & b)
            cleared_ever = cleared_ever | (a & ~b)
    sticky = np.flatnonzero((set_ever & ~cleared_ever) != 0)
    return sticky[:most].tolist()


def cut_bytes(walks, most=8):
    at_cut, elsewhere, cuts, steps = 0, 0, 0, 0
    for walk in walks:
        for (a, sa, _), (b, sb, _) in zip(walk, walk[1:]):
            changed = (a != b).astype(np.int64)
            if jumped(sa, sb):
                at_cut, cuts = at_cut + changed, cuts + 1
            else:
                elsewhere, steps = elsewhere + changed, steps + 1
    if not cuts:
        return []
    ratio = (at_cut / cuts) / (elsewhere / max(steps, 1) + 1e-3)
    good = [int(b) for b in np.argsort(-ratio) if at_cut[b] / cuts >= 0.5]
    return good[:most]


def poke_bytes(walks, share, most=16):
    """Rarely-changing bytes that alter the future in 5%+ of poked saves (95% of RAM never does), most effective first."""
    changes = sum((a != b).astype(np.int64) for walk in walks for (a, _, _), (b, _, _) in zip(walk, walk[1:]))
    rate = changes / sum(len(walk) - 1 for walk in walks)
    order = np.lexsort((rate, -share))  # matters most first; ties broken by changing least
    candidates = [int(b) for b in order if share[b] >= 0.05 and rate[b] < 0.01]
    return candidates[:most]


def common_tiles(walks, share=0.5):
    counts = {}
    for walk in walks:
        for _, _, sprites in walk:
            for tile in set(sprites[:, 1].tolist()):
                counts[tile] = counts.get(tile, 0) + 1
    steps = sum(len(walk) for walk in walks)
    return sorted(t for t, n in counts.items() if n >= share * steps)


def pick(emulator, start_state, rng, settings, poke_share=None):
    """What the chosen add-ons need, as Settings fields: {"extra_bytes": ..., "common_tiles": ...}."""
    extras = settings.place_extras.split(",")
    walks = wander(emulator, start_state, rng, settings.turbo)
    found = {}
    if "bytes" in extras:
        chosen = {"sticky": lambda: sticky_bytes(walks), "cuts": lambda: json.loads(Path("cache/cut_bytes.json").read_text()),
                  "poke": lambda: poke_bytes(walks, poke_share)}[settings.byte_picker]()
        found["extra_bytes"] = ",".join(map(str, chosen))
    if "cast" in extras:
        found["common_tiles"] = ",".join(map(str, common_tiles(walks)))
    return found


MACROS = [button_mask(b) for b in ("LEFT", "RIGHT", "UP", "DOWN", "A", "B")]
PROBE_STEPS = 5  # 20 frames
WALK_STEPS, WALKS, RETURN_WALKS = 15, 4, 6


def _fuzzy_weights(length):
    return np.random.default_rng(12345).choice([-1, 1], (8, length)).astype(np.int32)


class Extras:
    """Computes the add-on part of places for one worker. carry holds what persists along a try (the burst signature)."""

    def __init__(self, emulator, settings, reference_state):
        self.emulator, self.settings = emulator, settings
        self.names = [n for n in settings.place_extras.split(",") if n]
        self.extra_bytes = [int(b) for b in settings.extra_bytes.split(",") if b]
        self.common = {int(t) for t in settings.common_tiles.split(",") if t}
        self.player = [int(b) for b in settings.player_bytes.split(",") if b]
        self.buttons = actions(settings.turbo)
        if "fuzzy" in self.names:
            self.reference = emulator.memory(reference_state).astype(np.int32)
            self.weights = _fuzzy_weights(len(self.reference))

    def start(self, start_place):
        """Carry for a try from its start place (the burst signature it stood in)."""
        stored = lambda name: start_place[len(start_place) - len(self.names) + self.names.index(name)] \
            if name in self.names and start_place[0] == "scroll" else None
        return {"burst": stored("burst"), "oneway": stored("oneway") or 0, "scroll": None, "block": None, "probed": {},
                "probe_frames": 0}

    def part(self, state, memory, carry, block=None, position=None):
        """block: the scroll place's block (probes rerun when it changes); position: the exact dead-reckoned position."""
        out, scroll = [], self.emulator.scroll(state)
        cut = jumped(carry["scroll"], scroll)
        if "burst" in self.names and (carry["burst"] is None or cut):
            carry["burst"] = zlib.crc32(repr(sorted(tile_bag(self.emulator.nametable(state)))).encode())
        if "oneway" in self.names and cut and not self._can_return(state, carry["scroll"], carry):
            carry["oneway"] += 1
        if block != carry["block"] or not carry["probed"]:
            carry["block"], carry["probed"] = block, {}
            if "afford" in self.names:
                carry["probed"]["afford"] = self._afford(state, memory, scroll, carry)
            if "reach" in self.names:
                carry["probed"]["reach"] = self._reach(state, scroll, position, carry)
        carry["scroll"] = scroll
        for name in self.names:
            if name == "bytes":
                out.append(tuple(int(memory[b]) for b in self.extra_bytes))
            elif name == "cast":
                # Tiles in groups of 16: an enemy's animation frames usually sit together, so they don't split places.
                out.append(tuple(sorted({t >> 4 for t in set(oam(self.emulator, state)[:, 1].tolist()) - self.common})))
            elif name == "area":
                out.append(zlib.crc32(self.emulator.block(state, "CHRR") + self.emulator.block(state, "MAPR")))
            elif name == "fuzzy":
                ram = self.emulator.memory(state).astype(np.int32)
                out.append(int(np.packbits(self.weights[:FUZZY_BITS] @ (ram - self.reference) > 0)[0]))
            elif name == "burst":
                out.append(carry["burst"])
            elif name == "patch":
                out.append(self._patch(state, memory, scroll))
            elif name in ("afford", "reach"):
                out.append(carry["probed"][name])
            elif name == "oneway":
                out.append(carry["oneway"])
        return tuple(out)

    def _run(self, buttons, steps, carry):
        for _ in range(steps):
            self.emulator.run(buttons, FRAMES_PER_STEP)
        carry["probe_frames"] += steps * FRAMES_PER_STEP

    def _afford(self, state, memory, scroll, carry):
        bits = []
        before = [int(memory[b]) for b in self.player]
        for buttons in MACROS:
            self.emulator.load(state)
            self._run(buttons, PROBE_STEPS, carry)
            after = self.emulator.save()
            view = self.emulator.view(after)
            moved = any(abs(int(view[b]) - v) > 2 for b, v in zip(self.player, before)) or self.emulator.scroll(after) != scroll
            bits += [moved, jumped(scroll, self.emulator.scroll(after))]
        self.emulator.load(state)
        return int(np.packbits(bits)[0]) << 8 | int(np.packbits(bits)[1])

    def _walks(self, state, count, carry):
        """Scroll readings along count short random walks; seeded by the state, so the same save probes the same way."""
        rng = np.random.default_rng(zlib.crc32(state[:4096]))
        for _ in range(count):
            self.emulator.load(state)
            readings = []
            for _ in range(WALK_STEPS):
                self._run(self.buttons[rng.integers(len(self.buttons))], 1, carry)
                readings.append(self.emulator.scroll(self.emulator.save()))
            yield readings
        self.emulator.load(state)

    def _reach(self, state, scroll, position, carry):
        from explorers import moved
        ends = set()
        for readings in self._walks(state, WALKS, carry):
            here, previous = position or scroll, scroll
            for reading in readings:
                here, previous = moved(here, previous, reading, 8), reading
            ends.add(tuple(p // 64 for p in here))
        return tuple(sorted(ends))

    def _can_return(self, state, before_jump, carry):
        for readings in self._walks(state, RETURN_WALKS, carry):
            if any(max(min(abs(a - b), r - abs(a - b)) for a, b, r in zip(reading, before_jump, (512, 480))) <= CUT
                   for reading in readings):
                return True
        return False

    def _patch(self, state, memory, scroll):
        if len(self.player) < 3:
            return None
        x, y = int(memory[self.player[0]]), int(memory[self.player[2]])
        table = np.frombuffer(self.emulator.nametable(state), np.uint8)
        column, row = ((scroll[0] + x) // 8) % 64, ((scroll[1] + y) // 8) % 60
        cells = []
        for dy in range(-2, 2):
            for dx in range(-2, 2):
                c, r = (column + dx) % 64, (row + dy) % 60
                # Which of the two name tables depends on the game's mirroring; either way it's one of the two.
                cells.append(int(table[((c // 32 + r // 30) % 2) * 1024 + (r % 30) * 32 + c % 32]))
        return zlib.crc32(bytes(cells))
