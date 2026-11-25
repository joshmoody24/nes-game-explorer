"""
How far exploration has got. The generic measures work for any NES game; the Metroid ones read
documented addresses and exist only to check that the generic ones aren't misleading us. None of
this is ever shown to the agent.
"""

from dataclasses import dataclass, field

import numpy as np

from metroid import (AREA, ENERGY_TANKS, GAME_MODE, ITEMS, KRAID_STATUE, MAP_X, MAP_Y, MAX_MISSILES, PLAYING,
                     RIDLEY_STATUE, at)

ITEM_BITS = {0x01: "bombs", 0x02: "high jump", 0x04: "long beam", 0x08: "screw attack",
             0x10: "morph ball", 0x20: "varia", 0x40: "wave beam", 0x80: "ice beam"}
AREAS = {0x00: "Brinstar", 0x10: "Brinstar", 0x11: "Norfair", 0x12: "Kraid", 0x13: "Tourian", 0x14: "Ridley"}


@dataclass
class Progress:
    """
    Accumulates what has ever been seen. Feed it states with see(), and pairs of memories a few steps
    apart within one run with see_change(). Pairs must be consecutive in time: comparing everything to
    one shared start state would make every bit that moves away from it look one-way.
    """
    layouts: set = field(default_factory=set)
    ever_up: np.ndarray = None  # bits ever seen going 0 -> 1 between a before and an after
    ever_down: np.ndarray = None
    rooms: set = field(default_factory=set)
    areas: set = field(default_factory=set)
    items: int = 0
    max_tanks: int = 0
    max_missiles: int = 0
    statues: int = 0  # bit 0 Kraid, bit 1 Ridley: statue raised, i.e. boss defeated

    def see_change(self, before, after):
        before, after = np.unpackbits(before), np.unpackbits(after)
        if self.ever_up is None:
            self.ever_up = np.zeros(len(before), dtype=bool)
            self.ever_down = np.zeros(len(before), dtype=bool)
        self.ever_up |= after > before
        self.ever_down |= after < before

    def see(self, memory, nametable):
        self.layouts.add(hash(nametable))
        if memory[GAME_MODE] != PLAYING:
            return
        self.rooms.add((int(memory[AREA]), int(memory[MAP_Y]), int(memory[MAP_X])))
        self.areas.add(AREAS.get(int(memory[AREA]), hex(memory[AREA])))
        self.items |= int(memory[at(ITEMS)])
        self.max_tanks = max(self.max_tanks, int(memory[at(ENERGY_TANKS)]))
        self.max_missiles = max(self.max_missiles, int(memory[at(MAX_MISSILES)]))
        self.statues |= ((memory[at(KRAID_STATUE)] >> 7) & 1) | (((memory[at(RIDLEY_STATUE)] >> 7) & 1) << 1)

    def merge(self, other):
        """Folds in what another Progress (e.g. from a worker process) has seen."""
        self.layouts |= other.layouts
        if other.ever_up is not None:
            self.ever_up = other.ever_up if self.ever_up is None else self.ever_up | other.ever_up
            self.ever_down = other.ever_down if self.ever_down is None else self.ever_down | other.ever_down
        self.rooms |= other.rooms
        self.areas |= other.areas
        self.items |= other.items
        self.max_tanks = max(self.max_tanks, other.max_tanks)
        self.max_missiles = max(self.max_missiles, other.max_missiles)
        self.statues |= other.statues

    def summary(self):
        return {
            "background layouts": len(self.layouts),
            "one-way bits": int((self.ever_up ^ self.ever_down).sum()) if self.ever_up is not None else 0,
            "rooms": len(self.rooms),
            "areas": sorted(self.areas),
            "items": [name for bit, name in ITEM_BITS.items() if self.items & bit],
            "energy tanks": self.max_tanks,
            "max missiles": self.max_missiles,
            "bosses": [name for bit, name in [(1, "Kraid"), (2, "Ridley")] if self.statues & bit],
        }
