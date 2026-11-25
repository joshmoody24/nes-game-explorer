"""
How far exploration has got. The generic measures work for any NES game; the Metroid ones read
documented addresses and exist only to check that the generic ones aren't misleading us. None of
this is ever shown to the agent.
"""

from dataclasses import dataclass, field

import numpy as np

CARTRIDGE = 0x800 - 0x6000  # memory arrays hold cartridge RAM ($6000-) right after the 2 KB system RAM
ITEM_BITS = {0x01: "bombs", 0x02: "high jump", 0x04: "long beam", 0x08: "screw attack",
             0x10: "morph ball", 0x20: "varia", 0x40: "wave beam", 0x80: "ice beam"}
AREAS = {0x00: "Brinstar", 0x10: "Brinstar", 0x11: "Norfair", 0x12: "Kraid", 0x13: "Tourian", 0x14: "Ridley"}
PLAYING = 3


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
        if memory[0x1E] != PLAYING:
            return
        self.rooms.add((memory[0x74], memory[0x4F], memory[0x50]))
        self.areas.add(AREAS.get(memory[0x74], hex(memory[0x74])))
        self.items |= int(memory[0x6878 + CARTRIDGE])
        self.max_tanks = max(self.max_tanks, int(memory[0x6877 + CARTRIDGE]))
        self.max_missiles = max(self.max_missiles, int(memory[0x687A + CARTRIDGE]))
        self.statues |= ((memory[0x687B + CARTRIDGE] >> 7) & 1) | (((memory[0x687C + CARTRIDGE] >> 7) & 1) << 1)

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
