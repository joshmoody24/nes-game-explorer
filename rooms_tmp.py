import sys
from pathlib import Path

import numpy as np

from explorers import ROM

rom = Path(ROM).read_bytes()
grid = np.frombuffer(rom[0x254E:0x254E + 1024], np.uint8).reshape(32, 32)


def parse(offset):
    """(objects, doors, enemies, length) of the room data at a file offset, or None if it doesn't parse."""
    i, objects = offset + 1, 0
    if rom[offset] > 3:
        return None
    while rom[i] not in (0xFD, 0xFF):
        i += 3
        objects += 1
        if objects > 80:
            return None
    doors, enemies = [], []
    if rom[i] == 0xFD:
        i += 1
        while rom[i] != 0xFF:
            if rom[i] == 0x02:
                doors.append(rom[i + 1]); i += 2
            else:
                enemies.append(tuple(rom[i:i + 3])); i += 3
            if len(enemies) > 20 or len(doors) > 6:
                return None
    return objects, doors, enemies, i - offset


def table(table_offset, bank_base, x, count=60):
    """Room data offsets from a pointer table (pointers stored low byte first; file = pointer - x within the bank)."""
    rooms = []
    for n in range(count):
        p = rom[table_offset + 2 * n] | rom[table_offset + 2 * n + 1] << 8
        f = bank_base + ((p - x) & 0xFFFF) - (bank_base & 0xFFFF) + (bank_base & 0xFFFF)
        f = (bank_base & ~0xFFFF) | ((p - x) & 0xFFFF)
        rooms.append(f)
    return rooms


if __name__ == "__main__":
    kraid_room = int(grid[29, 8])
    print("map room number at Kraid's cell (29,8):", hex(kraid_room))
    kraid_rooms = table(0x121E5, 0x10000, 0x7FF0)
    print("Kraid room 0 at", hex(kraid_rooms[0]), "(page says 0x122C7)")
    print("Kraid's room data:", parse(kraid_rooms[kraid_room]))
    # Ridley: same in-bank table offset in its bank (pointer = file + 0x3FF0, bank digit 1)
    for guess in (0x161E5, 0x161E5 - 0x10, 0x161E5 + 0x10):
        rooms = table(guess, 0x10000, 0x3FF0)
        ok = sum(parse(r) is not None for r in rooms[:40])
        print("Ridley table guess", hex(guess), "parses", ok, "/ 40")


def find_table(bank_lo, bank_hi, x, run=40):
    best = (0, None)
    for t in range(bank_lo, bank_hi - 2 * run):
        rooms = table(t, bank_lo & ~0xFFFF if bank_lo >= 0x10000 else 0, x, run)
        if all(bank_lo <= r < bank_hi for r in rooms):
            ok = sum(parse(r) is not None for r in rooms)
            if ok > best[0]:
                best = (ok, t)
                if ok == run:
                    break
    return best


def bosses(t, bank_base, x, count=64):
    found = []
    for n, r in enumerate(table(t, bank_base, x, count)):
        room = parse(r) if bank_base <= r < bank_base + 0x10000 else None
        if room:
            for e in room[2]:
                if e[1] >> 4 == 4:
                    found.append((n, hex(e[1])))
    return found


if __name__ == "__main__":
    print("Kraid table by scan:", [hex(v) if isinstance(v, int) and v > 100 else v for v in find_table(0x10010, 0x14010, 0x7FF0)])
    ok, t = find_table(0x14010, 0x18010, 0x3FF0)
    print("Ridley table by scan:", ok, hex(t))
    ridley_bosses = bosses(t, 0x10000, 0x3FF0)
    print("Ridley-bank rooms with a boss-type enemy (room, type):", ridley_bosses)
    for room, kind in ridley_bosses:
        cells = [(y, x) for y in range(32) for x in range(32) if grid[y, x] == room and x >= 13 and y >= 20]
        print(f"  room {room:#x} ({kind}) at map cells (Ridley region) {cells}")
    print("Kraid-bank boss rooms:", bosses(0x121E5, 0x10000, 0x7FF0))


if __name__ == "__main__":
    rooms = table(0x1618F, 0x10000, 0x3FF0, 64)
    door_kind = {0: "red (5 missiles)", 1: "blue", 2: "yellow/purple (10 missiles)"}
    for cell in ((29, 15), (29, 16), (29, 17), (29, 18), (29, 19), (30, 17)):
        n = int(grid[cell])
        if n == 0xFF:
            print(cell, "no room"); continue
        objects, doors, enemies, _ = parse(rooms[n])
        print(cell, f"room {n:#x}: doors", [("right" if d >> 4 == 0xA else "left", door_kind.get(d & 0xF, d & 0xF)) for d in doors],
              "enemies", [hex(e[1]) for e in enemies])
