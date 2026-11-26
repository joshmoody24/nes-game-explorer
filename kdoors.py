from pathlib import Path
import numpy as np
from explorers import ROM
rom = Path(ROM).read_bytes()
grid = np.frombuffer(rom[0x254E:0x254E + 1024], np.uint8).reshape(32, 32)

def parse(offset):
    i, objects = offset + 1, 0
    if rom[offset] > 3: return None
    while rom[i] not in (0xFD, 0xFF):
        i += 3; objects += 1
        if objects > 80: return None
    doors, enemies = [], []
    if rom[i] == 0xFD:
        i += 1
        while rom[i] != 0xFF:
            if rom[i] == 0x02: doors.append(rom[i + 1]); i += 2
            else: enemies.append(tuple(rom[i:i + 3])); i += 3
            if len(enemies) > 20 or len(doors) > 6: return None
    return objects, doors, enemies

def room_offset(table_offset, x, n):
    p = rom[table_offset + 2 * n] | rom[table_offset + 2 * n + 1] << 8
    return 0x10000 | ((p - x) & 0xFFFF)

kinds = {0: "red (5 missiles)", 1: "blue", 2: "yellow/purple (10 missiles)"}
if __name__ == "__main__":
      for cell in ((27, 1), (27, 2), (27, 3), (23, 4), (22, 4), (23, 7), (22, 7), (23, 12), (22, 12)):
        n = int(grid[cell])
        if n == 0xFF: print(cell, "no room"); continue
        room = parse(room_offset(0x121E5, 0x7FF0, n))
        print(cell, f"room {n:#x}:", "doors", [("right" if d >> 4 == 0xA else "left", kinds.get(d & 0xF, d & 0xF)) for d in room[1]] if room else "unparsed")
