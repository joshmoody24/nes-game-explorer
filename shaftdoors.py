from kdoors import parse, rom, grid, kinds
from bdoors import room_offset
for cell in [(y, 6) for y in range(13, 21)] + [(18, 7), (18, 8), (19, 7), (14, 7), (14, 8), (13, 7), (15, 7), (17, 7), (18, 5), (14, 5)]:
    n = int(grid[cell])
    if n == 0xFF: print(cell, "no room"); continue
    room = parse(room_offset(0x6324, 0x3FF0, n, 0x0000))
    print(cell, f"room {n:#x}:", "doors", [("right" if d >> 4 == 0xA else "left", kinds.get(d & 0xF, d & 0xF)) for d in room[1]] if room else "unparsed")
