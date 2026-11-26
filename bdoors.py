from kdoors import parse, rom, grid, kinds
def room_offset(table, x, n, base):
    p = rom[table + 2 * n] | rom[table + 2 * n + 1] << 8
    return base | ((p - x) & 0xFFFF)
def find_table(lo, hi, x, base, run=40):
    best = (0, None)
    for t in range(lo, hi - 2 * run):
        rooms = [room_offset(t, x, n, base) for n in range(run)]
        if all(lo <= r < hi for r in rooms):
            ok = sum(parse(r) is not None for r in rooms)
            if ok > best[0]:
                best = (ok, t)
                if ok == run: break
    return best
if __name__ == "__main__":
    ok, t = find_table(0x4010, 0x8010, 0x3FF0, 0x0000)
    print("Brinstar room table", hex(t), ok, "/ 40 parse")
    for cell in ((14, 5), (14, 6), (14, 7), (13, 6), (15, 6)):
        n = int(grid[cell])
        if n == 0xFF: print(cell, "no room"); continue
        room = parse(room_offset(t, 0x3FF0, n, 0x0000))
        print(cell, f"room {n:#x}:", "doors", [("right" if d >> 4 == 0xA else "left", kinds.get(d & 0xF, d & 0xF)) for d in room[1]] if room else "unparsed")
