

if __name__ == "__main__":
    print("--- Kraid-area doors")
    krooms = table(0x121E5, 0x10000, 0x7FF0, 64)
    kinds = {0: "red (5 missiles)", 1: "blue", 2: "yellow/purple (10 missiles)"}
    for cell in ((27, 2), (27, 3), (23, 4), (22, 4), (23, 7), (22, 7), (23, 12), (22, 12)):
        n = int(grid[cell])
        if n == 0xFF: print(cell, "no room"); continue
        room = parse(krooms[n])
        print(cell, f"room {n:#x}:", "doors", [("right" if d >> 4 == 0xA else "left", kinds.get(d & 0xF, d & 0xF)) for d in room[1]] if room else "unparsed")
