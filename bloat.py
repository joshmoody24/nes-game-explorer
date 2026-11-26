import pickle, sys, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
if __name__ == "__main__":
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    places, chosen, productive = saved["places"], np.array(saved["times_chosen"]), np.array(saved.get("productive", [0] * len(saved["places"])))
    rec = np.array([p[0] == "record" for p in places])
    print(f"{len(places)} visits: {rec.sum()} record ({chosen[rec].sum() / chosen.sum():.0%} of choices), {(~rec).sum()} hand-made")
    per_place = Counter(tuple(p[:-2]) for p in places if p[0] != "record")
    sizes = sorted(per_place.values())
    print(f"hand-made: {len(per_place)} distinct places (ignoring held fields); versions per place median {sizes[len(sizes)//2]}, 90% {sizes[int(len(sizes)*0.9)]}, max {sizes[-1]}")
    room = lambda p: tuple(p[1][:3]) if p[0] == "record" else tuple(p[:3])
    by_room = defaultdict(lambda: [0, 0, 0])
    for p, c, k in zip(places, chosen, productive):
        r = by_room[room(p)]; r[0] += 1; r[1] += c; r[2] += k
    rows = sorted(by_room.items(), key=lambda kv: -kv[1][1])
    total_c, total_k = chosen.sum(), max(productive.sum(), 1)
    dead = [(r, v) for r, v in rows if v[2] / max(v[1], 1) < 0.002]
    print(f"{len(by_room)} rooms; choices into rooms with <0.2% productive tries: {sum(v[1] for _, v in dead) / total_c:.0%} ({len(dead)} rooms)")
    print("top 8 rooms by choices (room, visits, share of choices, productive rate):", [(r, v[0], f"{v[1] / total_c:.0%}", f"{v[2] / max(v[1], 1):.2%}") for r, v in rows[:8]])
