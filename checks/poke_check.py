"""Which RAM bytes matter? For saves from a run: change one byte, play 60 frames of a fixed button sequence, and see
whether the picture differs from the unpoked run. Bytes that never change the future are junk for places.
Usage: python -m checks.poke_check <checkpoint.pkl.z> [saves] [processes]. Metroid addresses only grade the answer."""
import os, pickle, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np

FRAMES, STEPS = 4, 15  # 15 steps of 4 frames = 60 frames


def matters(job):
    from controls import actions
    from emulator import Emulator
    from explorers import ROM
    state, seed = job
    e = Emulator(os.environ['QUICKNES_CORE'], ROM)
    rng = np.random.default_rng(seed)
    buttons = actions(True)
    presses = [buttons[rng.integers(len(buttons))] for _ in range(STEPS)]
    def future(s):
        e.load(s)
        for b in presses[:-1]:
            e.run(b, FRAMES)
        e.run(presses[-1], FRAMES, picture=True)
        return e.gray()
    base = future(state)
    offsets = e.state_offsets(np.arange(2048 + 8192))
    out = np.zeros(2048 + 8192, bool)
    poked = bytearray(state)
    for i, offset in enumerate(offsets):
        old = poked[offset]
        poked[offset] = old ^ 0x55
        out[i] = not np.array_equal(future(bytes(poked)), base)
        poked[offset] = old
    return out


if __name__ == "__main__":
    saves = pickle.loads(zlib.decompress(Path(sys.argv[1]).read_bytes()))['states']
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = np.random.default_rng(0)
    picks = [saves[i] for i in rng.choice(len(saves), count, replace=False)]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        share = np.mean(pool.map(matters, [(s, i) for i, s in enumerate(picks)]), axis=0)
    np.save(Path(sys.argv[1]).parent / 'poke_share.npy', share)
    for cut in [0.0, 0.1, 0.3, 0.5, 0.9]:
        print(f'bytes that changed the future in more than {cut:.0%} of saves: {(share > cut).sum()} of {len(share)}')
    from metroid import MAP_X, MAP_Y, ROOM_X, ROOM_Y
    for name, b in [('map x', MAP_X), ('map y', MAP_Y), ('Samus x', ROOM_X), ('Samus y', ROOM_Y), ('items', 0x1078),
                    ('missiles', 0x1079), ('area', 0x74), ('scroll x', 0xFD), ('scroll y', 0xFC)]:
        print(f'  {name}: matters in {share[b]:.0%} of saves, rank {(share > share[b]).sum()}')
