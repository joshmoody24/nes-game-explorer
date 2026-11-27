"""Which RAM bytes respond to the controller (contingency)? From 300 saves: right vs left and jump vs nothing, 3 vs 6 steps.
Usage: python -m checks.contingency_check <checkpoint.pkl.z>. Metroid addresses are printed only to check the answer."""
import os, pickle, sys, zlib
from pathlib import Path
import numpy as np
from controls import FRAMES_PER_STEP
from emulator import Emulator, button_mask
from explorers import ROM
from metroid import ROOM_X, ROOM_Y, MAP_X, MAP_Y
e = Emulator(os.environ['QUICKNES_CORE'], ROM)
s = pickle.loads(zlib.decompress(Path(sys.argv[1]).read_bytes()))['states']
rng = np.random.default_rng(0)
def after(state, buttons, steps=6):
    e.load(state)
    for _ in range(steps): e.run(buttons, FRAMES_PER_STEP)
    return e.memory().astype(np.int16)
pairs = {'right vs left': (button_mask('RIGHT'), button_mask('LEFT')), 'jump vs nothing': (button_mask('A'), 0)}
for label, (a, b) in pairs.items():
    agree, grew = 0, 0
    n = 300
    for i in rng.choice(len(s), n, replace=False):
        short = after(s[i], a, 3) - after(s[i], b, 3)
        long = after(s[i], a, 6) - after(s[i], b, 6)
        agree = agree + np.sign(long); grew = grew + (np.abs(long) > np.abs(short))
    # Consistent direction, and the gap widens with a longer press: a position does, a facing flag doesn't.
    score = np.minimum(np.abs(agree), grew) / n
    top = np.argsort(-score)[:8]
    print(label, 'top bytes:', [(int(t), f'{score[t]:.2f}') for t in top])
    for name, b_ in [('ROOM_X', ROOM_X), ('ROOM_Y', ROOM_Y), ('scroll x $FD', 0xFD), ('scroll y $FC', 0xFC)]:
        print(f'   {name} ({b_}): score {score[b_]:.2f}, rank {int((score > score[b_]).sum())}')
