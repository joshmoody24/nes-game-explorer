"""Pick the healthy save with the lowest boss HP (energy >= 60) from a run: python ratchet.py <run> <out.state>"""
import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import energy_of, at
from richest import healthy
def boss_hp(v):
    hp = [int(v[0x040B + 16 * k]) for k in range(6) if v[0x040F + 16 * k] & 0x40 and v[at(0x6AF4) + 16 * k]]
    return min(hp) if hp else None
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    total = lambda v: (int(v[0x107]) >> 4) * 100 + energy_of(v)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    dead = sum(1 for s in saved["states"] if emulator.view(s)[at(0x687C)] & 3)
    cands = sorted((s for s in saved["states"] if boss_hp(emulator.view(s)) is not None and 60 <= total(emulator.view(s)) <= 500),
                   key=lambda s: (boss_hp(emulator.view(s)), -total(emulator.view(s))))
    s = next(s for s in cands if healthy(emulator, s)); v = emulator.view(s)
    Path(sys.argv[2]).write_bytes(s)
    print(f"{sys.argv[1]}: Ridley dead saves {dead} | next start: Ridley HP {boss_hp(v)}, energy {total(v)}, missiles {int(v[at(0x6879)])}")
