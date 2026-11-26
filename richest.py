"""Richest healthy saves of runs: (filled tanks, energy, missiles), healthy = survives 20 idle steps without losing energy."""
import os, pickle, sys, zlib
from pathlib import Path
from controls import FRAMES_PER_STEP
from emulator import Emulator
from explorers import ROM
from metroid import GAME_MODE, GAME_OVER, at, energy_of

def healthy(emulator, s):
    emulator.load(s); e0 = energy_of(emulator.memory())
    for _ in range(20):
        emulator.run(0, FRAMES_PER_STEP)
        if emulator.memory()[GAME_MODE] == GAME_OVER: return False
    return energy_of(emulator.memory()) >= e0 - 2

if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for run in sys.argv[1:]:
        states = pickle.loads(zlib.decompress(Path(f"runs/{run}/checkpoint.pkl.z").read_bytes()))["states"]
        rich = lambda s: (int(emulator.view(s)[263]) >> 4, energy_of(emulator.view(s)), int(emulator.view(s)[at(0x6879)]))
        ranked = sorted(states, key=rich, reverse=True)
        top = [rich(s) for s in ranked[:400] if healthy(emulator, s)][:5]
        print(f"{run}: {len(states)} saves; richest healthy (filled tanks, energy, missiles): {top}")
