import os, pickle, sys, zlib
from pathlib import Path
from emulator import Emulator
from explorers import ROM
from metroid import at
if __name__ == "__main__":
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    saved = pickle.loads(zlib.decompress(Path(f"runs/{sys.argv[1]}/checkpoint.pkl.z").read_bytes()))
    ms = [emulator.memory(s) for s in saved["states"]]
    hp = [int(m[0x40B]) for m in ms if m[at(0x6AF4)] and m[0x40F] & 0x40]
    print(len(ms), "saves;", sum(1 for p in saved["places"] if p[0] == "record"), "record places | boss HP min", min(hp) if hp else None, "| Kraid dead saves:", sum(1 for m in ms if m[at(0x687B)] & 3), "| Ridley dead saves:", sum(1 for m in ms if m[at(0x687C)] & 3))
