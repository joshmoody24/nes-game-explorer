"""
Metroid progression graph, for judging runs (never used by the explorer). Milestones come from Falcon Zero's FAQ
(metroiddatabase.com/old_site/m1/m1guide_fz.txt); each needs the milestones listed with it. `reached(memory)` says
which milestones one save has; `score(run)` reports the furthest milestones a run reached and what is missing next.
Usage: python progression.py <run> [<run> ...]
"""
import os, pickle, sys, zlib
from pathlib import Path
from metroid import ITEMS, KRAID_STATUE, RIDLEY_STATUE, MAX_MISSILES, area_of, at

BOMBS, HIGH_JUMP, LONG_BEAM, SCREW_ATTACK, MORPH_BALL, VARIA, WAVE_BEAM, ICE_BEAM = (1 << b for b in range(8))
BRINSTAR, NORFAIR, KRAID_LAIR, TOURIAN, RIDLEY_LAIR = range(5)

def item(bit):
    return lambda m: bool(m[at(ITEMS)] & bit)

def area(a):
    return lambda m: area_of(m) == a

# name: (test on one save's memory, milestones it needs first, the FAQ's gate for it)
MILESTONES = {
    "morph ball":   (item(MORPH_BALL), [], "roll under the rocks left of the start"),
    "missiles":     (lambda m: m[at(MAX_MISSILES)] > 0, ["morph ball"], "Missile Tank behind the first orange shaft"),
    "long beam":    (item(LONG_BEAM), ["missiles"], "red door, five missiles"),
    "bombs":        (item(BOMBS), ["missiles"], "red door at the far side, five missiles"),
    "ice beam":     (item(ICE_BEAM), ["bombs", "missiles"], "bomb a hole in the Waver room floor, then a red door"),
    "norfair":      (area(NORFAIR), ["morph ball"], "Brinstar elevator to Norfair"),
    "high jump":    (item(HIGH_JUMP), ["norfair", "bombs", "missiles"], "bomb a hidden tunnel, red door"),
    "kraid lair":   (area(KRAID_LAIR), ["bombs"], "bomb the floor of the corridor above the Kraid shaft"),
    "kraid dead":   (lambda m: bool(m[at(KRAID_STATUE)] & 3), ["kraid lair", "missiles"], "red door to Kraid's room"),
    "ridley lair":  (area(RIDLEY_LAIR), ["norfair", "bombs"], "bomb the fake floor half-way down Norfair's purple shaft"),
    "ridley dead":  (lambda m: bool(m[at(RIDLEY_STATUE)] & 3), ["ridley lair", "missiles"], "Ridley's room"),
    "tourian":      (area(TOURIAN), ["kraid dead", "ridley dead"], "shoot both statues, bridge, elevator"),
    "ending":       (lambda m: m[at(0x6883)] != 0, ["tourian", "ice beam"], "Metroids (ice + missiles), 10-missile door, Zebetites, Mother Brain, escape"),
}

def reached(memory):
    return {name for name, (test, _, _) in MILESTONES.items() if test(memory)}

def score(run_dir, emulator):
    from emulator import Emulator  # noqa: F401
    states = pickle.loads(zlib.decompress((run_dir / "checkpoint.pkl.z").read_bytes()))["states"]
    got = set()
    for s in states:
        got |= reached(emulator.view(s))
    ready = [n for n, (_, needs, _) in MILESTONES.items() if n not in got and all(x in got for x in needs)]
    return got, ready

if __name__ == "__main__":
    from emulator import Emulator
    from explorers import ROM
    emulator = Emulator(os.environ["QUICKNES_CORE"], ROM)
    for name in sys.argv[1:]:
        if name.endswith(".state"):
            got, ready = reached(emulator.view(Path(name).read_bytes())), []
            print(f"{name}: items byte {emulator.view(Path(name).read_bytes())[at(ITEMS)]:#04x} | has {sorted(got)}")
            continue
        got, ready = score(Path("runs") / name, emulator)
        order = list(MILESTONES)
        print(f"{name}: reached {[n for n in order if n in got]}")
        for n in ready:
            print(f"    next possible: {n} - gate: {MILESTONES[n][2]}")
