"""Metroid-specific knowledge: getting into the game, and RAM addresses (Data Crystal / nmikstas disassembly)."""

from emulator import button_mask

PLAYING = 3  # values of the main routine index $001E; picking up an item briefly switches to 9 (wait timer)
GAME_OVER = 4
GAME_MODE = 0x1E
AREA = 0x74
MAP_Y, MAP_X = 0x4F, 0x50
ROOM_Y, ROOM_X = 0x30D, 0x30E
ITEMS = 0x6878
ENERGY_TANKS = 0x6877
MAX_MISSILES = 0x687A
KRAID_STATUE, RIDLEY_STATUE = 0x687B, 0x687C


def at(address):
    """Index into Emulator.memory(), which holds cartridge RAM ($6000-) right after the 2 KB system RAM."""
    return address - 0x6000 + 0x800 if address >= 0x6000 else address


def start_game(emulator):
    """From power-on, presses START through the title screens and returns the save state once play begins."""
    emulator.run(frames=3000)
    for _ in range(4):
        emulator.run(button_mask("START"), frames=5)
        emulator.run(frames=120)
    for _ in range(600):
        if emulator.ram[GAME_MODE] == PLAYING:
            return emulator.save()
        emulator.run()
    raise RuntimeError(f"Game never reached play mode; $001E = {emulator.ram[GAME_MODE]}")
