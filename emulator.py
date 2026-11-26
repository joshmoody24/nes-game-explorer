"""
Headless NES: drives a libretro core directly through ctypes. Video and audio are thrown away; all
we use is advancing frames, save states, and the 2 KB of system RAM.
"""

import ctypes
import os

import numpy as np

BUTTONS = {"B": 0, "SELECT": 2, "START": 3, "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7, "A": 8}
SYSTEM_RAM = 2
ENVIRONMENT_GET_CAN_DUPE = 3
ENVIRONMENT_GET_SYSTEM_DIRECTORY = 9
ENVIRONMENT_SET_PIXEL_FORMAT = 10
ENVIRONMENT_GET_SAVE_DIRECTORY = 31
ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE = 47 | 0x10000
PIXEL_FORMAT_XRGB8888 = 1
PIXEL_FORMAT_RGB565 = 2

EnvironmentCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VideoCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t)
# The per-frame callbacks are answered in C (shim/shim.c) so emulation never calls back into Python.
SHIM_CALLBACKS = {
    "retro_set_video_refresh": "shim_video",
    "retro_set_audio_sample": "shim_audio_sample",
    "retro_set_audio_sample_batch": "shim_audio_batch",
    "retro_set_input_poll": "shim_input_poll",
    "retro_set_input_state": "shim_input_state",
}


class GameInfo(ctypes.Structure):
    _fields_ = [("path", ctypes.c_char_p), ("data", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("meta", ctypes.c_char_p)]


def state_blocks(state):
    """Offsets of the tagged blocks in a QuickNES save state: {tag: (start, size)}. Layout: 8-byte header, then tag, size, data."""
    blocks, offset = {}, 8
    while offset + 8 <= len(state):
        tag, size = state[offset:offset + 4].decode(), int.from_bytes(state[offset + 4:offset + 8], "little")
        blocks[tag] = (offset + 8, size)
        offset += 8 + size
    return blocks


class MemoryView:
    """
    Single bytes of a save state's memory, indexed like Emulator.memory() (system RAM, then cartridge
    RAM from 0x800), without building the 10 KB array. For hot loops that read a handful of bytes.
    """
    __slots__ = ("_state", "_system_start", "_cartridge_start")

    def __init__(self, state, system_start, cartridge_start):
        self._state, self._system_start, self._cartridge_start = state, system_start, cartridge_start

    def __getitem__(self, index):
        return self._state[self._system_start + index if index < 0x800 else self._cartridge_start + index - 0x800]

    def take(self, indices):
        """Many bytes at once, like numpy's take on Emulator.memory()."""
        indices = np.asarray(indices)
        offsets = np.where(indices < 0x800, self._system_start + indices, self._cartridge_start + indices - 0x800)
        return np.frombuffer(self._state, np.uint8)[offsets]


def button_mask(*names):
    return sum(1 << BUTTONS[name] for name in names)


TURBO = 1 << 15  # not a real button: Emulator.run presses A and B for only the first half of the frames
TURBO_BUTTONS = button_mask("A", "B")


class Emulator:
    """
    One emulator per process: libretro cores keep their state in globals, so a second instance in
    the same process would share it.
    """

    def __init__(self, core_path, rom_path, scratch_dir="/tmp/metroid-explorer", shim_path=None):
        os.makedirs(scratch_dir, exist_ok=True)
        self._scratch_dir = ctypes.c_char_p(scratch_dir.encode())
        self._buttons = 0
        self._pixel_format = 0
        self._last_frame = None
        self._capturing_video = False
        self._core = ctypes.CDLL(core_path)
        self._shim = ctypes.CDLL(shim_path or os.environ["SHIM"])
        # Kept as attributes so they aren't garbage collected while the core holds them.
        self._environment_callback = EnvironmentCallback(self._environment)
        self._capture_callback = VideoCallback(self._capture)
        self._core.retro_set_environment(self._environment_callback)
        for setter, shim_function in SHIM_CALLBACKS.items():
            getattr(self._core, setter)(ctypes.cast(getattr(self._shim, shim_function), ctypes.c_void_p))
        self._retro_run = ctypes.cast(self._core.retro_run, ctypes.c_void_p)
        self._core.retro_init()
        self._rom = open(rom_path, "rb").read()
        game = GameInfo(rom_path.encode(), ctypes.cast(ctypes.c_char_p(self._rom), ctypes.c_void_p),
                        len(self._rom), None)
        if not self._core.retro_load_game(ctypes.byref(game)):
            raise RuntimeError(f"Core {core_path} failed to load {rom_path}")
        self._core.retro_serialize_size.restype = ctypes.c_size_t
        self._core.retro_get_memory_data.restype = ctypes.c_void_p
        self._core.retro_get_memory_size.restype = ctypes.c_size_t
        self.state_size = self._core.retro_serialize_size()
        self._state_buffer = ctypes.create_string_buffer(self.state_size)
        ram_pointer = self._core.retro_get_memory_data(SYSTEM_RAM)
        ram_size = self._core.retro_get_memory_size(SYSTEM_RAM)
        if not ram_pointer or ram_size != 2048:
            raise RuntimeError(f"Core {core_path} exposes no 2 KB system RAM (got {ram_size} bytes)")
        # A live view: it always shows the current RAM, no copying.
        self.ram = np.ctypeslib.as_array(ctypes.cast(ram_pointer, ctypes.POINTER(ctypes.c_uint8)), (ram_size,))
        # The core doesn't expose cartridge RAM ($6000-$7FFF) directly, but save states include it.
        blocks = state_blocks(self.save())
        if "SRAM" not in blocks:
            raise RuntimeError(f"Save states from {core_path} have no cartridge RAM block")
        # Every state from this core and game has the same layout, so the offsets are found once.
        self._system_slice = slice(blocks["LRAM"][0], sum(blocks["LRAM"]))
        self._cartridge_slice = slice(blocks["SRAM"][0], sum(blocks["SRAM"]))
        self._nametable_slice = slice(blocks["NTAB"][0], sum(blocks["NTAB"]))

    def _environment(self, command, data):
        if command == ENVIRONMENT_GET_CAN_DUPE:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
            return True
        if command in (ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_SAVE_DIRECTORY):
            ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.cast(self._scratch_dir, ctypes.c_void_p).value
            return True
        if command == ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE:
            # Bit 0 enables video, bit 1 audio. Cores that honor this skip producing frames we'd discard.
            ctypes.cast(data, ctypes.POINTER(ctypes.c_int))[0] = int(self._capturing_video)
            return True
        if command == ENVIRONMENT_SET_PIXEL_FORMAT:
            self._pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_int))[0]
            return True
        return False

    def _capture(self, data, width, height, pitch):
        if data:
            self._last_frame = (ctypes.string_at(data, height * pitch), width, height, pitch)

    def run(self, buttons=0, frames=1):
        """Holds buttons for frames frames. The loop runs in C. With TURBO set, A and B are let go for the second
        half: most games act once per press, so holding fire fires once."""
        if buttons & TURBO:
            held = buttons & ~TURBO
            pressed = frames // 2
            self._buttons = held
            self._shim.shim_run(self._retro_run, held, pressed)
            buttons, frames = held & ~TURBO_BUTTONS, frames - pressed
        self._buttons = buttons
        self._shim.shim_run(self._retro_run, buttons, frames)

    def save(self):
        if not self._core.retro_serialize(self._state_buffer, self.state_size):
            raise RuntimeError("Save state failed")
        return self._state_buffer.raw

    def memory(self, state=None):
        """System RAM ($0000-$07FF) followed by cartridge RAM ($6000-$7FFF), as one uint8 array."""
        state = state or self.save()
        return np.concatenate([np.frombuffer(state[self._system_slice], np.uint8),
                               np.frombuffer(state[self._cartridge_slice], np.uint8)])

    def state_offsets(self, memory_indices):
        """Where each given index of memory() sits inside a save state, so hot loops can read bytes directly."""
        memory_indices = np.asarray(memory_indices)
        return np.where(memory_indices < 0x800, self._system_slice.start + memory_indices,
                        self._cartridge_slice.start + memory_indices - 0x800)

    def view(self, state):
        """Like memory(state), but reads bytes on demand; much cheaper when only a few are needed."""
        return MemoryView(state, self._system_slice.start, self._cartridge_slice.start)

    def nametable(self, state=None):
        """The background tile layout (2 KB), from a save state or the current one."""
        return (state or self.save())[self._nametable_slice]

    def load(self, state):
        if not self._core.retro_unserialize(state, len(state)):
            raise RuntimeError("Load state failed")

    def screenshot(self):
        """Runs one more frame and returns it as RGB [height, width, 3]. For debugging only."""
        return self.frame(self._buttons)

    def frame(self, buttons):
        """Holds buttons for one frame, with video on, and returns the picture as RGB [height, width, 3]."""
        self._core.retro_set_video_refresh(self._capture_callback)
        self._capturing_video = True
        self.run(buttons)
        self._capturing_video = False
        self._core.retro_set_video_refresh(ctypes.cast(self._shim.shim_video, ctypes.c_void_p))
        raw, width, height, pitch = self._last_frame
        if self._pixel_format == PIXEL_FORMAT_XRGB8888:
            pixels = np.frombuffer(raw, np.uint8).reshape(height, pitch // 4, 4)[:, :width]
            return pixels[..., 2::-1].copy()
        words = np.frombuffer(raw, np.uint16).reshape(height, pitch // 2)[:, :width].astype(np.uint32)
        if self._pixel_format == PIXEL_FORMAT_RGB565:
            channels = [(words >> 11) & 31, (words >> 5) & 63, words & 31]
            maxima = [31, 63, 31]
        else:
            channels = [(words >> 10) & 31, (words >> 5) & 31, words & 31]
            maxima = [31, 31, 31]
        return np.stack([c * 255 // m for c, m in zip(channels, maxima)], -1).astype(np.uint8)
