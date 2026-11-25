"""
Headless NES: drives a libretro core directly through ctypes. Video and audio are thrown away; all
we use is advancing frames, save states, and the 2 KB of system RAM.
"""

import ctypes
import os

import numpy as np

BUTTONS = {"B": 0, "SELECT": 2, "START": 3, "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7, "A": 8}
JOYPAD = 1
SYSTEM_RAM = 2
ENVIRONMENT_GET_CAN_DUPE = 3
ENVIRONMENT_GET_SYSTEM_DIRECTORY = 9
ENVIRONMENT_SET_PIXEL_FORMAT = 10
ENVIRONMENT_GET_SAVE_DIRECTORY = 31
PIXEL_FORMAT_XRGB8888 = 1
PIXEL_FORMAT_RGB565 = 2

EnvironmentCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VideoCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t)
AudioSampleCallback = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
AudioBatchCallback = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
InputPollCallback = ctypes.CFUNCTYPE(None)
InputStateCallback = ctypes.CFUNCTYPE(ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)


class GameInfo(ctypes.Structure):
    _fields_ = [("path", ctypes.c_char_p), ("data", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("meta", ctypes.c_char_p)]


def button_mask(*names):
    return sum(1 << BUTTONS[name] for name in names)


class Emulator:
    """
    One emulator per process: libretro cores keep their state in globals, so a second instance in
    the same process would share it.
    """

    def __init__(self, core_path, rom_path, scratch_dir="/tmp/metroid-explorer"):
        os.makedirs(scratch_dir, exist_ok=True)
        self._scratch_dir = ctypes.c_char_p(scratch_dir.encode())
        self._pressed = 0
        self._pixel_format = 0
        self._last_frame = None
        self._capturing = False
        self._core = ctypes.CDLL(core_path)
        # Kept as attributes so the callbacks aren't garbage collected while the core holds them.
        self._callbacks = [
            (self._core.retro_set_environment, EnvironmentCallback(self._environment)),
            (self._core.retro_set_video_refresh, VideoCallback(self._video)),
            (self._core.retro_set_audio_sample, AudioSampleCallback(lambda *_: None)),
            (self._core.retro_set_audio_sample_batch, AudioBatchCallback(lambda data, frames: frames)),
            (self._core.retro_set_input_poll, InputPollCallback(lambda: None)),
            (self._core.retro_set_input_state, InputStateCallback(self._input_state)),
        ]
        for setter, callback in self._callbacks:
            setter(callback)
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

    def _environment(self, command, data):
        if command == ENVIRONMENT_GET_CAN_DUPE:
            ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
            return True
        if command in (ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_SAVE_DIRECTORY):
            ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.cast(self._scratch_dir, ctypes.c_void_p).value
            return True
        if command == ENVIRONMENT_SET_PIXEL_FORMAT:
            self._pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_int))[0]
            return True
        return False

    def _video(self, data, width, height, pitch):
        if self._capturing and data:
            self._last_frame = (ctypes.string_at(data, height * pitch), width, height, pitch)

    def _input_state(self, port, device, index, button):
        return (self._pressed >> button) & 1 if port == 0 and device == JOYPAD else 0

    def run(self, buttons=0, frames=1):
        self._pressed = buttons
        for _ in range(frames):
            self._core.retro_run()

    def save(self):
        if not self._core.retro_serialize(self._state_buffer, self.state_size):
            raise RuntimeError("Save state failed")
        return self._state_buffer.raw

    def load(self, state):
        if not self._core.retro_unserialize(state, len(state)):
            raise RuntimeError("Load state failed")

    def screenshot(self):
        """Runs one more frame and returns it as RGB [height, width, 3]. For debugging only."""
        self._capturing = True
        self.run(self._pressed)
        self._capturing = False
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
