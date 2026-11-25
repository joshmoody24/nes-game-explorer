// Answers the libretro core's per-frame callbacks in C, so the emulator never calls back into Python.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define JOYPAD 1

static unsigned pressed = 0;

int16_t shim_input_state(unsigned port, unsigned device, unsigned index, unsigned button) {
    return port == 0 && device == JOYPAD ? (pressed >> button) & 1 : 0;
}

void shim_video(const void *data, unsigned width, unsigned height, size_t pitch) {}
void shim_audio_sample(int16_t left, int16_t right) {}
size_t shim_audio_batch(const int16_t *data, size_t frames) { return frames; }
void shim_input_poll(void) {}

// Holds the buttons for `frames` frames, calling the core's retro_run each time.
void shim_run(void (*retro_run)(void), unsigned buttons, unsigned frames) {
    pressed = buttons;
    for (unsigned i = 0; i < frames; i++) retro_run();
}
