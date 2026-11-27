# nes-game-explorer

Save-state exploration (Go-Explore style) for NES games, driven from raw RAM. It has beaten NES Metroid from power-on to
the credits without edited RAM; the current work is removing the remaining hand-made and Metroid-specific pieces so the
same method can run on other games.

- `go_explore.py`, `explorers.py`: the explorer (archive of save states, random tries, keep rules).
- `bench.py`: a benchmark of hard spots ("walls") from the winning route, scored in frames of play to get through.
- `autochain.py`: chains short explorations without a human picking where to restart.
- `FINDINGS.md`: a dated log of every experiment and what it showed.

No ROM is included. Bring your own `roms/metroid.nes`; the Nix flake provides the emulator core (QuickNES).
