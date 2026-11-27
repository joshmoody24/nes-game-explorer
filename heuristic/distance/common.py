"""Shared loading, run split, and generic byte choice for the steps-apart model."""
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"
# Whole runs held out for testing; chosen by bench name before looking at any results.
TEST_PREFIXES = ("fert1_", "q2-stale40_")
# Two trajectory samples are this many steps apart (see build_data.EVERY).
EVERY = 4


def runs(test):
    return [p for p in sorted(DATA.glob("*.npz")) if p.name.startswith(TEST_PREFIXES) == test]


def load(path):
    return dict(np.load(path))


def change_rate(memory, trajectory):
    """Per byte: share of consecutive samples in a trajectory where it changed."""
    same_trajectory = trajectory[1:] == trajectory[:-1]
    return (memory[1:][same_trajectory] != memory[:-1][same_trajectory]).mean(0)


def room(memory):
    """Ground truth for evaluation only: (area, map y, map x)."""
    return (memory[:, 0x74].astype(int) & 15) * 65536 + memory[:, 0x4F].astype(int) * 256 + memory[:, 0x50]
