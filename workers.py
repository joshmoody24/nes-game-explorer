"""
One pool of worker processes for everything that emulates in parallel (replaying recordings, sampling and
running trips), started once per run: a spawned worker re-imports the main script, which usually imports
JAX, so starting 18 of them costs ~2 s. Tasks carry what they need; each worker keeps its own emulator.
Workers run models on the CPU, one thread each, so 18 of them neither grab the GPU nor oversubscribe cores.
"""

import functools
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

COUNT = 18
# In the environment workers inherit: they import JAX while re-importing the main script, before any of
# their own code runs.
_CPU_ONLY = {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"}


def _started(_):
    return os.getpid()


@functools.cache
def _pool():
    saved = {name: os.environ.get(name) for name in _CPU_ONLY}
    os.environ.update(_CPU_ONLY)
    try:
        pool = ProcessPoolExecutor(COUNT, mp_context=get_context("spawn"))
        # Workers start on demand, one per waiting task, so give them COUNT tasks to start them all now.
        list(pool.map(_started, range(COUNT)))
        return pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value


def map(function, tasks):
    """function(task) for each task, in the workers, in order."""
    return _pool().map(function, tasks)


def split(items, pieces_per_worker=8, most=None):
    """items in consecutive chunks, enough to keep every worker busy, each at most `most` long."""
    size = max(1, -(-len(items) // (COUNT * pieces_per_worker)))
    size = min(size, most) if most else size
    return [items[i:i + size] for i in range(0, len(items), size)]


@functools.cache
def emulator():
    """In a worker: its emulator."""
    from emulator import Emulator
    from explorers import ROM
    return Emulator(os.environ["QUICKNES_CORE"], ROM)
