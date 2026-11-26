"""
The archive as a map. Replaying stored paths shows which locations lead to which, and with what buttons:
each time a path's location changes from u to v, the buttons it pressed while in u are a "chunk" for the
edge u -> v. travel() then carries any state (with whatever it holds) across known territory: plan the
shortest chain of locations, and for each hop try the known chunks until one lands in the next location.
Locations are hand-made places without their inventory fields (see location()); diagnostic, like places.
"""

from collections import deque

import numpy as np

import workers
from controls import ACTIONS, FRAMES_PER_STEP
from explorers import Settings, hand_made_place

CHUNKS_PER_EDGE = 6
ARRIVALS_TO_COMPARE = 20  # local exploration keeps the richest of this many arrivals


def location(memory):
    """Area, room and rough position: a hand-made place without items, tanks or missiles."""
    return hand_made_place(memory, Settings())[:5]


def _path_edges(task):
    """In a worker: (u, v, chunk) for every location change along one stored path."""
    start, path = task
    emulator = workers.emulator()
    emulator.load(start)
    edges, here, entered = [], location(emulator.memory()), 0
    for step, action in enumerate(path):
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        there = location(emulator.memory())
        if there != here:
            edges.append((here, there, bytes(path[entered:step + 1])))
            here, entered = there, step + 1
    return edges


def build(journeys):
    """{u: {v: [chunks, shortest first]}} from replaying journeys: (start state, buttons) pairs."""
    graph = {}
    for edges in workers.map(_path_edges, journeys):
        for u, v, chunk in edges:
            chunks = graph.setdefault(u, {}).setdefault(v, [])
            if chunk not in chunks:
                chunks.append(chunk)
    for u in graph:
        for v in graph[u]:
            graph[u][v] = sorted(graph[u][v], key=len)[:CHUNKS_PER_EDGE]
    return graph


def plan(graph, here, targets, banned=frozenset()):
    """Shortest chain of locations from here to any of targets, avoiding banned edges, or None."""
    previous, queue = {here: None}, deque([here])
    while queue:
        u = queue.popleft()
        if u in targets:
            chain = [u]
            while previous[chain[-1]] is not None:
                chain.append(previous[chain[-1]])
            return chain[::-1]
        for v in graph.get(u, {}):
            if v not in previous and (u, v) not in banned:
                previous[v] = u
                queue.append(v)
    return None


def _hop(emulator, state, chunk, ahead):
    """Runs chunk from state. Returns (state, position on the chain) on first arriving at a location in ahead."""
    emulator.load(state)
    for action in chunk:
        emulator.run(ACTIONS[action], FRAMES_PER_STEP)
        here = location(emulator.memory())
        if here in ahead:
            return emulator.save(), ahead[here]
    return None, -1


def _random_chunk(rng, steps):
    action, chunk = int(rng.integers(len(ACTIONS))), []
    for _ in range(steps):
        if rng.random() > 0.9:
            action = int(rng.integers(len(ACTIONS)))
        chunk.append(action)
    return bytes(chunk)


def _local_search(emulator, state, ahead, rng, keep, segments=3000, steps=40):
    """
    A small Go-Explore from state toward any location in ahead, for hops no known chunk makes: an archive of
    locations, each keeping its richest visit (by keep) or else its first, explored from with sticky random
    segments, rarely chosen first. Of the first ARRIVALS_TO_COMPARE arrivals, returns the furthest along
    richest and then furthest along, as (state, position on the chain); (None, -1) if none.
    """
    archive = {location(emulator.memory(state)): [state, 0]}
    arrivals = []
    for _ in range(segments):
        if len(arrivals) >= ARRIVALS_TO_COMPARE:
            break
        places = list(archive)
        weights = np.array([1 / np.sqrt(1 + archive[p][1]) for p in places])
        chosen = places[int(rng.choice(len(places), p=weights / weights.sum()))]
        archive[chosen][1] += 1
        emulator.load(archive[chosen][0])
        action = int(rng.integers(len(ACTIONS)))
        for _ in range(steps):
            if rng.random() > 0.9:
                action = int(rng.integers(len(ACTIONS)))
            emulator.run(ACTIONS[action], FRAMES_PER_STEP)
            memory = emulator.memory()
            here = location(memory)
            if here in ahead:
                arrivals.append(((keep(memory) if keep else 0, ahead[here]), emulator.save()))
                if len(arrivals) >= ARRIVALS_TO_COMPARE:
                    break
                continue
            if here not in archive:
                archive[here] = [emulator.save(), 0]
            elif keep and keep(memory) > keep(emulator.memory(archive[here][0])):
                archive[here][0] = emulator.save()
    if not arrivals:
        return None, -1
    (_, index), best = max(arrivals, key=lambda arrival: arrival[0])
    return best, index


def travel(emulator, state, graph, targets, rng, keep=None, random_tries=200, random_steps=60, slack=4,
           most_replans=50):
    """
    Carries state to any of targets (locations) across known territory, using save states. Plans the
    shortest chain of locations; for each hop tries the known chunks, then random retries, accepting arrival
    anywhere further along the chain; a hop none of those makes gets a small local exploration (see
    _local_search). A hop that still can't be made is banned and the route re-planned.
    keep(memory) -> comparable: when given, every candidate is tried and the arrival that holds the most, then
    gets furthest, is kept instead of the first arrival.
    Returns (arrived, last state, steps tried, hops made, the hop it last failed or None).
    """
    banned, steps, hops, failed = set(), 0, 0, None
    for _ in range(most_replans):
        here = location(emulator.memory(state))
        if here in targets:
            return True, state, steps, hops, failed
        chain = plan(graph, here, set(targets), frozenset(banned))
        if chain is None:
            return False, state, steps, hops, failed
        ahead = {loc: k for k, loc in enumerate(chain[1:], start=1)}
        candidates = [chunk + bytes(slack) for chunk in graph[chain[0]][chain[1]]]
        candidates += [_random_chunk(rng, random_steps) for _ in range(random_tries)]
        best = None
        for chunk in candidates:
            arrived, index = _hop(emulator, state, chunk, ahead)
            steps += len(chunk)
            if arrived:
                # Riches first: any arrival is progress, and carrying is for keeping what you hold.
                rank = (keep(emulator.memory(arrived)) if keep else 0, index)
                if best is None or rank > best[0]:
                    best = (rank, arrived, index)
                if keep is None:
                    break
        if best is None:
            # No known chunk or quick try makes this hop: explore it.
            arrived, index = _local_search(emulator, state, ahead, rng, keep)
            if arrived:
                best = (None, arrived, index)
        if best:
            _, state, index = best
            hops += index
        else:
            banned.add((chain[0], chain[1]))
            failed = (chain[0], chain[1])
    return False, state, steps, hops, failed


def _travel_task(task):
    state, world, targets, seed, keep = task
    return travel(workers.emulator(), state, world, targets, np.random.default_rng(seed), keep)


def travel_many(states, world, targets, keep=None, seed=0):
    """
    travel() for many states at once, in the worker processes: targets is one set of locations for all, or
    a list with one set per state. keep must be a module-level function.
    """
    per_state = targets if isinstance(targets, list) else [targets] * len(states)
    return list(workers.map(_travel_task, [(s, world, t, seed + i, keep) for i, (s, t) in enumerate(zip(states, per_state))]))
