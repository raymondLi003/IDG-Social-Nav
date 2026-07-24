"""Grid geometry and BFS shared by planners and advisors.

"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection

import numpy as np

from idg_social_nav.core import DIR_OFFSET

Cell = tuple[int, int]

_EMPTY: frozenset[Cell] = frozenset()


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_direction(src: Cell, dst: Cell) -> int | None:
    """Direction whose unit offset moves src onto dst; None if not adjacent."""
    delta = (dst[0] - src[0], dst[1] - src[1])
    for d, offset in DIR_OFFSET.items():
        if offset == delta:
            return d
    return None


def bfs_first_step(
        walls: np.ndarray,
        start: Cell,
        is_target: Callable[[Cell], bool],
        blocked: Collection[Cell] = _EMPTY,
) -> tuple[Cell | None, Cell | None]:
    """
    (first step, target) on a shortest path from start to the nearest
    cell satisfying is_target
    (None, None) when none is reachable.

    start itself is never tested against is_target.
    walls and blocked cells are impassable.
    """
    h, w = walls.shape
    parent: dict[Cell, Cell] = {start: start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur != start and is_target(cur):
            node = cur
            while parent[node] != start:
                node = parent[node]
            return node, cur
        for dr, dc in DIR_OFFSET.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if walls[nxt[0], nxt[1]] == 1 or nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    return None, None


def bfs_next_toward(
        walls: np.ndarray,
        start: Cell,
        goal: Cell,
        blocked: Collection[Cell] = _EMPTY,
) -> Cell | None:
    """First cell of a shortest path from start to goal
    or None when already there or unreachable."""
    if start == goal:
        return None
    step, _ = bfs_first_step(walls, start, lambda cell: cell == goal, blocked)
    return step


def bfs_distances(
        walls: np.ndarray,
        goal: Cell,
        blocked: Collection[Cell] = _EMPTY,
) -> np.ndarray:
    """Shortest-path distance to goal per cell (inf where unreachable)."""
    h, w = walls.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    if goal in blocked:
        return dist
    dist[goal[0], goal[1]] = 0.0
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR_OFFSET.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if walls[nr, nc] == 1 or (nr, nc) in blocked:
                continue
            if np.isinf(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1.0
                queue.append((nr, nc))
    return dist
