"""Social discomfort field
This will only be seen by the validator

Replaces or extends the lava channel with a graded, asymmetric personal-space
zone around each pedestrian, elongated along its facing direction. 
Entering the field is similar to stepping in lava,
this is graded rather than terminal.

Gestures influence the field: 
STOP extends the frontal zone (the pedestrian asks for the right of way)
GO shrinks it (the pedestrian yields).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idg_social_nav.core import DIR_OFFSET, Gesture


@dataclass(frozen=True)
class DiscomfortParams:
    front_extent: float = 3.0     # semi-axis (cells) ahead of the pedestrian
    lateral_extent: float = 1.5   # semi-axis to the sides
    behind_extent: float = 1.0    # semi-axis behind
    stop_front_scale: float = 2.0  # STOP gesture: frontal zone extends
    go_front_scale: float = 0.5    # GO gesture: frontal zone shrinks
    los_masking: bool = True       # walls block the field (no discomfort through walls)
    high_threshold: float = 0.5    # tau: intensity at or above this is high discomfort


def has_line_of_sight(
        walls: np.ndarray,
        a: tuple[int, int],
        b: tuple[int, int],
) -> bool:
    """line-of-sight between two cells
        endpoints never block."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        if (r, c) != (r1, c1) and walls[r, c] == 1:
            return False
    return True


def pedestrian_field(
        walls: np.ndarray,
        ped_pos: tuple[int, int],
        ped_facing: int,
        gesture: Gesture,
        params: DiscomfortParams,
) -> np.ndarray:
    """Graded discomfort field of a single pedestrian, in [0, 1].

    Intensity at a cell offset decomposed into a forward component f (along
    the pedestrian's facing) and a lateral component l:

        d_eff = sqrt((f / a)^2 + (l / lateral_extent)^2)
        intensity = max(0, 1 - d_eff)

    with a = front_extent (scaled by gestures) when the cell is in front
    (f >= 0), else a = behind_extent. The pedestrian's cell is 1.0.
    """
    h, w = walls.shape
    out = np.zeros((h, w), dtype=np.float32)

    front_extent = params.front_extent
    if gesture == Gesture.STOP:
        front_extent *= params.stop_front_scale
    elif gesture == Gesture.GO:
        front_extent *= params.go_front_scale

    fr, fc = DIR_OFFSET[ped_facing]
    pr, pc = ped_pos

    # only cells within the largest possible reach can be nonzero
    reach = int(np.ceil(max(front_extent, params.lateral_extent, params.behind_extent)))
    for r in range(max(0, pr - reach), min(h, pr + reach + 1)):
        for c in range(max(0, pc - reach), min(w, pc + reach + 1)):
            if walls[r, c] == 1:
                continue
            dr = r - pr
            dc = c - pc
            f = dr * fr + dc * fc                   # forward component
            l = abs(dr * fc - dc * fr)              # lateral component
            a = front_extent if f >= 0 else params.behind_extent
            d_eff = np.sqrt((f / a) ** 2 + (l / params.lateral_extent) ** 2)
            intensity = max(0.0, 1.0 - float(d_eff))
            if intensity <= 0.0:
                continue
            if params.los_masking and not has_line_of_sight(walls, ped_pos, (r, c)):
                continue
            out[r, c] = intensity

    out[pr, pc] = 1.0
    return out


def discomfort_field(
        walls: np.ndarray,
        pedestrians: list[tuple[tuple[int, int], int, Gesture]],
        params: DiscomfortParams,
) -> np.ndarray:
    """Combined field over all pedestrians
    """
    h, w = walls.shape
    out = np.zeros((h, w), dtype=np.float32)
    for pos, facing, gesture in pedestrians:
        np.maximum(out, pedestrian_field(walls, pos, facing, gesture, params), out=out)
    return out
