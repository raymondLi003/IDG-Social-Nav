"""Headless frame renderer: the grid analogue of the robot's camera feed.

Draws the full world state (walls, goal, agent, pedestrians with gestures).

The frame is what pixel-mode VLM advisors "see". The privileged discomfort
field is drawn only when show_field=True (demo videos) but never for advisor
frames, just to have privileged evidence not leak into the advisor's camera.
Output is deterministic for identical env states, 
so CachedAdvisor can precompute and cache advice for every reachable state in that scenario
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from idg_social_nav.core import DIR_OFFSET, DOWN, LEFT, RIGHT, UP, Gesture

pygame.init()

_COLOR_WALL = (0, 0, 0)
_COLOR_FLOOR = (200, 200, 200)
_COLOR_FLOOR_ALT = (185, 185, 185)
_COLOR_GRID = (150, 150, 150)
_COLOR_GOAL = (0, 180, 0)
_COLOR_AGENT = (30, 60, 200)
_COLOR_PED = (235, 140, 40)
_COLOR_PED_ARROW = (120, 60, 10)
_COLOR_STOP = (220, 30, 30)
_COLOR_GO = (30, 170, 60)
_COLOR_LABEL = (255, 255, 255)
_COLOR_MARKER = (255, 255, 255)

_FONT_CACHE: dict[int, pygame.font.Font | None] = {}


def _get_font(size: int):
    """default font."""
    if size not in _FONT_CACHE:
        font = None
        try:
            if not pygame.font.get_init():
                pygame.font.init()
            font = pygame.font.Font(None, size)
        except Exception:
            font = None
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def _blit_text(surface, text, center, size, color) -> None:
    font = _get_font(size)
    if font is None:
        return
    try:
        label = font.render(text, True, color)
    except Exception:
        return
    surface.blit(label, label.get_rect(center=(int(center[0]), int(center[1]))))


def _dir_vector(direction: int) -> tuple[int, int]:
    """Screen-space (dx, dy) unit vector for a grid direction."""
    dr, dc = DIR_OFFSET[direction]
    return dc, dr


def _draw_arrow(surface, color, start, end, width=2) -> None:
    pygame.draw.line(surface, color, start, end, width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / norm, dy / norm
    size = max(4.0, 0.3 * norm)
    left = (end[0] - size * ux + 0.5 * size * uy, end[1] - size * uy - 0.5 * size * ux)
    right = (end[0] - size * ux - 0.5 * size * uy, end[1] - size * uy + 0.5 * size * ux)
    pygame.draw.polygon(surface, color, [end, left, right])


def _draw_agent(surface, pos, direction, ts, annotate) -> None:
    r, c = pos
    cx = c * ts + ts / 2
    cy = r * ts + ts / 2
    size = ts * 0.35
    if direction == UP:
        pts = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    elif direction == RIGHT:
        pts = [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]
    elif direction == DOWN:
        pts = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    elif direction == LEFT:
        pts = [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]
    else:
        pts = []
    if pts:
        pygame.draw.polygon(surface, _COLOR_AGENT, pts)
    if annotate:
        dx, dy = _dir_vector(direction)
        _draw_arrow(
            surface, _COLOR_MARKER,
            (cx, cy), (cx + 0.42 * ts * dx, cy + 0.42 * ts * dy), width=2,
        )
        _blit_text(
            surface, "R",
            (cx - 0.26 * ts * dx, cy - 0.26 * ts * dy),
            max(10, int(ts * 0.38)), _COLOR_MARKER,
        )


def _draw_pedestrian(surface, pos, facing, gesture, ts) -> None:
    """
    Person glyph: head circle + shoulder ellipse, 
    short heading arrow,
    and a gesture mark (STOP: red raised palm, GO: green sweeping arm)."""
    r, c = pos
    cx = c * ts + ts / 2
    cy = r * ts + ts / 2
    dx, dy = _dir_vector(facing)

    shoulders = pygame.Rect(0, 0, int(ts * 0.52), int(ts * 0.34))
    shoulders.center = (int(cx), int(cy + 0.12 * ts))
    pygame.draw.ellipse(surface, _COLOR_PED, shoulders)
    pygame.draw.circle(
        surface, _COLOR_PED, (int(cx), int(cy - 0.16 * ts)), int(ts * 0.15))
    _draw_arrow(
        surface, _COLOR_PED_ARROW,
        (cx, cy), (cx + 0.45 * ts * dx, cy + 0.45 * ts * dy), width=2,
    )

    side = dx if dx != 0 else 1.0
    if gesture == Gesture.STOP:
        palm = (cx + 0.32 * ts * side, cy - 0.40 * ts)
        pygame.draw.line(
            surface, _COLOR_STOP,
            (cx + 0.10 * ts * side, cy - 0.02 * ts), palm, 3,
        )
        pygame.draw.circle(
            surface, _COLOR_STOP, (int(palm[0]), int(palm[1])), int(ts * 0.11))
    elif gesture == Gesture.GO:
        hand = (cx + 0.40 * ts * side, cy + 0.34 * ts)
        pygame.draw.line(
            surface, _COLOR_GO,
            (cx + 0.10 * ts * side, cy + 0.06 * ts), hand, 3,
        )
        arc_rect = pygame.Rect(0, 0, int(ts * 0.9), int(ts * 0.5))
        arc_rect.center = (int(cx), int(cy + 0.20 * ts))
        pygame.draw.arc(surface, _COLOR_GO, arc_rect, math.pi, 2 * math.pi, 3)


def _draw_labels(surface, walls, ts) -> None:
    """AutoSpatial-style row/col index labels along the border walls."""
    h, w = walls.shape
    size = max(10, int(ts * 0.38))
    for r in range(1, h - 1):
        _blit_text(surface, str(r), (ts / 2, r * ts + ts / 2), size, _COLOR_LABEL)
    for c in range(1, w - 1):
        _blit_text(surface, str(c), (c * ts + ts / 2, ts / 2), size, _COLOR_LABEL)


def render_frame(env, tile_size: int = 48, annotate: bool = True,
                 show_field: bool = False) -> np.ndarray:
    """Render the env's world state to an (H*ts, W*ts, 3) uint8 RGB array.

    The boolean "annotate" adds row/col index labels along the border plus an "R" marker
    and heading arrow on the agent (for the VLM). 
    The show_field variable shades the discomfort field in red. 
    The discomfort field is drawn only in demo videos, never in advisor frames.
    """
    walls = np.asarray(env.walls)
    h, w = walls.shape
    ts = int(tile_size)
    surface = pygame.Surface((w * ts, h * ts))
    surface.fill(_COLOR_FLOOR)

    field = env.field if show_field else None
    for r in range(h):
        for c in range(w):
            rect = pygame.Rect(c * ts, r * ts, ts, ts)
            if walls[r, c] == 1:
                pygame.draw.rect(surface, _COLOR_WALL, rect)
                continue
            color = _COLOR_FLOOR if (r + c) % 2 == 0 else _COLOR_FLOOR_ALT
            if field is not None:
                v = float(field[r, c])
                if v > 0.0:
                    color = (
                        int(color[0] + (255 - color[0]) * v),
                        int(color[1] * (1.0 - 0.75 * v)),
                        int(color[2] * (1.0 - 0.75 * v)),
                    )
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, _COLOR_GRID, rect, width=2)

    gr, gc = env.goal_pos
    goal_rect = pygame.Rect(gc * ts, gr * ts, ts, ts)
    pygame.draw.rect(
        surface, _COLOR_GOAL, goal_rect.inflate(-10, -10), border_radius=8)

    for ped in env.ped_states:
        _draw_pedestrian(surface, tuple(ped.pos), ped.facing, ped.gesture, ts)

    _draw_agent(
        surface, (int(env.agent_pos[0]), int(env.agent_pos[1])),
        int(env.agent_dir), ts, annotate,
    )

    if annotate:
        _draw_labels(surface, walls, ts)

    frame = pygame.surfarray.array3d(surface)
    return np.ascontiguousarray(np.transpose(frame, (1, 0, 2))).astype(np.uint8)
