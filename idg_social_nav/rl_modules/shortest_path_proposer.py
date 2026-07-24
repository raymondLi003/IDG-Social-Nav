"""BFS shortest-path leader: the scripted goal planner.

The proposer sees the same privileged view as the learned proposer,
and it uses BFS to find a shortest path to the goal under the layout
that is most consistent with the current view.
The proposer avoids cells with pedestrians in view, but it does not know their future positions,
so it may propose a path that collides with a pedestrian.
The proposer does not use any internal state of the environment or any LLM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from idg_social_nav.core import CH_PED, CH_WALL, ProposerAction, to_numpy
from idg_social_nav.env import egocentric_view
from idg_social_nav.grid import bfs_next_toward, manhattan, step_direction
from idg_social_nav.scenarios import SCENARIO_NAMES, VIEW_RADIUS, make_scenario


def _build_layout_registry() -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Unique (walls, goal) pairs over the scenario suite."""
    layouts: list[tuple[np.ndarray, tuple[int, int]]] = []
    seen: set[bytes] = set()
    for name in SCENARIO_NAMES:
        cfg = make_scenario(name)
        key = cfg.walls.tobytes()
        if key in seen:
            continue
        seen.add(key)
        layouts.append((np.array(cfg.walls, dtype=np.uint8), tuple(cfg.goal_pos)))
    return layouts


_LAYOUTS = _build_layout_registry()


class ShortestPathProposerRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These are the layout index and position of the last step,
        # used to maintain continuity across steps
        self._layout_idx: int | None = None
        self._last_pos: tuple[int, int] | None = None

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        obs_env = batch[SampleBatch.OBS]["env"]
        obs_pose = batch[SampleBatch.OBS]["pose"]
        actions = [
            self._get_action(to_numpy(obs_env[i]), to_numpy(obs_pose[i]))
            for i in range(len(obs_env))
        ]
        return {SampleBatch.ACTIONS: batch_func(actions)}


    # localization
    @staticmethod
    def _candidates(
            wall_view: np.ndarray, pose: np.ndarray,
    ) -> list[tuple[int, tuple[int, int], int]]:
        """
        Returns a list of candidate layouts that are consistent with the
        current egocentric wall view and pose.
        Each candidate is a tuple of (layout index, position, direction).
        The position is the (row, col) of the agent in the layout,
        and the direction is the agent's facing direction (0: up, 1: right, 2: down, 3: left).
        """
        direction = int(np.argmax(pose[2:6]))
        observed = wall_view > 0.5
        out = []
        for idx, (walls, _goal) in enumerate(_LAYOUTS):
            h, w = walls.shape
            row_f = float(pose[0]) * (h - 1)
            col_f = float(pose[1]) * (w - 1)
            row, col = int(round(row_f)), int(round(col_f))
            if abs(row_f - row) > 1e-3 or abs(col_f - col) > 1e-3:
                continue
            if not (0 <= row < h and 0 <= col < w) or walls[row, col] == 1:
                continue
            pred = egocentric_view(
                walls, (row, col), direction,
                np.zeros((h, w, 0), dtype=np.float32), VIEW_RADIUS,
            )
            if np.array_equal(pred[..., 0] > 0.5, observed):
                out.append((idx, (row, col), direction))
        return out


    # planning
    @staticmethod
    def _visible_ped_cells(
            obs: np.ndarray, pos: tuple[int, int], direction: int,
            shape: tuple[int, int],
    ) -> set[tuple[int, int]]:
        """
        Returns the set of cells that contain pedestrians and are visible
        from the agent's current position and direction.
        The visibility is determined by the egocentric view of the environment,
        and the cells are transformed back to the global coordinates of the layout."""
        cells = set()
        for er, ec in np.argwhere(obs[..., CH_PED] > 0.0):
            a = int(er) - VIEW_RADIUS
            b = int(ec) - VIEW_RADIUS
            for _ in range(direction):
                a, b = b, -a
            r, c = pos[0] + a, pos[1] + b
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                cells.add((r, c))
        return cells

    @staticmethod
    def _action_toward(
            pos: tuple[int, int], direction: int, next_cell: tuple[int, int],
    ) -> int:
        desired = step_direction(pos, next_cell)
        if desired is None or desired == direction:
            return ProposerAction.forward.value
        diff = (desired - direction) % 4
        if diff == 1:
            return ProposerAction.turn_right.value
        if diff == 3:
            return ProposerAction.turn_left.value
        # 180 degrees: either way works. turn right by default
        return ProposerAction.turn_right.value


    # per-step action selection
    def _get_action(self, obs: np.ndarray, pose: np.ndarray) -> int:
        candidates = self._candidates(obs[..., CH_WALL], pose)
        if not candidates:
            return ProposerAction.forward.value

        chosen = None
        if self._layout_idx is not None and self._last_pos is not None:
            for cand in candidates:
                if (cand[0] == self._layout_idx
                        and manhattan(cand[1], self._last_pos) <= 1):
                    chosen = cand
                    break
        if chosen is None:
            # choose the candidate with the largest number of walls in
            # view, as a heuristic for the most consistent layout
            chosen = max(candidates, key=lambda cand: int(_LAYOUTS[cand[0]][0].sum()))
        layout_idx, pos, direction = chosen
        self._layout_idx = layout_idx
        self._last_pos = pos

        walls, goal = _LAYOUTS[layout_idx]
        ped_cells = self._visible_ped_cells(obs, pos, direction, walls.shape)
        next_cell = bfs_next_toward(walls, pos, goal, ped_cells)
        if next_cell is None and pos != goal:
            next_cell = bfs_next_toward(walls, pos, goal)
        if next_cell is None:
            return ProposerAction.forward.value
        return self._action_toward(pos, direction, next_cell)
