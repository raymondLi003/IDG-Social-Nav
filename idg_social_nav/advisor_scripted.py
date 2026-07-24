"""Deterministic rule-based social advisor.

Simulates a socially competent VLM without any API call and suggests the lowest
discomfort move that still makes goal progress

Used as the default advisor for RL validator training, the oracle in the
error decomposition, and the fallback when there's no vlm backend
"""

from __future__ import annotations

import numpy as np

from idg_social_nav.core import (
    DIR_OFFSET,
    Advice,
    Advisor,
    AdvisorContext,
)
from idg_social_nav.discomfort import DiscomfortParams, discomfort_field
from idg_social_nav.grid import bfs_distances, bfs_first_step, step_direction

# preference order on equal goal progress: forward, then keep right
_PREFERENCE_RANK = {
    Advice.FORWARD: 0,
    Advice.TURN_RIGHT: 1,
    Advice.TURN_LEFT: 2,
    Advice.WAIT: 3,
}


class ScriptedSocialAdvisor(Advisor):
    def __init__(self, params: DiscomfortParams | None = None):
        self.params = params or DiscomfortParams()

    def advise(self, context: AdvisorContext) -> Advice:
        visible = [p for p in context.pedestrians if p.visible]
        if not visible:
            return Advice.NONE

        field = discomfort_field(
            context.walls,
            [(p.pos, p.facing, p.gesture) for p in visible],
            self.params,
        )
        tau = self.params.high_threshold
        ped_cells = {tuple(p.pos) for p in context.pedestrians}
        dist = bfs_distances(context.walls, context.goal_pos, ped_cells)

        ar, ac = context.agent_pos
        d = context.agent_dir
        candidate_cells = {
            Advice.FORWARD: (ar + DIR_OFFSET[d][0], ac + DIR_OFFSET[d][1]),
            Advice.TURN_RIGHT: (ar + DIR_OFFSET[(d + 1) % 4][0],
                                ac + DIR_OFFSET[(d + 1) % 4][1]),
            Advice.TURN_LEFT: (ar + DIR_OFFSET[(d - 1) % 4][0],
                               ac + DIR_OFFSET[(d - 1) % 4][1]),
            Advice.WAIT: (ar, ac),
        }

        scored = []
        for advice, cell in candidate_cells.items():
            r, c = cell
            if advice != Advice.WAIT:
                if context.walls[r, c] == 1 or cell in ped_cells:
                    continue
            discomfort = float(field[r, c])
            goal_dist = float(dist[r, c])
            scored.append((advice, discomfort, goal_dist))

        safe = [s for s in scored if s[1] < tau and np.isfinite(s[2])]
        if safe:
            return safe[0][0]

        escape = self._escape_advice(context, field, tau, ped_cells)
        if escape is not None:
            return escape

        # if fully boxed in high discomfort zone,
        # minimize discomfort and progress
        scored.sort(key=lambda s: (s[1], s[2], _PREFERENCE_RANK[s[0]]))
        return scored[0][0] if scored else Advice.WAIT

    @staticmethod
    def _escape_advice(
            context: AdvisorContext,
            field: np.ndarray,
            tau: float,
            ped_cells: set[tuple[int, int]],
    ) -> Advice | None:
        """First move toward the nearest reachable cell below tau
        WAIT when already standing on one
        None when no low-discomfort cell is reachable at all."""
        start = tuple(context.agent_pos)
        if field[start] < tau:
            return Advice.WAIT

        step, _ = bfs_first_step(
            context.walls, start, lambda cell: field[cell] < tau, ped_cells)
        if step is None:
            return None

        desired = step_direction(start, step)
        diff = (desired - context.agent_dir) % 4
        if diff == 0:
            return Advice.FORWARD
        if diff == 3:
            return Advice.TURN_LEFT
        # keep to the right as a convention
        return Advice.TURN_RIGHT


class NoisyAdvisor(Advisor):
    """Wraps an advisor and flips its suggestion with probability epsilon
    so we can do a controlled stress test for validator robustness"""

    def __init__(self, base: Advisor, epsilon: float, seed=None):
        assert 0.0 <= epsilon <= 1.0
        self.base = base
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def advise(self, context: AdvisorContext) -> Advice:
        advice = self.base.advise(context)
        if advice != Advice.NONE and self.rng.random() < self.epsilon:
            alternatives = [a for a in Advice if a not in (advice, Advice.NONE)]
            advice = Advice(alternatives[int(self.rng.integers(0, len(alternatives)))])
        return advice

    def reset(self) -> None:
        self.base.reset()
