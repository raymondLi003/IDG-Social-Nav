"""The four VLM-Social-Nav benchmark scenarios as grid episodes.

  - frontal_approach: open corridor, pedestrian walking toward the agent
  
  - narrow_doorway: 1-wide gap, pedestrian approaching from the far side

  - intersection: perpendicular rails crossing the agent's corridor.
  - frontal_gesture: frontal approach where the pedestrian displays STOP
    (the pedestrian asserts right of way and keeps walking) or GO (the pedestrian yields and waits).

Scenarios are deterministic, which makes advisor calls
enumerable and cacheable offline and keeps the oracle well-defined.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from idg_social_nav.core import DIR_OFFSET, RIGHT, UP, Gesture
from idg_social_nav.discomfort import DiscomfortParams

# uniform egocentric radius so observation shapes match across scenarios
VIEW_RADIUS = 5
DEFAULT_MAX_STEPS = 40

# The pedestrian yields to the agent when the agent is in front and within this Manhattan distance.
# this is when the pedestrian's GO gesture is active, and the agent should proceed.
YIELD_DISTANCE = 3


def facing_toward(src: tuple[int, int], dst: tuple[int, int], fallback: int) -> int:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for d, (odr, odc) in DIR_OFFSET.items():
        if (odr, odc) == (np.sign(dr), np.sign(dc)):
            return d
    return fallback


@dataclass(frozen=True)
class PedestrianConfig:
    """Scripted pedestrian: a rail of cells walked one cell per env turn."""
    rail: tuple[tuple[int, int], ...]
    gesture: Gesture = Gesture.NONE
    start_delay: int = 0
    initial_facing: int | None = None  # derived from the rail when None


@dataclass
class PedestrianState:
    """Mutable per-episode pedestrian state."""
    config: PedestrianConfig
    pos: tuple[int, int]
    facing: int
    delay_remaining: int
    rail_idx: int = 0

    @classmethod
    def from_config(cls, config: PedestrianConfig) -> PedestrianState:
        start = config.rail[0]
        if config.initial_facing is not None:
            facing = config.initial_facing
        elif len(config.rail) > 1:
            facing = facing_toward(start, config.rail[1], UP)
        else:
            facing = UP
        return cls(
            config=config,
            pos=start,
            facing=facing,
            delay_remaining=config.start_delay,
        )

    @property
    def gesture(self) -> Gesture:
        return self.config.gesture

    def _agent_passed(self, agent_pos: tuple[int, int]) -> bool:
        """True when the agent is behind the pedestrian (negative projection
        of the offset onto the pedestrian's facing)."""
        fr, fc = DIR_OFFSET[self.facing]
        dr = agent_pos[0] - self.pos[0]
        dc = agent_pos[1] - self.pos[1]
        return (dr * fr + dc * fc) < 0

    def _should_yield(self, agent_pos: tuple[int, int]) -> bool:
        if self.gesture != Gesture.GO:
            return False
        if self._agent_passed(agent_pos):
            return False
        dist = abs(agent_pos[0] - self.pos[0]) + abs(agent_pos[1] - self.pos[1])
        return dist <= YIELD_DISTANCE

    def step(self, agent_pos: tuple[int, int], occupied: set[tuple[int, int]]) -> None:
        """Advance one cell along the rail.
        The pedestrian yields to the agent when the agent is in front and within the YIELD_DISTANCE (GO gesture). 
        The pedestrian does not step onto occupied cells.
        
        """
        if self.delay_remaining > 0:
            self.delay_remaining -= 1
            return
        if self.rail_idx >= len(self.config.rail) - 1:
            return
        if self._should_yield(agent_pos):
            return
        nxt = self.config.rail[self.rail_idx + 1]
        if nxt == tuple(agent_pos) or nxt in occupied:
            return
        self.facing = facing_toward(self.pos, nxt, self.facing)
        self.rail_idx += 1
        self.pos = nxt


@dataclass(frozen=True)
class ScenarioConfig:
    """Deterministic episode specification."""
    name: str
    variant: dict
    walls: np.ndarray
    agent_start: tuple[int, int]
    agent_dir: int
    goal_pos: tuple[int, int]
    pedestrians: tuple[PedestrianConfig, ...]
    max_steps: int = DEFAULT_MAX_STEPS
    view_radius: int = VIEW_RADIUS
    discomfort_params: DiscomfortParams = field(default_factory=DiscomfortParams)


def _bordered_grid(height: int, width: int) -> np.ndarray:
    walls = np.zeros((height, width), dtype=np.uint8)
    walls[0, :] = 1
    walls[-1, :] = 1
    walls[:, 0] = 1
    walls[:, -1] = 1
    return walls


def _row_rail(row: int, start_col: int, end_col: int) -> tuple[tuple[int, int], ...]:
    step = 1 if end_col >= start_col else -1
    return tuple((row, c) for c in range(start_col, end_col + step, step))


def _col_rail(col: int, start_row: int, end_row: int) -> tuple[tuple[int, int], ...]:
    step = 1 if end_row >= start_row else -1
    return tuple((r, col) for r in range(start_row, end_row + step, step))


# build the scenarios
def _frontal_approach(variant: dict) -> ScenarioConfig:
    """7x11 open room
    the pedestrian walks down the agent's row toward the agent, starting from the far side."""
    walls = _bordered_grid(7, 11)
    ped = PedestrianConfig(
        rail=_row_rail(3, variant["ped_start_col"], 1),
        start_delay=variant["ped_delay"],
    )
    return ScenarioConfig(
        name="frontal_approach",
        variant=dict(variant),
        walls=walls,
        agent_start=(3, 1),
        agent_dir=RIGHT,
        goal_pos=(3, 9),
        pedestrians=(ped,),
    )


def _narrow_doorway(variant: dict) -> ScenarioConfig:
    """Wall bisecting the room with a 1-wide gap
    the pedestrian comes through from the far side. 
    Timing decides who passes first."""
    walls = _bordered_grid(7, 11)
    walls[:, 5] = 1
    walls[3, 5] = 0
    ped = PedestrianConfig(
        rail=_row_rail(3, 8, 2),
        start_delay=variant["ped_delay"],
    )
    return ScenarioConfig(
        name="narrow_doorway",
        variant=dict(variant),
        walls=walls,
        agent_start=(3, 1),
        agent_dir=RIGHT,
        goal_pos=(3, 9),
        pedestrians=(ped,),
    )


def _intersection(variant: dict) -> ScenarioConfig:
    """Perpendicular one-cell wide corridors
    the pedestrian crosses the agent's corridor at the junction."""
    walls = _bordered_grid(9, 11)
    walls[1:-1, 1:-1] = 1
    walls[4, 1:-1] = 0   # agent corridor
    walls[1:-1, 5] = 0   # pedestrian corridor
    if variant["ped_direction"] == "down":
        rail = _col_rail(5, 1, 7)
    else:
        rail = _col_rail(5, 7, 1)
    ped = PedestrianConfig(rail=rail, start_delay=variant["ped_delay"])
    return ScenarioConfig(
        name="intersection",
        variant=dict(variant),
        walls=walls,
        agent_start=(4, 1),
        agent_dir=RIGHT,
        goal_pos=(4, 9),
        pedestrians=(ped,),
    )


def _frontal_gesture(variant: dict) -> ScenarioConfig:
    """Frontal approach where the pedestrian displays an explicit gesture.

    STOP: the pedestrian asserts right of way and keeps walking. 
    Its frontal discomfort zone is extended, so the agent should clear the lane early.
    GO: the pedestrian yields (waits while the agent is close and in front). 
    Its frontal zone shrinks, so the agent should simply proceed.
    """
    base = _frontal_approach(
        {"ped_start_col": variant["ped_start_col"], "ped_delay": variant["ped_delay"]}
    )
    gesture = Gesture[variant["gesture"]]
    ped = replace(base.pedestrians[0], gesture=gesture)
    return replace(
        base,
        name="frontal_gesture",
        variant=dict(variant),
        pedestrians=(ped,),
    )


_SCENARIO_BUILDERS = {
    "frontal_approach": _frontal_approach,
    "narrow_doorway": _narrow_doorway,
    "intersection": _intersection,
    "frontal_gesture": _frontal_gesture,
}

_VARIANT_SPACES: dict[str, dict[str, list]] = {
    "frontal_approach": {
        "ped_start_col": [7, 8],
        "ped_delay": [0, 1, 2],
    },
    "narrow_doorway": {
        "ped_delay": [0, 1, 2, 3],
    },
    "intersection": {
        "ped_direction": ["down", "up"],
        "ped_delay": [0, 1, 2, 3],
    },
    "frontal_gesture": {
        "gesture": ["STOP", "GO"],
        "ped_start_col": [7, 8],
        "ped_delay": [0, 1],
    },
}

SCENARIO_NAMES = tuple(_SCENARIO_BUILDERS)


def enumerate_variants(name: str) -> list[dict]:
    """All variant dicts of a scenario."""
    space = _VARIANT_SPACES[name]
    keys = list(space)
    return [dict(zip(keys, values, strict=True))
            for values in itertools.product(*space.values())]


def sample_variant(name: str, rng: np.random.Generator) -> dict:
    variants = enumerate_variants(name)
    return variants[int(rng.integers(0, len(variants)))]


def make_scenario(name: str, variant: dict | None = None,
                  rng: np.random.Generator | None = None) -> ScenarioConfig:
    if name not in _SCENARIO_BUILDERS:
        raise ValueError(f"Unknown scenario: {name!r}. Known: {SCENARIO_NAMES}")
    if variant is None:
        if rng is not None:
            variant = sample_variant(name, rng)
        else:
            variant = enumerate_variants(name)[0]
    else:
        expected = set(_VARIANT_SPACES[name])
        if set(variant) != expected:
            raise ValueError(
                f"Variant keys {set(variant)} do not match {expected} for {name}")
    return _SCENARIO_BUILDERS[name](variant)
