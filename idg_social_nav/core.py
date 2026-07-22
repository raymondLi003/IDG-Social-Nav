"""Core types shared across the IDG social-nav package
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class EnvironmentAction(IntEnum):
    NO_OP = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    MOVE_FORWARD = 3


class ProposerAction(IntEnum):
    forward = 0
    turn_left = 1
    turn_right = 2


class ValidatorAction(IntEnum):
    obey = 0
    override = 1
    disobey = 1


class Advice(IntEnum):
    """Discrete polite suggestion"""

    NONE = 0
    TURN_LEFT = 1
    FORWARD = 2
    TURN_RIGHT = 3
    WAIT = 4


class Gesture(IntEnum):
    NONE = 0
    STOP = 1  # pedestrian asks the robot to stop
    GO = 2    # pedestrian yields 


# Orientation encoding
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

DIR_OFFSET: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

DIR_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}


PROPOSER_TO_ENV_ACTION = {
    ProposerAction.forward: EnvironmentAction.MOVE_FORWARD,
    ProposerAction.turn_left: EnvironmentAction.TURN_LEFT,
    ProposerAction.turn_right: EnvironmentAction.TURN_RIGHT,
}

ADVICE_TO_ENV_ACTION = {
    Advice.NONE: EnvironmentAction.NO_OP,
    Advice.TURN_LEFT: EnvironmentAction.TURN_LEFT,
    Advice.FORWARD: EnvironmentAction.MOVE_FORWARD,
    Advice.TURN_RIGHT: EnvironmentAction.TURN_RIGHT,
    Advice.WAIT: EnvironmentAction.NO_OP,
}


def advice_override_protocol(
        proposer_action: ProposerAction,
        validator_action: ValidatorAction,
        advice: Advice,
) -> EnvironmentAction:
    """Advice-aware override procedure
    """
    if validator_action == ValidatorAction.obey:
        try:
            return PROPOSER_TO_ENV_ACTION[ProposerAction(proposer_action)]
        except ValueError:
            raise ValueError(
                f"Invalid proposer action: {proposer_action}") from None
    elif validator_action == ValidatorAction.override:
        return ADVICE_TO_ENV_ACTION[Advice(advice)]
    raise ValueError(f"Invalid validator action: {validator_action}")


def nullify_override_protocol(
        proposer_action: ProposerAction,
        validator_action: ValidatorAction,
        advice: Advice,
) -> EnvironmentAction:
    if validator_action == ValidatorAction.obey:
        return PROPOSER_TO_ENV_ACTION[ProposerAction(proposer_action)]
    return EnvironmentAction.NO_OP


OVERRIDE_PROTOCOLS = {
    "adopt": advice_override_protocol,
    "nullify": nullify_override_protocol,
}



# VLM
VLM_HEADINGS = ("left", "straight", "right")
VLM_SPEEDS = ("slow", "stop", "constant")


def map_vlm_tokens_to_advice(heading: str, speed: str) -> Advice:
    """Map the VLM-Social-Nav vocabulary onto the grid world

    stop always maps to WAIT and slowing down while going straight has no grid
    analogue other than not advancing, so it also maps to WAIT.
    """
    heading = heading.strip().lower()
    speed = speed.strip().lower()
    if heading not in VLM_HEADINGS or speed not in VLM_SPEEDS:
        raise ValueError(f"Invalid VLM tokens: heading={heading!r} speed={speed!r}")
    if speed == "stop":
        return Advice.WAIT
    if heading == "left":
        return Advice.TURN_LEFT
    if heading == "right":
        return Advice.TURN_RIGHT
    # heading == straight
    return Advice.FORWARD if speed == "constant" else Advice.WAIT


# ---------------------------------------------------------------------------
# Observation channels
# ---------------------------------------------------------------------------

CH_WALL = 0
CH_AGENT = 1
CH_GOAL = 2
CH_PED = 3
CH_DISCOMFORT = 4

# proposer sees [wall, agent, goal, pedestrian]
# while the social-hazard discomfort channel is validator-only 
N_PROPOSER_CHANNELS = 4
N_VALIDATOR_CHANNELS = 5

# pedestrian channel encodes ego-relative facing as intensity 
# so the CNN validator and the ASCII renderer can get it
# value = 0.25 * (ego_relative_facing + 1) with facings UP/RIGHT/DOWN/LEFT
PED_FACING_VALUES = {UP: 0.25, RIGHT: 0.5, DOWN: 0.75, LEFT: 1.0}


def ped_channel_value(ped_facing: int, agent_dir: int) -> float:
    """Intensity for the pedestrian channel, encoding facing relative to the
    (rotated) egocentric frame in which the agent faces up."""
    ego_facing = (ped_facing - agent_dir) % 4
    return PED_FACING_VALUES[ego_facing]


def decode_ped_facing(value: float) -> int:
    """Inverse of ped_channel_value, returns the ego-relative facing."""
    idx = int(round(value / 0.25)) - 1
    if idx not in (UP, RIGHT, DOWN, LEFT):
        raise ValueError(f"Invalid pedestrian channel value: {value}")
    return idx



@dataclass
class PedestrianSnapshot:
    """Immutable per-step pedestrian summary handed to advisors"""
    pos: tuple[int, int]
    facing: int
    gesture: Gesture
    visible: bool


@dataclass
class AdvisorContext:
    """Everything an advisor may condition on for one gated query.

    frame_provider renders the RGB frame 
    so text-only advisors never pay the rendering cost.
    """
    scenario_name: str
    walls: np.ndarray
    agent_pos: tuple[int, int]
    agent_dir: int
    goal_pos: tuple[int, int]
    pedestrians: Sequence[PedestrianSnapshot]
    step: int
    frame_provider: Callable[[], np.ndarray] | None = None
    extras: dict = field(default_factory=dict)


class Advisor:
    """Base advisor: returns a polite suggestion B_h for a gated query.

    The env only queries when a pedestrian is within detection range (the
    grid analogue of the YOLO trigger); otherwise advice is Advice.NONE and
    the advisor is never called.
    """

    def advise(self, context: AdvisorContext) -> Advice:
        raise NotImplementedError

    def reset(self) -> None:
        """Called on env reset, the stateless advisors ignore this"""


def to_numpy(x) -> np.ndarray:
    """Torch tensor or array-like to a numpy array"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nearest_gesture(
        pedestrians: Sequence[tuple[tuple[int, int], Gesture]],
        agent_pos: tuple[int, int],
) -> Gesture:
    """Gesture of the Manhattan-nearest pedestrian, NONE when empty."""
    best, best_dist = Gesture.NONE, None
    for pos, gesture in pedestrians:
        dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
        if best_dist is None or dist < best_dist:
            best, best_dist = gesture, dist
    return Gesture(best)
