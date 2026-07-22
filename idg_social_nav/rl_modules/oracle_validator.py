"""Oracle validator: a validator that is optimized for the binary-table reward function.

It is given the same observation as the learned validator, which is a privileged view of the environment and the proposer action, 
and it decides whether to override or obey based on the discomfort of the leader's proposal. 

The decision is made by comparing the discomfort to a threshold tau, which can be set in the model configuration.
The oracle validator does not use any internal state of the environment or any LLM.

"""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from idg_social_nav.core import (
    CH_DISCOMFORT,
    CH_PED,
    CH_WALL,
    PROPOSER_TO_ENV_ACTION,
    EnvironmentAction,
    ProposerAction,
    ValidatorAction,
    to_numpy,
)
from idg_social_nav.scenarios import VIEW_RADIUS

# agent cell in the egocentric view (bottom-center) and the cell ahead
_EGO_AGENT = (VIEW_RADIUS, VIEW_RADIUS)
_EGO_AHEAD = (VIEW_RADIUS - 1, VIEW_RADIUS)


class OracleValidatorRLM(RLModule):
    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        obs = batch[SampleBatch.OBS]
        tau = float((self.model_config or {}).get("tau", 0.5))
        actions = [
            self.decide(
                to_numpy(obs["env"][i]),
                int(to_numpy(obs["proposer_action"][i]).argmax()),
                int(to_numpy(obs["advice"][i]).argmax()),
                tau,
            )
            for i in range(len(obs["env"]))
        ]
        return {SampleBatch.ACTIONS: batch_func(actions)}

    @staticmethod
    def _candidate_discomfort(
            env_view: np.ndarray, env_action: EnvironmentAction,
    ) -> float:
        """
        Returns the discomfort of the cell that would be executed by the given environment action, given the egocentric view of the environment.
        The discomfort is defined as the value of the CH_DISCOMFORT channel in the cell that would be executed by the given environment action. 
        The cell that would be executed is determined by the following rules:
        - If the action is MOVE_FORWARD and there is a pedestrian in the cell ahead, return 1.0 (the discomfort of the pedestrian cell).
        - If the action is MOVE_FORWARD and there is no wall in the cell ahead, return the discomfort of the cell ahead.
        - If the action is MOVE_FORWARD and there is a wall in the cell ahead, return the discomfort of the agent's current cell.
        - If the action is TURN_LEFT or TURN_RIGHT, return the discomfort of the agent's current cell.
        - If the action is NONE, return the discomfort of the agent's current cell.
        """
        ar, ac = _EGO_AHEAD
        r, c = _EGO_AGENT
        if env_action == EnvironmentAction.MOVE_FORWARD:
            if env_view[ar, ac, CH_PED] > 0.0:
                return 1.0
            if env_view[ar, ac, CH_WALL] <= 0.5:
                r, c = ar, ac
        return float(env_view[r, c, CH_DISCOMFORT])

    @staticmethod
    def decide(env_view: np.ndarray, proposer_action: int, advice: int,
               tau: float = 0.5) -> int:
        """
        Decide whether to override or obey based on the discomfort of the leader's proposal.
        If the discomfort is greater than or equal to tau, override; otherwise, obey.  
        """
        leader_action = PROPOSER_TO_ENV_ACTION[ProposerAction(proposer_action)]
        d_lead = OracleValidatorRLM._candidate_discomfort(env_view, leader_action)
        if d_lead >= tau:
            return ValidatorAction.override.value
        return ValidatorAction.obey.value
