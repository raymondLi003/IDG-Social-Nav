"""Regression tests for the doorway-standoff deadlock
"""

import numpy as np

from idg_social_nav.core import DOWN, LEFT, RIGHT, Advice
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.eval_common import (
    _batch,
    _extract_action,
    oracle_validator_factory,
    scripted_proposer_factory,
)
from idg_social_nav.scenarios import enumerate_variants


def test_gate_is_heading_independent():
    env = SocialNavEnv(scenario="narrow_doorway", randomize_variant=False, seed=0)
    env.reset(options={"variant": {"ped_delay": 2}})
    # place the agent in the gap facing away from the pedestrian
    env.agent_pos = np.array([3, 5], dtype=np.int32)
    env.ped_states[0].pos = (3, 6)
    env.ped_states[0].facing = LEFT
    env._recompute_field()
    for direction in (RIGHT, LEFT, DOWN):
        env.agent_dir = direction
        assert env._visible_ped_indices() == [0], (
            f"pedestrian one cell away must be detected facing {direction}")
    advice, _ = env._query_advisor()
    assert advice != Advice.NONE


def test_advisor_suggests_retreat_in_standoff():
    env = SocialNavEnv(scenario="narrow_doorway", randomize_variant=False, seed=0)
    env.reset(options={"variant": {"ped_delay": 2}})
    env.agent_pos = np.array([3, 5], dtype=np.int32)
    env.agent_dir = LEFT  # already turned around, ready to back out
    env.ped_states[0].pos = (3, 6)
    env.ped_states[0].facing = LEFT
    env._recompute_field()
    advice, _ = env._query_advisor()
    assert advice == Advice.FORWARD  # step out of the gap, away from the ped


def test_oracle_resolves_every_doorway_variant():
    for variant in enumerate_variants("narrow_doorway"):
        env = SocialNavEnv(scenario="narrow_doorway", randomize_variant=False, seed=0)
        obs, _ = env.reset(options={"variant": dict(variant)})
        proposer = scripted_proposer_factory(env)
        validator = oracle_validator_factory(env)
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        collisions = 0
        while not (terminated["__all__"] or truncated["__all__"]):
            if "proposer" in obs:
                out = proposer.forward_inference(_batch(obs["proposer"]))
                obs, _, terminated, truncated, _ = env.step(
                    {"proposer": _extract_action(proposer, out)})
            else:
                out = validator.forward_inference(_batch(obs["validator"]))
                obs, _, terminated, truncated, infos = env.step(
                    {"validator": _extract_action(validator, out)})
                collisions += int(infos["validator"]["collision_attempt"])
        assert env.last_info["reached_goal"], f"froze on variant {variant}"
        assert collisions == 0, f"collision on variant {variant}"
