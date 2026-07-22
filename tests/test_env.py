"""Tests for env.py: the SocialNavEnv wrapper around the core scenario simulator."""

import numpy as np
import pytest

from idg_social_nav.core import (
    CH_AGENT,
    CH_PED,
    DOWN,
    Advice,
    Advisor,
    EnvironmentAction,
    Gesture,
    ProposerAction,
    ValidatorAction,
    decode_ped_facing,
)
from idg_social_nav.env import SocialNavEnv

FWD = ProposerAction.forward.value
OBEY = ValidatorAction.obey.value
OVERRIDE = ValidatorAction.override.value

FRONTAL_7 = {"scenario": "frontal_approach",
             "variant": {"ped_start_col": 7, "ped_delay": 0}}
FRONTAL_8 = {"scenario": "frontal_approach",
             "variant": {"ped_start_col": 8, "ped_delay": 0}}

INFO_KEYS = {
    "turn", "scenario", "variant", "proposer_action", "validator_action",
    "advice", "executed_action", "d_lead", "d_exec", "overridden",
    "good_override", "bad_override", "failed_override", "missed_hazard",
    "collision_attempt", "intrusion", "gesture", "reached_goal",
}


class StubAdvisor(Advisor):
    """Fixed-advice advisor; counts gated queries."""

    def __init__(self, advice: Advice):
        self.advice = advice
        self.calls = 0

    def advise(self, context) -> Advice:
        self.calls += 1
        return self.advice


def _make_env(**kwargs):
    kwargs.setdefault("scenario", "frontal_approach")
    kwargs.setdefault("randomize_variant", False)
    return SocialNavEnv(**kwargs)


def _run_turns(env, seq, options=FRONTAL_7):
    """
    Run a sequence of proposer/validator turns in the given env, 
    returning the final obs, rewards, terminated, truncated, infos.
    """
    env.reset(options=options)
    result = None
    for proposer_action, validator_action in seq:
        env.step({"proposer": proposer_action})
        result = env.step({"validator": validator_action})
    return result


class TestSpaces:
    def test_space_shapes(self):
        env = _make_env()
        proposer = env.observation_spaces["proposer"]
        validator = env.observation_spaces["validator"]
        assert proposer["env"].shape == (6, 11, 4)
        assert proposer["pose"].shape == (6,)
        assert proposer["validator_action"].shape == (2,)
        assert validator["env"].shape == (6, 11, 5)
        assert validator["pose"].shape == (6,)
        assert validator["proposer_action"].shape == (3,)
        assert validator["advice"].shape == (5,)
        assert validator["gesture"].shape == (3,)
        assert env.action_spaces["proposer"].n == len(ProposerAction)
        assert env.action_spaces["validator"].n == len(ValidatorAction)

    def test_obs_within_spaces_and_dtypes(self):
        env = _make_env()
        obs, _ = env.reset(options=FRONTAL_7)
        assert env.observation_spaces["proposer"].contains(obs["proposer"])
        for arr in obs["proposer"].values():
            assert arr.dtype == np.float32
        obs, *_ = env.step({"proposer": FWD})
        assert env.observation_spaces["validator"].contains(obs["validator"])
        for arr in obs["validator"].values():
            assert arr.dtype == np.float32

    def test_proposer_sees_discomfort_flag(self):
        env = _make_env(proposer_sees_discomfort=True)
        assert env.observation_spaces["proposer"]["env"].shape == (6, 11, 5)


class TestTurnProtocol:
    def test_alternating_agents(self):
        env = _make_env()
        obs, _ = env.reset(options=FRONTAL_7)
        assert set(obs) == {"proposer"}
        obs, rewards, terminated, truncated, infos = env.step({"proposer": FWD})
        assert set(obs) == {"validator"}
        assert rewards == {}
        assert infos == {}
        assert not terminated["__all__"]
        obs, rewards, terminated, truncated, infos = env.step({"validator": OBEY})
        assert set(obs) == {"proposer"}
        assert set(rewards) == {"proposer", "validator"}
        assert set(infos) == {"proposer", "validator"}

    def test_validator_sees_proposal_one_hot(self):
        env = _make_env()
        env.reset(options=FRONTAL_7)
        obs, *_ = env.step({"proposer": ProposerAction.turn_left.value})
        one_hot = obs["validator"]["proposer_action"]
        assert int(np.argmax(one_hot)) == ProposerAction.turn_left
        assert one_hot.sum() == pytest.approx(1.0)

    def test_invalid_action_dict_raises(self):
        env = _make_env()
        env.reset(options=FRONTAL_7)
        with pytest.raises(ValueError):
            env.step({"nobody": 0})

    def test_reset_options_select_scenario_and_variant(self):
        env = SocialNavEnv(scenario="all", seed=0)
        env.reset(options={"scenario": "narrow_doorway",
                           "variant": {"ped_delay": 2}})
        assert env.scenario.name == "narrow_doorway"
        assert env.scenario.variant == {"ped_delay": 2}


class TestAdviceGating:
    def test_no_advice_until_ped_visible(self):
        stub = StubAdvisor(Advice.TURN_RIGHT)
        env = _make_env(advisor=stub)
        env.reset(options=FRONTAL_7)
        # ped at (3, 7), agent at (3, 1): outside the ego view -> no query
        obs, *_ = env.step({"proposer": FWD})
        assert stub.calls == 0
        assert int(np.argmax(obs["validator"]["advice"])) == Advice.NONE
        env.step({"validator": OBEY})
        # agent (3, 2), ped (3, 6): inside the view -> gated query fires
        obs, *_ = env.step({"proposer": FWD})
        assert stub.calls == 1
        assert int(np.argmax(obs["validator"]["advice"])) == Advice.TURN_RIGHT

    def test_override_executes_advice(self):
        stub = StubAdvisor(Advice.TURN_RIGHT)
        env = _make_env(advisor=stub)
        _, _, _, _, infos = _run_turns(env, [(FWD, OBEY), (FWD, OVERRIDE)])
        info = infos["validator"]
        assert info["overridden"]
        assert info["advice"] == Advice.TURN_RIGHT
        assert info["executed_action"] == EnvironmentAction.TURN_RIGHT
        assert env.agent_dir == DOWN
        assert tuple(env.agent_pos) == (3, 2)

    def test_gesture_one_hot_when_visible(self):
        env = SocialNavEnv(scenario="frontal_gesture", randomize_variant=False)
        options = {"scenario": "frontal_gesture",
                   "variant": {"gesture": "STOP", "ped_start_col": 7,
                               "ped_delay": 0}}
        env.reset(options=options)
        obs, *_ = env.step({"proposer": FWD})
        assert int(np.argmax(obs["validator"]["gesture"])) == Gesture.NONE
        env.step({"validator": OBEY})
        obs, *_ = env.step({"proposer": FWD})
        assert int(np.argmax(obs["validator"]["gesture"])) == Gesture.STOP


class TestRewardTable:
    def test_obey_safe_is_zero(self):
        env = _make_env()
        _, rewards, _, _, infos = _run_turns(env, [(FWD, OBEY)])
        assert rewards["validator"] == 0.0
        info = infos["validator"]
        assert not info["overridden"]
        assert not (info["good_override"] or info["bad_override"]
                    or info["failed_override"] or info["missed_hazard"])

    def test_missed_hazard(self):
        env = _make_env()
        _, rewards, _, _, infos = _run_turns(env, [(FWD, OBEY)] * 3)
        assert rewards["validator"] == -1.0
        info = infos["validator"]
        assert info["missed_hazard"]
        assert not info["overridden"]
        assert info["d_lead"] == pytest.approx(2 / 3, abs=1e-5)

    def test_good_override(self):
        env = _make_env()
        _, rewards, _, _, infos = _run_turns(
            env, [(FWD, OBEY), (FWD, OBEY), (FWD, OVERRIDE)])
        assert rewards["validator"] == 1.0
        info = infos["validator"]
        assert info["good_override"]
        assert info["d_lead"] >= 0.5
        assert info["d_exec"] < 0.5
        # scripted advisor sidesteps: the override turned the agent
        assert info["advice"] == Advice.TURN_RIGHT
        assert info["executed_action"] == EnvironmentAction.TURN_RIGHT
        assert env.agent_dir == DOWN

    def test_bad_override(self):
        env = _make_env()
        _, rewards, _, _, infos = _run_turns(env, [(FWD, OVERRIDE)])
        assert rewards["validator"] == -1.0
        info = infos["validator"]
        assert info["bad_override"]
        assert info["d_lead"] < 0.5
        # advice NONE: the override degenerates to a nullify
        assert info["advice"] == Advice.NONE
        assert info["executed_action"] == EnvironmentAction.NO_OP
        assert tuple(env.agent_pos) == (3, 1)

    def test_failed_override(self):
        env = _make_env(advisor=StubAdvisor(Advice.WAIT))
        seq = [(FWD, OBEY), (FWD, OBEY),
               (FWD, OVERRIDE), (FWD, OVERRIDE), (FWD, OVERRIDE)]
        _, rewards, _, _, infos = _run_turns(env, seq, options=FRONTAL_8)
        assert rewards["validator"] == 0.0
        info = infos["validator"]
        assert info["failed_override"]
        assert info["d_lead"] == 1.0          # forward into the pedestrian
        assert info["d_exec"] >= 0.5          # waiting inside the zone

    def test_collision_attempt_on_obeyed_forward_into_ped(self):
        env = _make_env(advisor=StubAdvisor(Advice.WAIT))
        seq = [(FWD, OBEY), (FWD, OBEY),
               (FWD, OVERRIDE), (FWD, OVERRIDE), (FWD, OBEY)]
        _, rewards, _, _, infos = _run_turns(env, seq, options=FRONTAL_8)
        info = infos["validator"]
        assert rewards["validator"] == -1.0
        assert info["missed_hazard"]
        assert info["collision_attempt"]
        assert info["d_lead"] == 1.0
        assert tuple(env.agent_pos) == (3, 3)  # blocked by the body


class TestTruncation:
    def test_truncation_injects_both_obs(self):
        env = _make_env(max_steps=2)
        env.reset(options=FRONTAL_7)
        env.step({"proposer": FWD})
        obs, _, terminated, truncated, _ = env.step({"validator": OBEY})
        assert not truncated["__all__"]
        env.step({"proposer": FWD})
        obs, _, terminated, truncated, _ = env.step({"validator": OBEY})
        assert truncated["__all__"]
        assert truncated["proposer"] and truncated["validator"]
        assert not terminated["__all__"]
        assert set(obs) == {"proposer", "validator"}
        assert env.observation_spaces["proposer"].contains(obs["proposer"])
        assert env.observation_spaces["validator"].contains(obs["validator"])


class TestInfo:
    def test_info_keys_complete_on_both_agents(self):
        env = _make_env()
        _, _, _, _, infos = _run_turns(env, [(FWD, OBEY)])
        for agent in ("proposer", "validator"):
            assert INFO_KEYS <= set(infos[agent])
        info = infos["validator"]
        assert info["turn"] == 1
        assert info["scenario"] == "frontal_approach"
        assert info["variant"] == FRONTAL_7["variant"]
        assert info["proposer_action"] == FWD
        assert info["validator_action"] == OBEY
        assert not info["reached_goal"]


class TestPedChannel:
    def test_facing_encoding_in_ego_view(self):
        env = _make_env()
        env.reset(options=FRONTAL_7)
        env.step({"proposer": FWD})
        env.step({"validator": OBEY})
        obs, *_ = env.step({"proposer": FWD})
        view = obs["validator"]["env"]
        assert view[5, 5, CH_AGENT] == 1.0     # agent at bottom-center
        # agent (3, 2) facing right, ped (3, 6) facing left: 4 cells ahead
        ped_cells = np.argwhere(view[..., CH_PED] > 0)
        assert ped_cells.tolist() == [[1, 5]]
        value = float(view[1, 5, CH_PED])
        assert value == pytest.approx(0.75)    # walking toward the agent
        assert decode_ped_facing(value) == DOWN


class TestDeterminism:
    def test_identical_rollouts(self):
        seq = [(FWD, OBEY), (FWD, OVERRIDE), (FWD, OBEY), (FWD, OBEY)]
        traces = []
        for _ in range(2):
            env = _make_env()
            env.reset(options=FRONTAL_7)
            trace = []
            for proposer_action, validator_action in seq:
                val_obs, *_ = env.step({"proposer": proposer_action})
                prop_obs, rewards, _, _, infos = env.step(
                    {"validator": validator_action})
                trace.append((val_obs["validator"], prop_obs["proposer"],
                              rewards, infos["validator"]))
            traces.append(trace)
        for (vo1, po1, r1, i1), (vo2, po2, r2, i2) in zip(*traces, strict=True):
            for key in vo1:
                np.testing.assert_array_equal(vo1[key], vo2[key])
            for key in po1:
                np.testing.assert_array_equal(po1[key], po2[key])
            assert r1 == r2
            assert i1 == i2
