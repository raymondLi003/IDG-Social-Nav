"""Tests for the scripted validator RLModules (oracle, always-obey, fixed-blend)."""

import numpy as np
import pytest
import torch
import tree
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModuleSpec

from idg_social_nav.core import (
    CH_AGENT,
    CH_DISCOMFORT,
    CH_PED,
    CH_WALL,
    Advice,
    ProposerAction,
    ValidatorAction,
)
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.rl_modules import (
    AlwaysObeyValidatorRLM,
    FixedBlendValidatorRLM,
    OracleValidatorRLM,
    make_fixed_blend_class,
)

FWD = ProposerAction.forward.value
OBEY = ValidatorAction.obey.value
OVERRIDE = ValidatorAction.override.value


@pytest.fixture(scope="module")
def env():
    return SocialNavEnv(scenario="frontal_approach", randomize_variant=False)


def _build(env, module_class, model_config=None):
    spec = RLModuleSpec(
        module_class=module_class,
        observation_space=env.observation_spaces["validator"],
        action_space=env.action_spaces["validator"],
        model_config=model_config,
        inference_only=True,
    )
    return spec.build()


def _view(discomfort_ahead=0.0, discomfort_here=0.0, wall_ahead=False,
          ped_ahead=False):
    """Crafted validator ego view: agent at (5, 5), the cell ahead at (4, 5)."""
    view = np.zeros((6, 11, 5), dtype=np.float32)
    view[5, 5, CH_AGENT] = 1.0
    view[4, 5, CH_DISCOMFORT] = discomfort_ahead
    view[5, 5, CH_DISCOMFORT] = discomfort_here
    if wall_ahead:
        view[4, 5, CH_WALL] = 1.0
    if ped_ahead:
        view[4, 5, CH_PED] = 0.75
    return view


def _one_hot(index, size):
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def _advice_batch(advice_idx, n):
    advice = np.zeros((n, len(Advice)), dtype=np.float32)
    advice[:, advice_idx] = 1.0
    return {SampleBatch.OBS: {"advice": torch.tensor(advice)}}


def _actions(out):
    return np.asarray(out[SampleBatch.ACTIONS]).tolist()


class TestOracleDecide:
    def test_hazard_ahead_overrides(self):
        view = _view(discomfort_ahead=0.8)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OVERRIDE

    def test_safe_forward_obeys(self):
        assert OracleValidatorRLM.decide(_view(), FWD, Advice.NONE) == OBEY
        view = _view(discomfort_ahead=0.49)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OBEY

    def test_wall_ahead_stay_semantics(self):
        # the hazard behind a wall is unreachable: forward stays put
        view = _view(discomfort_ahead=0.9, wall_ahead=True)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OBEY
        # but staying on a hazardous cell is still a hazard
        view = _view(discomfort_ahead=0.9, discomfort_here=0.6, wall_ahead=True)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OVERRIDE

    def test_ped_body_ahead_is_collision(self):
        view = _view(ped_ahead=True)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OVERRIDE
        turn = ProposerAction.turn_left.value
        assert OracleValidatorRLM.decide(view, turn, Advice.NONE) == OBEY

    def test_turns_use_current_cell(self):
        view = _view(discomfort_ahead=0.9)
        for turn in (ProposerAction.turn_left, ProposerAction.turn_right):
            assert OracleValidatorRLM.decide(view, turn.value, Advice.NONE) == OBEY
        view = _view(discomfort_here=0.7)
        for turn in (ProposerAction.turn_left, ProposerAction.turn_right):
            assert OracleValidatorRLM.decide(view, turn.value, Advice.NONE) \
                == OVERRIDE

    def test_tau_threshold(self):
        view = _view(discomfort_ahead=0.5)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE) == OVERRIDE
        view = _view(discomfort_ahead=0.6)
        assert OracleValidatorRLM.decide(view, FWD, Advice.NONE, tau=0.7) == OBEY

    def test_advice_never_changes_decision(self):
        hazard = _view(discomfort_ahead=0.8)
        safe = _view()
        for advice in Advice:
            assert OracleValidatorRLM.decide(hazard, FWD, advice) == OVERRIDE
            assert OracleValidatorRLM.decide(safe, FWD, advice) == OBEY


class TestOracleModule:
    def test_forward_inference_batches_decisions(self, env):
        module = _build(env, OracleValidatorRLM)
        views = np.stack([_view(discomfort_ahead=0.8), _view()])
        proposer = np.stack([_one_hot(FWD, len(ProposerAction))] * 2)
        advice = np.stack([_one_hot(Advice.NONE, len(Advice))] * 2)
        batch = {SampleBatch.OBS: {
            "env": torch.tensor(views),
            "proposer_action": torch.tensor(proposer),
            "advice": torch.tensor(advice),
        }}
        out = module.forward_inference(batch)
        assert _actions(out) == [OVERRIDE, OBEY]


class TestAlwaysObey:
    def test_always_obeys(self, env):
        module = _build(env, AlwaysObeyValidatorRLM)
        env.reset(options={"scenario": "frontal_approach",
                           "variant": {"ped_start_col": 7, "ped_delay": 0}})
        obs, *_ = env.step({"proposer": FWD})
        batch = {SampleBatch.OBS: tree.map_structure(
            lambda x: torch.tensor(np.stack([x] * 4)), obs["validator"])}
        out = module.forward_inference(batch)
        assert _actions(out) == [OBEY] * 4


class TestFixedBlend:
    def test_p_zero_never_overrides(self, env):
        module = _build(env, FixedBlendValidatorRLM,
                        model_config={"blend_p": 0.0, "seed": 0})
        out = module.forward_inference(_advice_batch(Advice.TURN_RIGHT, 50))
        assert _actions(out) == [OBEY] * 50

    def test_p_one_always_overrides_with_advice(self, env):
        module = _build(env, FixedBlendValidatorRLM,
                        model_config={"blend_p": 1.0, "seed": 0})
        out = module.forward_inference(_advice_batch(Advice.FORWARD, 50))
        assert _actions(out) == [OVERRIDE] * 50

    def test_advice_none_never_overrides(self, env):
        module = _build(env, FixedBlendValidatorRLM,
                        model_config={"blend_p": 1.0, "seed": 0})
        out = module.forward_inference(_advice_batch(Advice.NONE, 50))
        assert _actions(out) == [OBEY] * 50

    def test_seeded_frequency_matches_p(self, env):
        module = _build(env, FixedBlendValidatorRLM,
                        model_config={"blend_p": 0.3, "seed": 123})
        out = module.forward_inference(_advice_batch(Advice.WAIT, 200))
        rate = float(np.mean(_actions(out)))
        assert rate == pytest.approx(0.3, abs=0.1)

    def test_seeded_reproducibility(self, env):
        draws = []
        for _ in range(2):
            module = _build(env, FixedBlendValidatorRLM,
                            model_config={"blend_p": 0.5, "seed": 7})
            out = module.forward_inference(_advice_batch(Advice.FORWARD, 40))
            draws.append(_actions(out))
        assert draws[0] == draws[1]
        assert OBEY in draws[0] and OVERRIDE in draws[0]

    def test_make_fixed_blend_class_bakes_defaults(self, env):
        cls = make_fixed_blend_class(1.0, seed=7)
        assert issubclass(cls, FixedBlendValidatorRLM)
        module = _build(env, cls)                      # no model_config
        assert module._p == 1.0
        out = module.forward_inference(_advice_batch(Advice.WAIT, 10))
        assert _actions(out) == [OVERRIDE] * 10
        # explicit model_config still wins over the baked defaults
        module = _build(env, cls, model_config={"blend_p": 0.0})
        out = module.forward_inference(_advice_batch(Advice.WAIT, 10))
        assert _actions(out) == [OBEY] * 10
