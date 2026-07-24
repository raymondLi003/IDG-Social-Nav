"""Tests for stochastic pedestrian hesitation (ped_hesitation).
"""

import numpy as np
import pytest

from idg_social_nav.config import SocialAgentConfig, env_config_for, experiment_name
from idg_social_nav.core import ProposerAction
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.scenarios import PedestrianConfig, PedestrianState


def _fresh_ped() -> PedestrianState:
    rail = tuple((3, c) for c in range(8, 1, -1))
    return PedestrianState.from_config(PedestrianConfig(rail=rail))


AGENT_FAR = (1, 1)


class TestPedestrianStepHesitation:
    def test_default_step_needs_no_rng(self):
        ped = _fresh_ped()
        ped.step(AGENT_FAR, set())
        assert ped.rail_idx == 1

    def test_hesitation_one_never_moves(self):
        ped = _fresh_ped()
        rng = np.random.default_rng(0)
        for _ in range(20):
            ped.step(AGENT_FAR, set(), rng=rng, hesitation_p=1.0)
        assert ped.rail_idx == 0

    def test_hesitation_zero_always_moves(self):
        ped = _fresh_ped()
        rng = np.random.default_rng(0)
        for _ in range(3):
            ped.step(AGENT_FAR, set(), rng=rng, hesitation_p=0.0)
        assert ped.rail_idx == 3

    def test_hesitating_ped_stays_on_rail(self):
        ped = _fresh_ped()
        rng = np.random.default_rng(7)
        rail_cells = set(ped.config.rail)
        for _ in range(30):
            ped.step(AGENT_FAR, set(), rng=rng, hesitation_p=0.5)
            assert ped.pos in rail_cells

    def test_delay_counts_down_before_hesitation_applies(self):
        rail = tuple((3, c) for c in range(8, 1, -1))
        ped = PedestrianState.from_config(
            PedestrianConfig(rail=rail, start_delay=2))
        rng = np.random.default_rng(0)
        ped.step(AGENT_FAR, set(), rng=rng, hesitation_p=1.0)
        ped.step(AGENT_FAR, set(), rng=rng, hesitation_p=1.0)
        assert ped.delay_remaining == 0
        assert ped.rail_idx == 0


def _run_episode(env: SocialNavEnv) -> list[tuple[int, int]]:
    """Roll one always-forward/always-obey episode; return the ped path."""
    obs, _ = env.reset(options={
        "scenario": "frontal_approach",
        "variant": {"ped_start_col": 8, "ped_delay": 0},
    })
    path = []
    while True:
        if "proposer" in obs:
            actions = {"proposer": int(ProposerAction.forward)}
        else:
            actions = {"validator": 0}  # always obey
        obs, _, terminated, truncated, _ = env.step(actions)
        path.append(tuple(env.ped_states[0].pos))
        if terminated["__all__"] or truncated["__all__"]:
            return path


class TestEnvHesitation:
    def test_rejects_invalid_probability(self):
        with pytest.raises(ValueError):
            SocialNavEnv(scenario="frontal_approach", ped_hesitation=1.5)

    def test_same_seed_reproduces_episode(self):
        paths = []
        for _ in range(2):
            env = SocialNavEnv(scenario="frontal_approach",
                               randomize_variant=False,
                               ped_hesitation=0.5, seed=123)
            paths.append(_run_episode(env))
        assert paths[0] == paths[1]

    def test_different_seeds_diverge(self):
        paths = []
        for seed in (1, 2, 3, 4, 5):
            env = SocialNavEnv(scenario="frontal_approach",
                               randomize_variant=False,
                               ped_hesitation=0.5, seed=seed)
            paths.append(tuple(_run_episode(env)))
        assert len(set(paths)) > 1

    def test_zero_hesitation_matches_legacy_behavior(self):
        legacy = SocialNavEnv(scenario="frontal_approach",
                              randomize_variant=False, seed=0)
        explicit = SocialNavEnv(scenario="frontal_approach",
                                randomize_variant=False,
                                ped_hesitation=0.0, seed=0)
        assert _run_episode(legacy) == _run_episode(explicit)


class TestConfigPlumbing:
    def test_env_config_carries_hesitation(self):
        cfg = SocialAgentConfig(ped_hesitation=0.3)
        assert env_config_for(cfg)["ped_hesitation"] == 0.3

    def test_experiment_name_unchanged_when_off(self):
        assert experiment_name(SocialAgentConfig()) == (
            "sac_scripted_proposer_learned_validator__all__binary")

    def test_experiment_name_suffixed_when_on(self):
        name = experiment_name(SocialAgentConfig(ped_hesitation=0.3))
        assert name.endswith("__hes0.3")
