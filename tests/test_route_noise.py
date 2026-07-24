"""Tests for randomized pedestrian routing (ped_route_noise).
"""

import numpy as np
import pytest

from idg_social_nav.config import SocialAgentConfig, env_config_for, experiment_name
from idg_social_nav.core import ProposerAction
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.grid import bfs_distances
from idg_social_nav.precompute_advice import _route_envelope
from idg_social_nav.scenarios import (
    ROUTE_DETOUR_SLACK,
    SCENARIO_NAMES,
    enumerate_variants,
    make_scenario,
)


def _run_episode(env: SocialNavEnv, scenario: str, variant: dict):
    """Always-forward/always-obey episode; returns per-turn (pos, facing)."""
    obs, _ = env.reset(options={"scenario": scenario, "variant": variant})
    trace = []
    while True:
        actions = ({"proposer": int(ProposerAction.forward)} if "proposer" in obs
                   else {"validator": 0})
        obs, _, terminated, truncated, _ = env.step(actions)
        trace.append((tuple(env.ped_states[0].pos), env.ped_states[0].facing))
        if terminated["__all__"] or truncated["__all__"]:
            return trace


class TestRouteStep:
    def test_rejects_invalid_probability(self):
        with pytest.raises(ValueError):
            SocialNavEnv(scenario="frontal_approach", ped_route_noise=-0.1)

    def test_zero_noise_matches_legacy_behavior(self):
        variant = {"ped_start_col": 8, "ped_delay": 0}
        legacy = SocialNavEnv(scenario="frontal_approach",
                              randomize_variant=False, seed=0)
        explicit = SocialNavEnv(scenario="frontal_approach",
                                randomize_variant=False,
                                ped_route_noise=0.0, seed=0)
        assert (_run_episode(legacy, "frontal_approach", variant)
                == _run_episode(explicit, "frontal_approach", variant))

    def test_stays_in_detour_ellipse_and_reaches_destination(self):
        variant = {"ped_start_col": 8, "ped_delay": 0}
        cfg = make_scenario("frontal_approach", variant)
        ped_cfg = cfg.pedestrians[0]
        dist_dest = bfs_distances(cfg.walls, ped_cfg.rail[-1])
        dist_start = bfs_distances(cfg.walls, ped_cfg.rail[0])
        budget = dist_dest[ped_cfg.rail[0]] + ROUTE_DETOUR_SLACK
        env = SocialNavEnv(scenario="frontal_approach",
                           randomize_variant=False,
                           ped_route_noise=0.8, seed=3)
        for _ in range(10):
            trace = _run_episode(env, "frontal_approach", dict(variant))
            for pos, _facing in trace:
                assert cfg.walls[pos] == 0
                assert dist_start[pos] + dist_dest[pos] <= budget

    def test_unobstructed_ped_always_arrives_within_budget(self):
        """Router-level guarantee: with no agent in the way, the pedestrian
        reaches its destination in at most dist(start) + slack moves."""
        variant = {"ped_start_col": 8, "ped_delay": 0}
        cfg = make_scenario("frontal_approach", variant)
        ped_cfg = cfg.pedestrians[0]
        dist = bfs_distances(cfg.walls, ped_cfg.rail[-1])
        budget0 = int(dist[ped_cfg.rail[0]]) + ROUTE_DETOUR_SLACK
        from idg_social_nav.scenarios import PedestrianState
        for seed in range(20):
            ped = PedestrianState.from_config(ped_cfg)
            ped.dist_map = dist
            ped.route_budget = budget0
            rng = np.random.default_rng(seed)
            for _ in range(budget0):
                ped.step((5, 9), set(), rng=rng, route_noise=0.9)
            assert tuple(ped.pos) == ped_cfg.rail[-1]

    def test_routes_vary_across_seeds(self):
        variant = {"ped_start_col": 8, "ped_delay": 0}
        traces = set()
        for seed in range(8):
            env = SocialNavEnv(scenario="frontal_approach",
                               randomize_variant=False,
                               ped_route_noise=0.8, seed=seed)
            trace = _run_episode(env, "frontal_approach", variant)
            traces.add(tuple(p for p, _ in trace))
        assert len(traces) > 1

    def test_routes_leave_the_rail(self):
        """With high noise the ped should visit off-rail cells sometimes."""
        variant = {"ped_start_col": 8, "ped_delay": 0}
        cfg = make_scenario("frontal_approach", variant)
        rail = set(cfg.pedestrians[0].rail)
        off_rail = 0
        for seed in range(10):
            env = SocialNavEnv(scenario="frontal_approach",
                               randomize_variant=False,
                               ped_route_noise=0.8, seed=seed)
            trace = _run_episode(env, "frontal_approach", variant)
            off_rail += sum(pos not in rail for pos, _ in trace)
        assert off_rail > 0

    def test_same_seed_reproduces(self):
        variant = {"ped_start_col": 8, "ped_delay": 0}
        runs = []
        for _ in range(2):
            env = SocialNavEnv(scenario="frontal_approach",
                               randomize_variant=False,
                               ped_route_noise=0.6, seed=42)
            runs.append(_run_episode(env, "frontal_approach", variant))
        assert runs[0] == runs[1]


class TestEnvelopeCompleteness:
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_every_visited_state_is_enumerated(self, scenario):
        """The precompute envelope must cover every (cell, facing) a routing
        pedestrian can reach, otherwise advice caches would have misses."""
        for variant in enumerate_variants(scenario):
            cfg = make_scenario(scenario, variant)
            envelope = set(_route_envelope(cfg.walls, cfg.pedestrians[0]))
            env = SocialNavEnv(scenario=scenario,
                               randomize_variant=False,
                               ped_route_noise=0.9,
                               ped_hesitation=0.2, seed=11)
            for _ in range(5):
                trace = _run_episode(env, scenario, dict(variant))
                for state in trace:
                    assert state in envelope, (
                        f"{scenario} {variant}: visited {state} "
                        "not in the precompute envelope")

    def test_envelope_is_superset_of_rail_states(self):
        for scenario in SCENARIO_NAMES:
            variant = enumerate_variants(scenario)[0]
            cfg = make_scenario(scenario, variant)
            ped = cfg.pedestrians[0]
            envelope_cells = {cell for cell, _ in
                              _route_envelope(cfg.walls, ped)}
            assert set(ped.rail) <= envelope_cells


class TestConfigPlumbing:
    def test_env_config_carries_route_noise(self):
        cfg = SocialAgentConfig(ped_route_noise=0.4)
        assert env_config_for(cfg)["ped_route_noise"] == 0.4

    def test_experiment_name_suffix(self):
        assert experiment_name(
            SocialAgentConfig(ped_route_noise=0.4)).endswith("__rn0.4")

    def test_combined_suffix_order(self):
        name = experiment_name(
            SocialAgentConfig(ped_hesitation=0.2, ped_route_noise=0.4))
        assert name.endswith("__hes0.2__rn0.4")

    def test_name_unchanged_when_off(self):
        assert "__rn" not in experiment_name(SocialAgentConfig())
