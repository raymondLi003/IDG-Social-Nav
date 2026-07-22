"""Tests for scenario variants, rails, and scripted pedestrian dynamics."""

import numpy as np
import pytest

from idg_social_nav.core import LEFT, UP, Gesture
from idg_social_nav.scenarios import (
    SCENARIO_NAMES,
    YIELD_DISTANCE,
    PedestrianConfig,
    PedestrianState,
    enumerate_variants,
    make_scenario,
)

EXPECTED_VARIANT_COUNTS = {
    "frontal_approach": 6,
    "narrow_doorway": 4,
    "intersection": 8,
    "frontal_gesture": 8,
}

_FAR_AGENT = (5, 9)


class TestVariants:
    def test_scenario_names(self):
        assert SCENARIO_NAMES == (
            "frontal_approach", "narrow_doorway", "intersection", "frontal_gesture")

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_variant_counts(self, name):
        variants = enumerate_variants(name)
        assert len(variants) == EXPECTED_VARIANT_COUNTS[name]
        assert len({tuple(sorted(v.items())) for v in variants}) == len(variants)

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_rails_on_free_cells(self, name):
        for variant in enumerate_variants(name):
            cfg = make_scenario(name, variant)
            for ped in cfg.pedestrians:
                for r, c in ped.rail:
                    assert cfg.walls[r, c] == 0

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_agent_and_goal_on_free_cells(self, name):
        for variant in enumerate_variants(name):
            cfg = make_scenario(name, variant)
            assert cfg.walls[cfg.agent_start] == 0
            assert cfg.walls[cfg.goal_pos] == 0

    @pytest.mark.parametrize("name", SCENARIO_NAMES)
    def test_make_scenario_deterministic(self, name):
        variant = enumerate_variants(name)[-1]
        a = make_scenario(name, variant)
        b = make_scenario(name, variant)
        assert np.array_equal(a.walls, b.walls)
        assert a.agent_start == b.agent_start
        assert a.agent_dir == b.agent_dir
        assert a.goal_pos == b.goal_pos
        assert a.pedestrians == b.pedestrians
        assert a.max_steps == b.max_steps
        assert a.variant == variant

    def test_default_variant_is_first(self):
        cfg = make_scenario("frontal_approach")
        assert cfg.variant == enumerate_variants("frontal_approach")[0]

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            make_scenario("hallway")

    def test_variant_keys_validated(self):
        with pytest.raises(ValueError):
            make_scenario("narrow_doorway", {"ped_delay": 0, "extra": 1})

    def test_frontal_gesture_carries_gesture(self):
        cfg = make_scenario(
            "frontal_gesture",
            {"gesture": "STOP", "ped_start_col": 7, "ped_delay": 0})
        assert cfg.pedestrians[0].gesture == Gesture.STOP


def _walker(gesture=Gesture.NONE, delay=0):
    rail = ((3, 8), (3, 7), (3, 6), (3, 5))
    return PedestrianState.from_config(
        PedestrianConfig(rail=rail, gesture=gesture, start_delay=delay))


class TestPedestrianStepping:
    def test_initial_facing_from_rail(self):
        ped = _walker()
        assert ped.pos == (3, 8)
        assert ped.facing == LEFT

    def test_initial_facing_override_and_single_cell_rail(self):
        forced = PedestrianState.from_config(
            PedestrianConfig(rail=((3, 8), (3, 7)), initial_facing=UP))
        assert forced.facing == UP
        lone = PedestrianState.from_config(PedestrianConfig(rail=((3, 8),)))
        assert lone.facing == UP
        lone.step(_FAR_AGENT, set())
        assert lone.pos == (3, 8)

    def test_delay_holds_then_walks(self):
        ped = _walker(delay=2)
        ped.step(_FAR_AGENT, set())
        assert ped.pos == (3, 8)
        ped.step(_FAR_AGENT, set())
        assert ped.pos == (3, 8)
        ped.step(_FAR_AGENT, set())
        assert ped.pos == (3, 7)

    def test_stops_at_rail_end(self):
        ped = _walker()
        for _ in range(10):
            ped.step(_FAR_AGENT, set())
        assert ped.pos == (3, 5)
        assert ped.rail_idx == 3

    def test_agent_blocks_next_cell(self):
        ped = _walker()
        ped.step((3, 7), set())
        assert ped.pos == (3, 8)
        ped.step(_FAR_AGENT, set())
        assert ped.pos == (3, 7)

    def test_other_pedestrian_blocks_next_cell(self):
        ped = _walker()
        ped.step(_FAR_AGENT, {(3, 7)})
        assert ped.pos == (3, 8)

    def test_go_yields_to_near_agent_in_front(self):
        ped = _walker(gesture=Gesture.GO)
        agent = (3, 8 - YIELD_DISTANCE + 1)   # dist 2, in front
        ped.step(agent, set())
        assert ped.pos == (3, 8)

    def test_go_walks_when_agent_far(self):
        ped = _walker(gesture=Gesture.GO)
        agent = (3, 8 - YIELD_DISTANCE - 2)   # dist 5 > YIELD_DISTANCE
        ped.step(agent, set())
        assert ped.pos == (3, 7)

    def test_go_resumes_after_agent_passes(self):
        ped = _walker(gesture=Gesture.GO)
        ped.step((3, 6), set())
        assert ped.pos == (3, 8)              # yielding
        ped.step((3, 9), set())               # agent now behind -> resume
        assert ped.pos == (3, 7)

    def test_non_go_gestures_never_yield(self):
        for gesture in (Gesture.NONE, Gesture.STOP):
            ped = _walker(gesture=gesture)
            ped.step((3, 6), set())           # near and in front
            assert ped.pos == (3, 7)
