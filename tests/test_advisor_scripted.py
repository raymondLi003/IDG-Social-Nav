"""Tests for the deterministic rule-based advisor (advisor_scripted.py)."""

import pytest

from idg_social_nav.advisor_scripted import NoisyAdvisor, ScriptedSocialAdvisor
from idg_social_nav.core import (
    DOWN,
    LEFT,
    RIGHT,
    Advice,
    Advisor,
    AdvisorContext,
    Gesture,
    PedestrianSnapshot,
)
from idg_social_nav.scenarios import make_scenario


def _context(walls, agent_pos, agent_dir, goal_pos, pedestrians,
             scenario_name="frontal_approach"):
    return AdvisorContext(
        scenario_name=scenario_name,
        walls=walls,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        goal_pos=goal_pos,
        pedestrians=pedestrians,
        step=0,
    )


def _frontal_context(gesture=Gesture.NONE):
    walls = make_scenario(
        "frontal_approach", {"ped_start_col": 7, "ped_delay": 0}).walls
    ped = PedestrianSnapshot(
        pos=(3, 5), facing=LEFT, gesture=gesture, visible=True)
    return _context(walls, (3, 3), RIGHT, (3, 9), [ped])


class TestScriptedSocialAdvisor:
    def test_frontal_approach_sidesteps_right(self):
        advice = ScriptedSocialAdvisor().advise(_frontal_context())
        assert advice == Advice.TURN_RIGHT

    def test_intersection_conflict_waits(self):
        walls = make_scenario(
            "intersection", {"ped_direction": "down", "ped_delay": 0}).walls
        ped = PedestrianSnapshot(
            pos=(3, 5), facing=DOWN, gesture=Gesture.NONE, visible=True)
        context = _context(walls, (4, 4), RIGHT, (4, 9), [ped],
                           scenario_name="intersection")
        assert ScriptedSocialAdvisor().advise(context) == Advice.WAIT

    def test_go_gesture_proceeds(self):
        advice = ScriptedSocialAdvisor().advise(
            _frontal_context(gesture=Gesture.GO))
        assert advice == Advice.FORWARD
        assert advice != Advice.WAIT

    def test_stop_gesture_clears_lane(self):
        advice = ScriptedSocialAdvisor().advise(
            _frontal_context(gesture=Gesture.STOP))
        assert advice not in (Advice.FORWARD, Advice.WAIT)
        assert advice == Advice.TURN_RIGHT

    def test_no_visible_ped_returns_none(self):
        walls = make_scenario(
            "frontal_approach", {"ped_start_col": 7, "ped_delay": 0}).walls
        hidden = PedestrianSnapshot(
            pos=(3, 7), facing=LEFT, gesture=Gesture.NONE, visible=False)
        advisor = ScriptedSocialAdvisor()
        assert advisor.advise(_context(walls, (3, 1), RIGHT, (3, 9), [hidden])) \
            == Advice.NONE
        assert advisor.advise(_context(walls, (3, 1), RIGHT, (3, 9), [])) \
            == Advice.NONE

    def test_deterministic(self):
        advisor = ScriptedSocialAdvisor()
        context = _frontal_context()
        assert all(advisor.advise(context) == Advice.TURN_RIGHT
                   for _ in range(5))


class _FixedAdvisor(Advisor):
    def __init__(self, advice: Advice):
        self.advice = advice

    def advise(self, context) -> Advice:
        return self.advice


class TestNoisyAdvisor:
    def test_epsilon_zero_is_identity(self):
        context = _frontal_context()
        noisy = NoisyAdvisor(_FixedAdvisor(Advice.FORWARD), epsilon=0.0, seed=0)
        assert all(noisy.advise(context) == Advice.FORWARD for _ in range(20))

    def test_epsilon_one_always_flips(self):
        context = _frontal_context()
        noisy = NoisyAdvisor(_FixedAdvisor(Advice.FORWARD), epsilon=1.0, seed=0)
        for _ in range(50):
            advice = noisy.advise(context)
            assert advice != Advice.FORWARD
            assert advice != Advice.NONE

    def test_none_advice_never_flipped(self):
        context = _frontal_context()
        noisy = NoisyAdvisor(_FixedAdvisor(Advice.NONE), epsilon=1.0, seed=0)
        assert all(noisy.advise(context) == Advice.NONE for _ in range(20))

    def test_seeded_reproducibility(self):
        context = _frontal_context()
        first = NoisyAdvisor(_FixedAdvisor(Advice.FORWARD), epsilon=0.5, seed=42)
        second = NoisyAdvisor(_FixedAdvisor(Advice.FORWARD), epsilon=0.5, seed=42)
        seq1 = [int(first.advise(context)) for _ in range(40)]
        seq2 = [int(second.advise(context)) for _ in range(40)]
        assert seq1 == seq2
        assert any(a != Advice.FORWARD for a in seq1)   # noise actually fires
        assert any(a == Advice.FORWARD for a in seq1)   # but not always

    def test_invalid_epsilon_rejected(self):
        with pytest.raises(AssertionError):
            NoisyAdvisor(_FixedAdvisor(Advice.FORWARD), epsilon=1.5)
