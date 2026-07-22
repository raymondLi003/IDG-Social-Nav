"""Tests for metrics utilities (metrics.py)."""

import pytest

from idg_social_nav.core import Advice, EnvironmentAction
from idg_social_nav.metrics import aggregate, confusion_counts, episode_metrics, oracle_agreement


def _info(**over) -> dict:
    base = {
        "intrusion": 0.0,
        "executed_action": int(EnvironmentAction.MOVE_FORWARD),
        "collision_attempt": False,
        "overridden": False,
        "good_override": False,
        "bad_override": False,
        "failed_override": False,
        "missed_hazard": False,
        "advice": int(Advice.NONE),
    }
    base.update(over)
    return base


def _crafted_episode() -> list[dict]:
    noop = int(EnvironmentAction.NO_OP)
    return [
        _info(intrusion=0.6, advice=int(Advice.TURN_RIGHT),
              overridden=True, good_override=True),
        _info(executed_action=noop, advice=int(Advice.WAIT)),
        _info(executed_action=noop),
        _info(executed_action=noop),
        _info(executed_action=noop),
        _info(executed_action=noop),
        _info(intrusion=0.5, collision_attempt=True, missed_hazard=True),
        _info(intrusion=0.1, executed_action=int(EnvironmentAction.TURN_LEFT),
              advice=int(Advice.TURN_LEFT), overridden=True, bad_override=True),
    ]


class TestEpisodeMetrics:
    def test_crafted_episode(self):
        infos = _crafted_episode()
        m = episode_metrics(infos, reached_goal=True, steps=len(infos), max_steps=40)
        assert m["success"] is True
        assert m["steps"] == 8
        assert m["max_steps"] == 40
        assert m["collisions"] == 1
        assert m["intrusion_sum"] == pytest.approx(1.2)
        assert m["intrusion_max"] == pytest.approx(0.6)
        assert m["high_intrusion_steps"] == 2  # 0.6 and 0.5 (tau inclusive)
        assert m["overrides"] == 2
        assert m["good_overrides"] == 1
        assert m["bad_overrides"] == 1
        assert m["failed_overrides"] == 0
        assert m["missed_hazards"] == 1
        assert m["freeze_steps"] == 5
        assert m["frozen"] is True  # five consecutive NO_OPs
        assert m["advice_steps"] == 3

    def test_frozen_needs_consecutive_noops(self):
        noop = int(EnvironmentAction.NO_OP)
        infos = ([_info(executed_action=noop)] * 4
                 + [_info()]
                 + [_info(executed_action=noop)] * 4)
        m = episode_metrics(infos, reached_goal=False, steps=len(infos), max_steps=40)
        assert m["freeze_steps"] == 8
        assert m["frozen"] is False

    def test_empty_episode(self):
        m = episode_metrics([], reached_goal=False, steps=0, max_steps=40)
        assert m["success"] is False
        assert m["intrusion_sum"] == 0.0
        assert m["intrusion_max"] == 0.0
        assert m["frozen"] is False


class TestOracleAgreement:
    def test_mixed_decisions(self):
        # (validator, oracle): one TP, one FP, one FN, one TN
        stats = oracle_agreement([(1, 1), (1, 0), (0, 1), (0, 0)])
        assert stats["override_precision"] == pytest.approx(0.5)
        assert stats["override_recall"] == pytest.approx(0.5)
        assert stats["agreement"] == pytest.approx(0.5)

    def test_perfect_validator(self):
        stats = oracle_agreement([(1, 1), (0, 0), (1, 1)])
        assert stats["override_precision"] == pytest.approx(1.0)
        assert stats["override_recall"] == pytest.approx(1.0)
        assert stats["agreement"] == pytest.approx(1.0)

    def test_zero_division_reports_none(self):
        # never overrides: precision undefined, recall 0
        stats = oracle_agreement([(0, 0), (0, 1)])
        assert stats["override_precision"] is None
        assert stats["override_recall"] == pytest.approx(0.0)
        assert stats["agreement"] == pytest.approx(0.5)

    def test_empty_decisions_all_none(self):
        stats = oracle_agreement([])
        assert stats["override_precision"] is None
        assert stats["override_recall"] is None
        assert stats["agreement"] is None

    def test_confusion_counts(self):
        c = confusion_counts([(1, 1), (1, 0), (0, 1), (0, 0)])
        assert c == {
            "override_tp": 1,
            "override_fp": 1,
            "override_fn": 1,
            "oracle_agreements": 2,
            "decisions": 4,
        }


class TestAggregate:
    def test_micro_averaged_confusion(self):
        ep1 = episode_metrics(
            _crafted_episode(), reached_goal=True, steps=8, max_steps=40)
        ep1.update(confusion_counts([(1, 1), (0, 0)]))
        ep2 = episode_metrics(
            [_info(), _info()], reached_goal=False, steps=2, max_steps=40)
        ep2.update(confusion_counts([(1, 0), (0, 1)]))

        agg = aggregate([ep1, ep2])
        assert agg["n_episodes"] == 2
        assert agg["success_pct"] == pytest.approx(50.0)
        assert agg["mean_steps"] == pytest.approx(5.0)
        assert agg["collision_episodes_pct"] == pytest.approx(50.0)
        assert agg["mean_intrusion_sum"] == pytest.approx(0.6)
        assert agg["overrides_per_ep"] == pytest.approx(1.0)
        assert agg["frozen_pct"] == pytest.approx(50.0)
        # summed counts: tp=1, fp=1, fn=1, agreements=2 over 4 decisions
        assert agg["override_precision"] == pytest.approx(0.5)
        assert agg["override_recall"] == pytest.approx(0.5)
        assert agg["oracle_agreement"] == pytest.approx(0.5)

    def test_no_decisions_reports_none(self):
        ep = episode_metrics([_info()], reached_goal=True, steps=1, max_steps=40)
        agg = aggregate([ep])
        assert agg["override_precision"] is None
        assert agg["override_recall"] is None
        assert agg["oracle_agreement"] is None

    def test_empty_episode_list(self):
        assert aggregate([]) == {"n_episodes": 0}
