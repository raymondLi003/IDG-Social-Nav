"""Tests for eval_common.run_pairing."""

from idg_social_nav.eval_common import (
    always_obey_factory,
    fixed_blend_factory,
    format_table,
    run_pairing,
    scripted_proposer_factory,
)
from idg_social_nav.scenarios import enumerate_variants

_RESULT_KEYS = (
    "name", "scenario", "n_episodes", "success_pct", "mean_steps",
    "collision_episodes_pct", "mean_collisions", "mean_intrusion_sum",
    "high_intrusion_steps_mean", "validator_mean_reward", "overrides_per_ep",
    "good_disobey_pct", "bad_disobey_pct", "override_precision",
    "override_recall", "frozen_pct",
)


def _one_variant() -> list[dict]:
    return enumerate_variants("frontal_approach")[:1]


class TestRunPairing:
    def test_always_obey_smoke(self):
        result = run_pairing(
            "always_obey",
            scripted_proposer_factory,
            always_obey_factory,
            "frontal_approach",
            variants=_one_variant(),
            reps=1,
        )
        for key in _RESULT_KEYS:
            assert key in result, f"missing result key: {key}"
        assert result["name"] == "always_obey"
        assert result["scenario"] == "frontal_approach"
        assert result["n_episodes"] == 1
        assert result["overrides_per_ep"] == 0.0
        assert result["good_disobey_pct"] == 0.0
        assert result["bad_disobey_pct"] == 0.0
        assert result["n_validator_decisions"] > 0

    def test_episode_count_scales_with_variants_and_reps(self):
        result = run_pairing(
            "always_obey",
            scripted_proposer_factory,
            always_obey_factory,
            "frontal_approach",
            variants=enumerate_variants("frontal_approach")[:2],
            reps=2,
        )
        assert result["n_episodes"] == 4

    def test_fixed_blend_p1_overrides_more_than_always_obey(self):
        obey = run_pairing(
            "always_obey",
            scripted_proposer_factory,
            always_obey_factory,
            "frontal_approach",
            variants=_one_variant(),
            reps=1,
        )
        blend = run_pairing(
            "fixed_blend_p1.0",
            scripted_proposer_factory,
            fixed_blend_factory(1.0, seed=0),
            "frontal_approach",
            variants=_one_variant(),
            reps=1,
        )
        assert obey["overrides_per_ep"] == 0.0
        assert blend["overrides_per_ep"] > 0.0

    def test_format_table_handles_none_ratios(self):
        result = run_pairing(
            "always_obey",
            scripted_proposer_factory,
            always_obey_factory,
            "frontal_approach",
            variants=_one_variant(),
            reps=1,
        )
        table = format_table([result])
        assert "always_obey" in table
        assert "frontal_approach" in table
