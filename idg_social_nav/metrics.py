"""Episode metrics for the social-nav benchmark and RLlib logging callbacks.

The episode metrics are computed from the per-step info dicts the env outputs
after every validator decision (see SocialNavEnv._step_validator), 
so the same code scores scripted, learned, and LLM validators identically.
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import TBXLoggerCallback

from idg_social_nav.core import Advice, EnvironmentAction

HIGH_DISCOMFORT_TAU = 0.5
FREEZE_RUN_LENGTH = 5


def _safe_div(num: float, denom: float) -> float | None:
    return num / denom if denom else None


def episode_metrics(
        step_infos: list[dict],
        reached_goal: bool,
        steps: int,
        max_steps: int,
        tau: float = HIGH_DISCOMFORT_TAU,
) -> dict:
    """Scalar summary of one episode from the validator-step info dicts."""
    intrusions = [float(info["intrusion"]) for info in step_infos]
    executed = [int(info["executed_action"]) for info in step_infos]

    longest_noop_run = 0
    run = 0
    for action in executed:
        run = run + 1 if action == EnvironmentAction.NO_OP else 0
        longest_noop_run = max(longest_noop_run, run)

    return {
        "success": bool(reached_goal),
        "steps": int(steps),
        "max_steps": int(max_steps),
        "collisions": sum(bool(info["collision_attempt"]) for info in step_infos),
        "intrusion_sum": float(sum(intrusions)),
        "intrusion_max": float(max(intrusions, default=0.0)),
        "high_intrusion_steps": sum(i >= tau for i in intrusions),
        "overrides": sum(bool(info["overridden"]) for info in step_infos),
        "good_overrides": sum(bool(info["good_override"]) for info in step_infos),
        "bad_overrides": sum(bool(info["bad_override"]) for info in step_infos),
        "failed_overrides": sum(bool(info["failed_override"]) for info in step_infos),
        "missed_hazards": sum(bool(info["missed_hazard"]) for info in step_infos),
        "freeze_steps": sum(a == EnvironmentAction.NO_OP for a in executed),
        "frozen": longest_noop_run >= FREEZE_RUN_LENGTH,
        "advice_steps": sum(int(info["advice"]) != Advice.NONE for info in step_infos),
    }


def confusion_counts(decisions: list[tuple[int, int]]) -> dict:
    """Override confusion counts from (validator, oracle) decision pairs."""
    tp = sum(1 for v, o in decisions if v == 1 and o == 1)
    fp = sum(1 for v, o in decisions if v == 1 and o == 0)
    fn = sum(1 for v, o in decisions if v == 0 and o == 1)
    agree = sum(1 for v, o in decisions if v == o)
    return {
        "override_tp": tp,
        "override_fp": fp,
        "override_fn": fn,
        "oracle_agreements": agree,
        "decisions": len(decisions),
    }


def oracle_agreement(decisions: list[tuple[int, int]]) -> dict:
    """Precision and recall of the validator's overrides against the oracle's.

    Undefined ratios (zero denominator) are reported as None.
    """
    c = confusion_counts(decisions)
    return {
        "override_precision": _safe_div(c["override_tp"], c["override_tp"] + c["override_fp"]),
        "override_recall": _safe_div(c["override_tp"], c["override_tp"] + c["override_fn"]),
        "agreement": _safe_div(c["oracle_agreements"], c["decisions"]),
    }


def aggregate(episodes: list[dict]) -> dict:
    """Means/rates over per-episode metric dicts.

    Precision/recall/agreement are micro-averaged from the summed confusion
    counts (episodes carry the confusion_counts keys when available).
    """
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0}

    def mean(key: str) -> float:
        return float(sum(e[key] for e in episodes)) / n

    def total(key: str) -> int:
        return sum(e.get(key, 0) for e in episodes)

    tp, fp, fn = total("override_tp"), total("override_fp"), total("override_fn")
    n_decisions = total("decisions")

    return {
        "n_episodes": n,
        "success_pct": 100.0 * sum(bool(e["success"]) for e in episodes) / n,
        "mean_steps": mean("steps"),
        "collision_episodes_pct": 100.0 * sum(e["collisions"] > 0 for e in episodes) / n,
        "mean_collisions": mean("collisions"),
        "mean_intrusion_sum": mean("intrusion_sum"),
        "mean_intrusion_max": mean("intrusion_max"),
        "high_intrusion_steps_mean": mean("high_intrusion_steps"),
        "overrides_per_ep": mean("overrides"),
        "good_overrides_per_ep": mean("good_overrides"),
        "bad_overrides_per_ep": mean("bad_overrides"),
        "failed_overrides_per_ep": mean("failed_overrides"),
        "missed_hazards_per_ep": mean("missed_hazards"),
        "freeze_steps_mean": mean("freeze_steps"),
        "frozen_pct": 100.0 * sum(bool(e["frozen"]) for e in episodes) / n,
        "advice_steps_mean": mean("advice_steps"),
        "override_precision": _safe_div(tp, tp + fp),
        "override_recall": _safe_div(tp, tp + fn),
        "oracle_agreement": _safe_div(total("oracle_agreements"), n_decisions),
    }


class CustomTBXLoggerCallback(TBXLoggerCallback):
    """logging minus RLlib's noisiest result subtrees"""

    def log_trial_result(self, iteration: int, trial, result: dict):
        result.pop("timers", None)
        if "env_runners" in result:
            for key in list(result["env_runners"].keys()):
                if "timer" in key:
                    del result["env_runners"][key]
            result["env_runners"].pop("module_to_env_connector", None)
            result["env_runners"].pop("env_to_module_connector", None)
            result["env_runners"].pop("time_between_sampling", None)
        result.pop("replay_buffer", None)
        if "learners" in result:
            for agent_id in list(result["learners"].keys()):
                if "learner_connector" in result["learners"][agent_id]:
                    del result["learners"][agent_id]["learner_connector"]
        result.pop("perf", None)
        super().log_trial_result(iteration, trial, result)


class ActionLoggerCallback(DefaultCallbacks):
    """Log every agent action as a tensorboard item series."""

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        for agent_id, actions in episode.get_actions().items():
            for action in actions:
                metrics_logger.log_value(
                    f"action/{agent_id}", int(action), reduce="item_series")
