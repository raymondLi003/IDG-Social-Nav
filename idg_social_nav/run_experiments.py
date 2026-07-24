"""Train the learned validator (scripted proposer x learned validator).

Usage:
    python -m idg_social_nav.run_experiments     # SAC on all scenarios
    python -m idg_social_nav.run_experiments --algo ppo
    python -m idg_social_nav.run_experiments --scenario narrow_doorway --reward-variant graded
"""

import argparse

import ray
from ray import tune

from idg_social_nav.config import (
    TRAINING_ITERATIONS,
    ProposerPolicies,
    SocialAgentConfig,
    ValidatorPolicies,
    create_rllib_config,
    experiment_name,
    register_envs,
)
from idg_social_nav.metrics import ActionLoggerCallback, CustomTBXLoggerCallback
from idg_social_nav.paths import TUNE_DIR
from idg_social_nav.scenarios import SCENARIO_NAMES

VALIDATOR_METRIC = "env_runners/module_episode_returns_mean/learned_validator"


def _metric_value(metrics: dict) -> float:
    """Read the validator return from a (possibly nested) result dict."""
    value = metrics.get(VALIDATOR_METRIC)
    if value is None:
        value = (
            metrics.get("env_runners", {})
            .get("module_episode_returns_mean", {})
            .get(str(ValidatorPolicies.LEARNED))
        )
    return float(value) if value is not None else float("-inf")


def _best_checkpoint(result) -> tune.Checkpoint | None:
    """Simple max over the trial's saved checkpoints"""
    ckpts = result.best_checkpoints or []
    scored = [(ckpt, _metric_value(m or {})) for ckpt, m in ckpts]
    scored = [(c, s) for c, s in scored if c is not None]
    if not scored:
        return result.checkpoint
    return max(scored, key=lambda x: x[1])[0]


def run_experiments(cfg: SocialAgentConfig, iters: int, num_env_runners: int | None) -> None:
    config = create_rllib_config(cfg)
    if num_env_runners is not None:
        config.env_runners(num_env_runners=num_env_runners)
    config.callbacks([ActionLoggerCallback])

    exp_name = experiment_name(cfg)
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": iters},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=25,
            ),
            storage_path=TUNE_DIR,
            name=exp_name,
            callbacks=[CustomTBXLoggerCallback()],
        ),
    )

    tuner_results = tuner.fit()
    best_result = tuner_results.get_best_result(metric=VALIDATOR_METRIC, mode="max", scope="all")
    best_ckpt = _best_checkpoint(best_result)
    if best_ckpt is None:
        raise RuntimeError(f"No checkpoint produced for {exp_name}.")
    best_ckpt.to_directory(str(TUNE_DIR / exp_name / "best_checkpoint"))
    print(f"Saved best checkpoint to {TUNE_DIR / exp_name / 'best_checkpoint'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--scenario", choices=["all", *SCENARIO_NAMES], default="all")
    parser.add_argument("--reward-variant", choices=["binary", "graded"], default="binary")
    parser.add_argument("--iters", type=int, default=TRAINING_ITERATIONS,
                        help=f"training iterations (default: {TRAINING_ITERATIONS})")
    parser.add_argument("--num-env-runners", type=int, default=None,
                        help="override the number of env runner workers")
    parser.add_argument("--ped-hesitation", type=float, default=0.0,
                        help="probability a pedestrian pauses instead of stepping "
                             "(0 = deterministic rails)")
    parser.add_argument("--ped-route-noise", type=float, default=0.0,
                        help="probability a moving pedestrian widens its step "
                             "choice to sideways cells (0 = fixed rails; > 0 = "
                             "randomized routes to the same destination)")
    args = parser.parse_args()

    cfg = SocialAgentConfig(
        proposer_policy=ProposerPolicies.SCRIPTED,
        validator_policy=ValidatorPolicies.LEARNED,
        algorithm_name=args.algo,
        scenario=args.scenario,
        reward_variant=args.reward_variant,
        ped_hesitation=args.ped_hesitation,
        ped_route_noise=args.ped_route_noise,
    )

    ray.init(ignore_reinit_error=True)
    register_envs(cfg)
    run_experiments(cfg, args.iters, args.num_env_runners)
    ray.shutdown()


if __name__ == "__main__":
    main()
