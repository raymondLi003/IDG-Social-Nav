"""Headline validator comparison across the social-nav scenarios.

run for every scenario and every validator, then summarize the results in a CSV
four scenarios: narrow_doorway, frontal_gesture, lateral_gesture, and crosswalk
validators: always_obey, fixed_blend, oracle, ppo, llm, llm_explain
fixed blend: p=0.0, 0.1, ..., 1.0 

Usage:
    python -m idg_social_nav.run_eval                     # baselines, all scenarios
    python -m idg_social_nav.run_eval --validators always_obey,oracle,ppo
    python -m idg_social_nav.run_eval --scenarios narrow_doorway --validators llm --video
    python -m idg_social_nav.run_eval --advisor noisy:0.3 --reps 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idg_social_nav.config import (
    EVAL_REPS_STOCHASTIC,
    ProposerPolicies,
    SocialAgentConfig,
    ValidatorPolicies,
    experiment_name,
)
from idg_social_nav.core import Advisor
from idg_social_nav.eval_common import (
    ModuleFactory,
    always_obey_factory,
    build_inference_module,
    fixed_blend_factory,
    load_learned_validator,
    oracle_validator_factory,
    print_summary,
    run_pairing,
    scripted_proposer_factory,
)
from idg_social_nav.scenarios import SCENARIO_NAMES

VALIDATOR_CHOICES = ("always_obey", "fixed_blend", "oracle", "ppo", "llm", "llm_explain")
DEFAULT_VALIDATORS = "always_obey,fixed_blend,oracle"
DEFAULT_BLEND_PS = ",".join(f"{p / 10:.1f}" for p in range(0, 11))
DEFAULT_LLM_MODELS = "haiku3=us.anthropic.claude-3-haiku-20240307-v1:0"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_scenarios(value: str) -> list[str]:
    if value == "all":
        return list(SCENARIO_NAMES)
    names = _parse_csv(value)
    for name in names:
        if name not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario: {name!r}. Known: {SCENARIO_NAMES}")
    return names


def _resolve_cache_path(path_spec: str, scenario_name: str) -> Path:
    """Resolve a cached:<path> spec for one scenario. 
    Supports a file name, a `{scenario}` placeholder, or a directory holding one
    `<scenario>_*.json` cache per scenario. The directory form is convenient for precomputing all scenarios in one go"""
    if "{scenario}" in path_spec:
        return Path(path_spec.format(scenario=scenario_name))
    path = Path(path_spec)
    if path.is_dir():
        matches = sorted(path.glob(f"{scenario_name}_*.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {scenario_name}_*.json cache in "
                f"{path}, found {len(matches)}: {[m.name for m in matches]}")
        return matches[0]
    return path


def make_advisor_provider(spec: str, seed: int):
    """Returns (provider, stochastic). 
    The provider(scenario_name) builds a new advisor per pairing so no rng or cache state leaks across conditions."""
    if spec == "scripted":
        return (lambda scenario_name: None), False
    if spec.startswith("cached:"):
        from idg_social_nav.advisor_scripted import ScriptedSocialAdvisor
        from idg_social_nav.vlm_advisor import CachedAdvisor
        path_spec = spec[len("cached:"):]

        def provider(scenario_name: str) -> Advisor:
            path = _resolve_cache_path(path_spec, scenario_name)
            if not path.exists():
                raise FileNotFoundError(f"Advice cache not found: {path}")
            return CachedAdvisor(
                ScriptedSocialAdvisor(), cache_path=str(path), read_only=True)

        return provider, False
    if spec.startswith("noisy:"):
        from idg_social_nav.advisor_scripted import NoisyAdvisor, ScriptedSocialAdvisor
        epsilon = float(spec[len("noisy:"):])

        def provider(scenario_name: str) -> Advisor:
            return NoisyAdvisor(ScriptedSocialAdvisor(), epsilon, seed=seed)

        return provider, epsilon > 0.0
    raise ValueError(f"Unknown advisor spec: {spec!r}")


def _make_llm_validator_class(base: type, model_name: str) -> type:
    """Build a subclass of `base` pinned to a specific LLM model."""
    return type(f"{base.__name__}_{model_name}", (base,), {"MODEL_NAME": model_name})


def _llm_factory(validator_class: type) -> ModuleFactory:
    def factory(env) -> object:
        return build_inference_module(env, "validator", validator_class)

    return factory


def parse_llm_models(value: str) -> list[tuple[str, str]]:
    """Parse display=model,..." to [(display, model), ...]."""
    models = []
    for item in _parse_csv(value):
        display, _, model = item.partition("=")
        if not model:
            raise ValueError(f"LLM model spec must be display=model, got {item!r}")
        models.append((display.strip(), model.strip()))
    return models


def build_conditions(args) -> list[tuple[str, ModuleFactory, int]]:
    """(name, validator_factory, reps) for every requested condition.

    Deterministic validators get 1 rep per condition.
    only stochastic ones (fixed blend with 0 < p < 1) use --reps.
    """
    conditions: list[tuple[str, ModuleFactory, int]] = []
    for validator in _parse_csv(args.validators):
        if validator == "always_obey":
            conditions.append(("always_obey", always_obey_factory, 1))
        elif validator == "oracle":
            conditions.append(("oracle", oracle_validator_factory, 1))
        elif validator == "fixed_blend":
            for p_str in _parse_csv(args.blend_ps):
                p = float(p_str)
                reps = 1 if p in (0.0, 1.0) else args.reps
                conditions.append((
                    f"fixed_blend_p{p:g}",
                    fixed_blend_factory(p, seed=args.seed),
                    reps,
                ))
        elif validator == "ppo":
            exp_name = args.checkpoint or experiment_name(SocialAgentConfig(
                proposer_policy=ProposerPolicies.SCRIPTED,
                validator_policy=ValidatorPolicies.LEARNED,
                algorithm_name="ppo",
                scenario="all",
                reward_variant=args.reward_variant,
            ))
            conditions.append(("ppo", load_learned_validator(exp_name), 1))
        elif validator in ("llm", "llm_explain"):
            from idg_social_nav.llm_validator import LLMValidatorSocial, LLMValidatorSocialExplain
            base = LLMValidatorSocialExplain if validator == "llm_explain" else LLMValidatorSocial
            for display, model in parse_llm_models(args.llm_models):
                validator_class = _make_llm_validator_class(base, model)
                conditions.append((f"{validator}_{display}", _llm_factory(validator_class), 1))
        else:
            raise ValueError(
                f"Unknown validator: {validator!r}. Known: {VALIDATOR_CHOICES}")
    return conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios", type=str, default="all",
                        help=f"'all' or comma-separated names from {SCENARIO_NAMES}")
    parser.add_argument("--validators", type=str, default=DEFAULT_VALIDATORS,
                        help=f"comma-separated subset of {VALIDATOR_CHOICES}")
    parser.add_argument("--blend-ps", type=str, default=DEFAULT_BLEND_PS,
                        help="override probabilities for the fixed-blend sweep")
    parser.add_argument("--reps", type=int, default=EVAL_REPS_STOCHASTIC,
                        help="reps per variant for stochastic validators "
                             "(deterministic validators always run 1)")
    parser.add_argument("--llm-models", type=str, default=DEFAULT_LLM_MODELS,
                        help="comma-separated display=model pairs for the LLM validators")
    parser.add_argument("--advisor", type=str, default="scripted",
                        help="scripted | cached:<file, dir, or path with {scenario}> "
                             "| noisy:<eps>")
    parser.add_argument("--reward-variant", choices=["binary", "graded"], default="binary")
    parser.add_argument("--override-semantics", choices=["adopt", "nullify"], default="adopt")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="tune experiment name for the ppo validator checkpoint "
                             "(default: the canonical all-scenario experiment)")
    parser.add_argument("--video", action="store_true",
                        help="record the first episode of each pairing to videos/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenario_names = resolve_scenarios(args.scenarios)
    conditions = build_conditions(args)
    advisor_provider, advisor_stochastic = make_advisor_provider(args.advisor, args.seed)
    print(f"Scenarios: {scenario_names}")
    print(f"Conditions: {[name for name, _, _ in conditions]}")

    results = []
    for scenario_name in scenario_names:
        for name, validator_factory, reps in conditions:
            # a stochastic advisor makes every condition stochastic
            eff_reps = max(reps, args.reps) if advisor_stochastic else reps
            try:
                results.append(run_pairing(
                    name=name,
                    proposer_factory=scripted_proposer_factory,
                    validator_factory=validator_factory,
                    scenario_name=scenario_name,
                    reps=eff_reps,
                    advisor=advisor_provider(scenario_name),
                    seed=args.seed,
                    reward_variant=args.reward_variant,
                    override_semantics=args.override_semantics,
                    record_video=args.video,
                ))
            except (FileNotFoundError, KeyError) as e:
                print(f"[Skipping {name}@{scenario_name}] {e}")

    if results:
        tag = "-".join(_parse_csv(args.validators))
        print_summary(results, tag)
    else:
        print("No evaluations were run.")


if __name__ == "__main__":
    main()
