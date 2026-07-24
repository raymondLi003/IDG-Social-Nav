"""Shared evaluation engine for the IDG social-nav runners

Used by:
  - run_eval.py
  - tests/test_eval_common.py

Keeps rollout / metrics / table-printing code in one place so the runner
scripts only enumerate the conditions and call the shared helpers

"""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import tqdm
import tree
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule, RLModuleSpec

from idg_social_nav.config import ValidatorPolicies
from idg_social_nav.core import Advisor
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.metrics import aggregate, confusion_counts, episode_metrics
from idg_social_nav.paths import EVAL_DIR, TUNE_DIR, VIDEO_DIR
from idg_social_nav.scenarios import enumerate_variants

ModuleFactory = Callable[[SocialNavEnv], RLModule]


def build_inference_module(
    env: SocialNavEnv,
    agent: str,
    module_class: type[RLModule],
    model_config: dict | None = None,
) -> RLModule:
    spec = RLModuleSpec(
        module_class=module_class,
        observation_space=env.observation_spaces[agent],
        action_space=env.action_spaces[agent],
        model_config=model_config,
        inference_only=True,
    )
    return spec.build()


def scripted_proposer_factory(env: SocialNavEnv) -> RLModule:
    from idg_social_nav.rl_modules.shortest_path_proposer import ShortestPathProposerRLM
    return build_inference_module(env, "proposer", ShortestPathProposerRLM)


def _env_tau(env: SocialNavEnv) -> float:
    """Hazard threshold (tau) this env uses in its reward table.

    Factories call this before reset(), when env.scenario is still None,
    so we rebuild the scenario config to read its threshold.
    Envs that mix scenarios just use the default.
    """
    from idg_social_nav.discomfort import DiscomfortParams
    from idg_social_nav.scenarios import make_scenario
    if len(env._scenario_names) == 1:
        return make_scenario(env._scenario_names[0]).discomfort_params.high_threshold
    return DiscomfortParams().high_threshold


def oracle_validator_factory(env: SocialNavEnv) -> RLModule:
    from idg_social_nav.rl_modules.oracle_validator import OracleValidatorRLM
    return build_inference_module(
        env, "validator", OracleValidatorRLM,
        model_config={"tau": _env_tau(env)},
    )


def always_obey_factory(env: SocialNavEnv) -> RLModule:
    from idg_social_nav.rl_modules.always_obey_validator import AlwaysObeyValidatorRLM
    return build_inference_module(env, "validator", AlwaysObeyValidatorRLM)


def fixed_blend_factory(p: float, seed: int = 0) -> ModuleFactory:
    def factory(env: SocialNavEnv) -> RLModule:
        from idg_social_nav.rl_modules.fixed_blend_validator import make_fixed_blend_class
        return build_inference_module(env, "validator", make_fixed_blend_class(p, seed=seed))

    return factory


def learned_validator_checkpoint(exp_name: str) -> Path:
    return (
        TUNE_DIR / exp_name / "best_checkpoint"
        / "learner_group" / "learner" / "rl_module" / ValidatorPolicies.LEARNED
    )


def load_learned_validator(exp_name: str) -> ModuleFactory:
    ckpt = learned_validator_checkpoint(exp_name)

    def factory(_env: SocialNavEnv) -> RLModule:
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Learned validator checkpoint not found at {ckpt}. "
                f"Run `python run_experiments.py` first."
            )
        return RLModule.from_checkpoint(str(ckpt))

    return factory


def _extract_action(module: RLModule, out: dict) -> int:
    """Deterministic eval. with explicit actions, else argmax over the logits."""
    if SampleBatch.ACTIONS in out:
        return int(out[SampleBatch.ACTIONS].item())
    logits = out[SampleBatch.ACTION_DIST_INPUTS]
    return int(torch.argmax(logits, dim=-1).item())


def _batch(obs_agent: dict) -> dict:
    return {SampleBatch.OBS: tree.map_structure(
        lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs_agent)}


def run_pairing(
    name: str,
    proposer_factory: ModuleFactory,
    validator_factory: ModuleFactory,
    scenario_name: str,
    variants: list[dict] | None = None,
    reps: int = 1,
    advisor: Advisor | None = None,
    seed: int = 0,
    reward_variant: str = "binary",
    override_semantics: str = "adopt",
    ped_hesitation: float = 0.0,
    ped_route_noise: float = 0.0,
    record_video: bool = False,
    video_dir: Path | str = VIDEO_DIR,
) -> dict:
    """Run a (proposer, validator) pairing over the variants of one scenario.

    Episodes are deterministic given (scenario, variant) unless the advisor,
    the validator, or the env (ped_hesitation > 0) is stochastic;
    only then do reps > 1 add information.
    """
    from idg_social_nav.rl_modules.oracle_validator import OracleValidatorRLM

    if variants is None:
        variants = enumerate_variants(scenario_name)

    env = SocialNavEnv(
        scenario=scenario_name,
        advisor=advisor,
        reward_variant=reward_variant,
        override_semantics=override_semantics,
        randomize_variant=False,
        ped_hesitation=ped_hesitation,
        ped_route_noise=ped_route_noise,
        record_render=record_video,
        seed=seed,
    )

    proposer_module = proposer_factory(env)
    validator_module = validator_factory(env)

    episodes: list[dict] = []
    validator_rewards: list[float] = []
    result: dict = {"name": name, "scenario": scenario_name}

    episode_specs = [(rep, variant) for rep in range(reps) for variant in variants]
    first_episode = True
    for _rep, variant in tqdm.tqdm(episode_specs, desc=f"{name}@{scenario_name}"):
        obs, _ = env.reset(options={"scenario": scenario_name, "variant": dict(variant)})
        tau = env.scenario.discomfort_params.high_threshold
        max_steps = env.max_steps

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        step_infos: list[dict] = []
        decisions: list[tuple[int, int]] = []

        while not terminated["__all__"]:
            actions: dict = {}
            if "proposer" in obs:
                out = proposer_module.forward_inference(_batch(obs["proposer"]))
                actions["proposer"] = _extract_action(proposer_module, out)
            elif "validator" in obs:
                val_obs = obs["validator"]
                out = validator_module.forward_inference(_batch(val_obs))
                validator_action = _extract_action(validator_module, out)
                actions["validator"] = validator_action
                oracle_action = OracleValidatorRLM.decide(
                    np.asarray(val_obs["env"]),
                    int(np.argmax(val_obs["proposer_action"])),
                    int(np.argmax(val_obs["advice"])),
                    tau=tau,
                )
                decisions.append((int(validator_action), int(oracle_action)))
            else:
                raise RuntimeError(f"No actionable agent in obs: {list(obs.keys())}")

            obs, rewards, terminated, truncated, infos = env.step(actions)
            if "validator" in rewards:
                validator_rewards.append(float(rewards["validator"]))
                step_infos.append(dict(infos["validator"]))
            if truncated["__all__"]:
                break

        reached_goal = bool(step_infos and step_infos[-1]["reached_goal"])
        ep = episode_metrics(step_infos, reached_goal, len(step_infos), max_steps, tau=tau)
        ep.update(confusion_counts(decisions))
        episodes.append(ep)

        if first_episode and record_video:
            video_path = Path(video_dir) / scenario_name / f"{name}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                env.save_video(str(video_path))
                result["video"] = str(video_path)
            except Exception as e:
                result["video_error"] = str(e)
            env._record_render = False
        first_episode = False

    agg = aggregate(episodes)
    n_decisions = sum(e.get("decisions", 0) for e in episodes)
    total_good = sum(e["good_overrides"] for e in episodes)
    total_bad = sum(e["bad_overrides"] for e in episodes)

    result.update(agg)
    result.update({
        "validator_mean_reward": (
            float(np.mean(validator_rewards)) if validator_rewards else 0.0),
        "n_validator_decisions": n_decisions,
        "good_disobey_pct": 100.0 * total_good / n_decisions if n_decisions else 0.0,
        "bad_disobey_pct": 100.0 * total_bad / n_decisions if n_decisions else 0.0,
    })

    if hasattr(validator_module, "_call_count"):
        result["llm_calls"] = int(validator_module._call_count)
        result["llm_cache_hits"] = int(validator_module._cache_hits)

    return result


def _fmt(value, spec: str = "6.2f") -> str:
    if value is None:
        return "n/a"
    return f"{value:{spec}}"


def format_table(results: list[dict]) -> str:
    cols = [
        ("pairing",   lambda r: r["name"]),
        ("scenario",  lambda r: r["scenario"]),
        ("eps",       lambda r: str(r["n_episodes"])),
        ("succ %",    lambda r: _fmt(r["success_pct"])),
        ("steps",     lambda r: _fmt(r["mean_steps"])),
        ("coll ep %", lambda r: _fmt(r["collision_episodes_pct"])),
        ("intr_sum",  lambda r: _fmt(r["mean_intrusion_sum"], "6.3f")),
        ("hi_intr",   lambda r: _fmt(r["high_intrusion_steps_mean"], "6.3f")),
        ("val_rew",   lambda r: _fmt(r["validator_mean_reward"], "+.4f")),
        ("ovr/ep",    lambda r: _fmt(r["overrides_per_ep"], "5.2f")),
        ("good/dec %", lambda r: _fmt(r["good_disobey_pct"])),
        ("bad/dec %", lambda r: _fmt(r["bad_disobey_pct"])),
        ("ovr_prec",  lambda r: _fmt(r["override_precision"], "5.3f")),
        ("ovr_rec",   lambda r: _fmt(r["override_recall"], "5.3f")),
        ("frozen %",  lambda r: _fmt(r["frozen_pct"])),
    ]
    headers = [h for h, _ in cols]
    rows = [[fn(r) for _, fn in cols] for r in results]
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]

    sep = "  ".join("-" * w for w in widths)
    lines = [sep,
             "  ".join(headers[i].ljust(widths[i]) for i in range(len(cols))),
             sep]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    lines.append(sep)
    return "\n".join(lines)


def print_table(results: list[dict]) -> None:
    print(format_table(results))


def _write_csv(results: list[dict], path: Path) -> None:
    keys = list(results[0].keys())
    for r in results[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})


def print_summary(results: list[dict], tag: str) -> None:
    from datetime import datetime

    header = ("\n" + "=" * 72 + "\n"
              f"IDG social-nav evaluation ({tag}, {len(results)} conditions)\n"
              + "=" * 72)
    body = format_table(results)
    print(header)
    print(body)

    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "eval"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    EVAL_DIR.mkdir(exist_ok=True)
    txt_path = EVAL_DIR / f"{script}_{tag}_{ts}.txt"
    txt_path.write_text(header + "\n" + body + "\n")
    csv_path = EVAL_DIR / f"{script}_{tag}_{ts}.csv"
    _write_csv(results, csv_path)
    print(f"\nSaved results to {txt_path}")
    print(f"Saved CSV to {csv_path}")

    for r in results:
        if "llm_calls" in r:
            denom = max(1, r["llm_calls"] + r["llm_cache_hits"])
            print(f"\n{r['name']} LLM stats: {r['llm_calls']} unique calls, "
                  f"{r['llm_cache_hits']} cache hits "
                  f"(hit rate {r['llm_cache_hits'] / denom * 100:.1f}%)")
        if "video" in r:
            print(f"Saved {r['video']}")
        if "video_error" in r:
            print(f"Video save failed for {r['name']}: {r['video_error']}")
