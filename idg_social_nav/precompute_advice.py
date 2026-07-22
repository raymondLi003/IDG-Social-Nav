"""Enumerate every reachable advisor query states and precompute advice caches to save api call costs

For every scenario variant, every pedestrian rail position (with the
variant's gesture), and every interior agent cell x 4 directions where the
pedestrian is inside the validator's egocentric view,
build an AdvisorContext and query the advisor through CachedAdvisor.

The resulting JSON cache makes VLM advisors deterministic, free, and usable
inside RLlib env runners without any LLM API calls.

Usage:
    python -m idg_social_nav.precompute_advice --scenario all \
        --backend scripted --mode symbolic
    python -m idg_social_nav.precompute_advice --scenario frontal_gesture \
        --backend openai --mode pixel --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from idg_social_nav.core import AdvisorContext, PedestrianSnapshot
from idg_social_nav.paths import ADVICE_CACHE_DIR
from idg_social_nav.scenarios import (
    SCENARIO_NAMES,
    VIEW_RADIUS,
    PedestrianState,
    enumerate_variants,
    facing_toward,
    make_scenario,
)
from idg_social_nav.vlm_advisor import (
    AnthropicBackend,
    CachedAdvisor,
    LLMProxyBackend,
    OpenAIBackend,
    VLMAdvisor,
    context_cache_key,
)

# rough per-call token footprints for the cost estimate
_EST_TOKENS_PER_CALL = {"symbolic": 400, "pixel": 1200}
_EST_USD_PER_MTOK = 1.0


def _rail_facing(rail, idx: int, initial: int) -> int:
    """Facing of a scripted pedestrian at rail index idx (derived from the
    step onto that cell, matching PedestrianState.step)."""
    if idx == 0:
        return initial
    return facing_toward(rail[idx - 1], rail[idx], initial)


def _ped_visible(agent_pos, agent_dir, ped_pos,
                 view_radius: int = VIEW_RADIUS) -> bool:
    """The env's advice gate: pedestrian within detection range (Chebyshev
    distance within the view radius; heading-independent, matching
    SocialNavEnv._visible_ped_indices)."""
    dr = ped_pos[0] - agent_pos[0]
    dc = ped_pos[1] - agent_pos[1]
    return max(abs(dr), abs(dc)) <= view_radius


def enumerate_query_states(name: str):
    """Yield (cfg, agent_pos, agent_dir, ped_specs) for every gated query
    state; ped_specs is one (rail_idx, pos, facing, gesture, visible) per
    pedestrian, with at least one pedestrian visible."""
    for variant in enumerate_variants(name):
        cfg = make_scenario(name, variant)
        rails = []
        for ped_cfg in cfg.pedestrians:
            initial = PedestrianState.from_config(ped_cfg).facing
            rails.append([
                (i, ped_cfg.rail[i],
                 _rail_facing(ped_cfg.rail, i, initial), ped_cfg.gesture)
                for i in range(len(ped_cfg.rail))
            ])
        h, w = cfg.walls.shape
        free = [
            (r, c) for r in range(h) for c in range(w)
            if cfg.walls[r, c] == 0 and (r, c) != tuple(cfg.goal_pos)
        ]
        for combo in itertools.product(*rails):
            ped_cells = {spec[1] for spec in combo}
            for pos in free:
                if pos in ped_cells:
                    continue
                for direction in range(4):
                    flags = [
                        _ped_visible(pos, direction, spec[1])
                        for spec in combo
                    ]
                    if not any(flags):
                        continue
                    ped_specs = [
                        (idx, ppos, facing, gesture, visible)
                        for (idx, ppos, facing, gesture), visible
                        in zip(combo, flags, strict=True)
                    ]
                    yield cfg, pos, direction, ped_specs


def _make_context(name, cfg, pos, direction, ped_specs,
                  frame_env=None) -> AdvisorContext:
    snapshots = [
        PedestrianSnapshot(pos=ppos, facing=facing, gesture=gesture,
                           visible=visible)
        for (_, ppos, facing, gesture, visible) in ped_specs
    ]
    frame_provider = None
    if frame_env is not None:
        frame_env.agent_pos = np.array(pos, dtype=np.int32)
        frame_env.agent_dir = int(direction)
        for state, (idx, ppos, facing, _, _) in zip(
                frame_env.ped_states, ped_specs, strict=True):
            state.rail_idx = idx
            state.pos = ppos
            state.facing = facing
            state.delay_remaining = 0
        frame_env._recompute_field()
        frame_provider = frame_env.render_frame
    return AdvisorContext(
        scenario_name=name,
        walls=cfg.walls,
        agent_pos=pos,
        agent_dir=direction,
        goal_pos=tuple(cfg.goal_pos),
        pedestrians=snapshots,
        step=0,
        frame_provider=frame_provider,
    )


def _build_advisor(args, cache_path) -> CachedAdvisor:
    if args.backend == "scripted":
        from idg_social_nav.advisor_scripted import ScriptedSocialAdvisor
        base = ScriptedSocialAdvisor()
    else:
        model_kwargs = {"model": args.model} if args.model else {}
        if args.backend == "openai":
            backend = OpenAIBackend(**model_kwargs)
        elif args.backend == "anthropic":
            backend = AnthropicBackend(**model_kwargs)
        else:
            backend = LLMProxyBackend(**model_kwargs)
        base = VLMAdvisor(backend, mode=args.mode)
    return CachedAdvisor(base, cache_path)


def _existing_keys(cache_path: Path) -> set[str]:
    if not cache_path.exists():
        return set()
    try:
        return set(json.loads(cache_path.read_text()))
    except (OSError, ValueError):
        return set()


def process_scenario(name: str, args) -> None:
    if args.cache_path is not None:
        cache_path = Path(args.cache_path)
    else:
        cache_path = ADVICE_CACHE_DIR / f"{name}_{args.mode}_{args.backend}.json"

    states = list(enumerate_query_states(name))
    if args.limit is not None:
        states = states[: args.limit]
    keys = [
        context_cache_key(_make_context(name, cfg, pos, d, ped_specs))
        for cfg, pos, d, ped_specs in states
    ]
    unique = set(keys)
    new = unique - _existing_keys(cache_path)

    n_variants = len(enumerate_variants(name))
    print(f"[{name}] variants={n_variants} states={len(states)} "
          f"unique_keys={len(unique)} new_queries={len(new)}")
    if args.backend == "scripted":
        print("  est cost: $0.00 (scripted advisor, no API calls)")
    else:
        tokens = len(new) * _EST_TOKENS_PER_CALL[args.mode]
        cost = tokens / 1e6 * _EST_USD_PER_MTOK
        print(f"  est cost: ~{len(new)} calls x "
              f"{_EST_TOKENS_PER_CALL[args.mode]} tok = {tokens / 1e6:.2f}M "
              f"tokens (~${cost:.2f} @ ${_EST_USD_PER_MTOK}/1M)")
    print(f"  cache: {cache_path}")
    if args.dry_run:
        return

    advisor = _build_advisor(args, cache_path)
    frame_env = None
    if args.mode == "pixel":
        from idg_social_nav.env import SocialNavEnv
        frame_env = SocialNavEnv(scenario=name, randomize_variant=False, seed=0)

    current_variant = None
    for cfg, pos, direction, ped_specs in states:
        if frame_env is not None and cfg.variant != current_variant:
            frame_env.reset(options={"scenario": name, "variant": cfg.variant})
            current_variant = cfg.variant
        context = _make_context(name, cfg, pos, direction, ped_specs,
                                frame_env=frame_env)
        advisor.advise(context)
    advisor.save()
    print(f"  done: queries={advisor._call_count} "
          f"cache_hits={advisor._cache_hits} cache_size={len(advisor)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute advisor caches over reachable query states.")
    parser.add_argument("--scenario", default="all",
                        choices=("all", *SCENARIO_NAMES))
    parser.add_argument("--mode", default="symbolic",
                        choices=("symbolic", "pixel"))
    parser.add_argument("--backend", default="scripted",
                        choices=("scripted", "openai", "anthropic", "llmproxy"))
    parser.add_argument("--model", default=None,
                        help="backend model override (backend default when unset)")
    parser.add_argument("--cache-path", default=None,
                        help="default: advice_cache/<scenario>_<mode>_<backend>.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="print state counts and cost estimate only")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of states per scenario")
    args = parser.parse_args(argv)
    if args.backend == "llmproxy" and args.mode == "pixel":
        parser.error("llmproxy backend is text-only; use --mode symbolic")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    names = list(SCENARIO_NAMES) if args.scenario == "all" else [args.scenario]
    for name in names:
        process_scenario(name, args)


if __name__ == "__main__":
    main()
