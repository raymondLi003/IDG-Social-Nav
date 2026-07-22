# IDG x VLM-Social-Nav: Intelligent Disobedience for Social Navigation

A grid-world benchmark that combines two ideas:

- **VLM-Social-Nav** (Song et al., RA-L 2025): a vision-language model watches
  the robot's camera and suggests a polite move whenever a pedestrian is near.
- **The Intelligent Disobedience Game** (Hornig & Mirsky, RaD-AI @ AAMAS '26):
  a follower that decides when to override its leader's command, scored as
  good or bad disobedience.

VLM-Social-Nav blends the VLM's advice into the planner with a fixed weight.
Here a **validator** replaces that weight: at every step it decides whether to
obey the planner or execute the advice, and each override is scored.

## How it works

Each turn:

1. **Leader** proposes a move toward the goal (forward / turn left / turn
   right). It cannot see the social discomfort field.
2. **Advisor** (a VLM, or a built-in rule-based advisor) suggests a polite
   move. We only query it only when a pedestrian is nearby.
3. **Validator** sees what the leader can't: the pedestrian's personal-space
   ("discomfort") field and the advice. The validator picks **obey** or **override**.
4. **Reward**: +1 for an override that avoided a hazard, -1 for a unnecessary
   override or for obeying into a hazard, 0 otherwise. 


Every validator has the same interface. We have the observation as input, obey or override as the output, 
so that we can compare the scripted, learned, and LLM validators. The
validators compared: `always_obey`, a `fixed_blend` sweep (execute advice
with fixed probability p in the fixed-weight baseline), `oracle` (that we 
optimized for the reward as the upper ceiling), `ppo` (learned), and `llm`.

## Scenarios

| Scenario           | Grid | Pedestrian                                   | Variants | Punishes |
|--------------------|------|----------------------------------------------|----------|----------|
| `frontal_approach` | 7x11 | walks down the robot's row toward it         | 6        | staying in the lane |
| `narrow_doorway`   | 7x11 | comes through a 1-wide gap from the far side | 4        | always yielding (freezing) |
| `intersection`     | 9x11 | crosses the robot's corridor at a junction   | 8        | bad timing |
| `frontal_gesture`  | 7x11 | frontal approach, showing STOP or GO         | 8        | ignoring the gesture |

Pedestrians walk fixed, scripted paths, so every episode is deterministic
given the scenario and the variant. That makes every possible advisor query listable
in advance, so that VLM advice can be precomputed once into a cache and replayed for free during training and evaluation.

## Measuring where failures come from

Three switches separate the possible failure points:

- **Perception**: pixel mode (the VLM must read the gesture from the rendered
  image) vs. symbolic mode (the gesture is given as text).
- **Advice**: `--advisor noisy:0.3` corrupts 30% of the advice on purpose, to
  test how well validators filter bad suggestions.
- **Override**: every validator decision is graded against the oracle on the
  same observation (the precision/recall columns in the results table).

## Quickstart

```bash
# install (Python >= 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # add API keys only if you use the LLM/VLM backends

# tests and linter
python -m pytest tests/
ruff check idg_social_nav tests

# baseline comparison: always-obey, fixed-blend sweep, oracle
idg-eval

# train the PPO validator
idg-train --scenario all

# compare everything, including the trained and LLM validators
idg-eval --validators always_obey,fixed_blend,oracle,ppo,llm

# precompute the VLM advice caches (one-time API cost)
idg-precompute-advice --scenario all --mode pixel --backend openai
idg-precompute-advice --scenario all --mode symbolic --backend openai

# evaluate against a frozen cache
idg-eval --advisor "cached:advice_cache/{scenario}_pixel_openai.json"

# figures: Pareto plot per scenario + disobedience breakdown
idg-plot
```

The `idg-*` commands can also run without installing, e.g.
`python -m idg_social_nav.run_eval`. Outputs go to `logs/`, `eval_results/`,
`curves/`, `videos/`, and `advice_cache/`.

## Module map (package `idg_social_nav/`)

| Module | Role |
|---|---|
| `core.py` | shared types: actions, advice, gestures, advisor interface |
| `grid.py` | BFS and grid geometry helpers |
| `paths.py` | output folder locations |
| `discomfort.py` | the pedestrian personal-space field |
| `scenarios.py` | the four scenarios + their variants |
| `env.py` | the turn-based environment (leader -> advisor -> validator) |
| `advisor_scripted.py` | rule-based advisor + `NoisyAdvisor` |
| `vlm_advisor.py` | VLM advisor (OpenAI / Anthropic / Tufts proxy) + cache |
| `precompute_advice.py` | CLI: fill the advice cache |
| `render.py` | draws the board as an image (the "camera feed") |
| `llm_validator.py` | LLM validator (ASCII view, answers obey/override) |
| `llmproxy.py` | Tufts LLM-proxy client (text only) |
| `rl_modules/` | leader + oracle / always-obey / fixed-blend validators |
| `config.py` | experiment config and RLlib training setup |
| `metrics.py` | episode metrics and oracle agreement |
| `eval_common.py` | shared evaluation loop, tables, CSVs |
| `run_experiments.py` | CLI: train the PPO validator |
| `run_eval.py` | CLI: the main validator comparison |
| `plot_results.py` | CLI: figures from the eval CSVs |

## References

- D. Song, J. Liang, A. Payandeh, A. H. Raj, X. Xiao, and D. Manocha,
  "VLM-Social-Nav: Socially Aware Robot Navigation Through Scoring Using
  Vision-Language Models," IEEE RA-L, 2025.
- B. Hornig and R. Mirsky, "The Intelligent Disobedience Game: Formulating
  Disobedience in Stackelberg Games and Markov Decision Processes," RaD-AI
  Workshop @ AAMAS, 2026.
