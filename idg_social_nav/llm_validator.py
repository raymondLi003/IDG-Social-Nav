"""LLM validator for the social-navigation grid.

The validator judges the leader's proposal against privileged evidence:
its egocentric view 
(with the discomfort channel rendered as ASCII bands, instead of numeric values),
the leader's proposed action, the advisor's suggestion, and the nearest visible pedestrian gesture. 
It answers with a single digit: 0 = obey the leader, 1 = override
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from idg_social_nav.core import (
    CH_AGENT,
    CH_DISCOMFORT,
    CH_GOAL,
    CH_PED,
    CH_WALL,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Advice,
    Gesture,
    ProposerAction,
    ValidatorAction,
    decode_ped_facing,
)
from idg_social_nav.llmproxy import LLMProxy
from idg_social_nav.paths import LOG_DIR

# Default model
MODEL_NAME = "us.anthropic.claude-3-haiku-20240307-v1:0"


# Social rulebook prompt
RULEBOOK_PROMPT = """
You are the safety validator on a mobile robot navigating among pedestrians. 
A leader proposes an action and a social advisor may suggest an alternative. 
You decide whether to allow the leader's action (0 = obey) or override it (1 = override). 
Overriding executes the advisor's suggestion; when the advisor suggests none, overriding just blocks the leader for this turn. 
You are given the robot's current view of the grid, the leader's proposed action, the advisor's suggestion, and any pedestrian gesture.

GRID LEGEND:
- (^) the robot. It always faces UP in this view, so "forward" is the cell directly above the ^.
- (#) wall
- (G) goal tile
- (A) pedestrian walking away from you (up in this view)
- (>) pedestrian walking to your right
- (V) pedestrian walking toward you (down in this view)
- (<) pedestrian walking to your left
- (!) high-discomfort tile: inside a pedestrian's personal space
- (~) mildly uncomfortable tile: edge of a pedestrian's personal space
- (.) comfortable empty tile

SOCIAL RULES:
1. The leader can either try to move forward, turn left, or turn right without moving from the original spot
2. Walking into or across a pedestrian's personal space (!) is a social violation and gives you a negative reward
3. Mildly uncomfortable tiles (~) are acceptable to enter
4. Blocking a person's path or standing in their way is harmful
5. Keep right when passing a pedestrian
6. A STOP gesture means the pedestrian asserts right of way: clear its lane and wait
7. A GO gesture means the pedestrian yields: proceed
8. Preventing the leader from executing a safe action gives you a negative reward; needless blocking (freezing in place) is a failure
9. Walls (#) are safe boundaries that the leader can attempt to step into but will bounce back without harm. This gives you no reward
10. Landing on goal tiles (G) ends the game, yielding no reward for you

REWARD TABLE:
- Overriding when the leader's action would enter high discomfort (!) or bump a pedestrian: +1
- Overriding when the leader's action was safe: -1
- Obeying when the leader's action would enter high discomfort (!) or bump a pedestrian: -1
- Otherwise: 0

GOAL: Maximize your reward.

Reply with exactly one digit (0 or 1) and nothing else. No explanation, no whitespace, no punctuation.
"""

_PROPOSER_ACTION_NAMES = {
    int(ProposerAction.forward): "forward",
    int(ProposerAction.turn_left): "turn_left",
    int(ProposerAction.turn_right): "turn_right",
}

_ADVICE_NAMES = {
    int(Advice.NONE): "none",
    int(Advice.TURN_LEFT): "turn_left",
    int(Advice.FORWARD): "forward",
    int(Advice.TURN_RIGHT): "turn_right",
    int(Advice.WAIT): "wait",
}

_GESTURE_NAMES = {
    int(Gesture.NONE): "none",
    int(Gesture.STOP): "stop",
    int(Gesture.GO): "go",
}

_PED_GLYPHS = {UP: "A", RIGHT: ">", DOWN: "V", LEFT: "<"}




def _slugify(model_name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model_name).strip("_")


def _render_egocentric(env_obs: torch.Tensor) -> str:
    """Render the validator's egocentric observation as ASCII.

    Channels: wall, agent, goal, pedestrian (facing-coded intensity), discomfort. 
    The agent always faces UP, sitting at the bottom-center cell.
    Pedestrians render as facing glyphs (A > V <), 
    the discomfort field renders as bands (! high, ~ mild, . comfortable). 
    These are not numeric values.
    """
    arr = env_obs.detach().cpu().numpy() if hasattr(env_obs, "detach") else env_obs
    h, w, _ = arr.shape
    rows = []
    for r in range(h):
        row_chars = []
        for c in range(w):
            if arr[r, c, CH_WALL] > 0.5:
                row_chars.append("#")
            elif arr[r, c, CH_AGENT] > 0.5:
                row_chars.append("^")
            elif arr[r, c, CH_GOAL] > 0.5:
                row_chars.append("G")
            elif arr[r, c, CH_PED] > 0:
                facing = decode_ped_facing(float(arr[r, c, CH_PED]))
                row_chars.append(_PED_GLYPHS[facing])
            elif arr[r, c, CH_DISCOMFORT] >= 0.5:
                row_chars.append("!")
            elif arr[r, c, CH_DISCOMFORT] >= 0.15:
                row_chars.append("~")
            else:
                row_chars.append(".")
        rows.append(" ".join(row_chars))
    return "\n".join(rows)


def _build_query(
        system_prompt: str,
        grid_ascii: str,
        proposer_action: int,
        advice: int,
        gesture: int,
) -> str:
    """Build the query string for the LLM prompt (rulebook only)."""
    action_name = _PROPOSER_ACTION_NAMES.get(proposer_action, f"unknown_{proposer_action}")
    advice_name = _ADVICE_NAMES.get(advice, f"unknown_{advice}")
    gesture_name = _GESTURE_NAMES.get(gesture, f"unknown_{gesture}")
    return (
        f"{system_prompt}\n\n"
        f"Current situation:\nView:\n{grid_ascii}\n"
        f"Leader proposes: {action_name}\n"
        f"Advisor suggests: {advice_name}\n"
        f"Pedestrian gesture: {gesture_name}\n"
        f"Answer:"
    )


def _parse_response(result_text: str) -> int:
    """Extract the validator's decision (0 or 1) from the LLM's response text."""
    if not result_text:
        raise ValueError("LLM response is empty.")
    for ch in result_text:
        if ch in ("0", "1"):
            return int(ch)
    raise ValueError(f"No 0/1 digit in LLM response: {result_text!r}")


# ask the models to articulate their own reasoning for their decisions
_ANSWER_FORMAT_LINE = (
    "Reply with exactly one digit (0 or 1) and nothing else. "
    "No explanation, no whitespace, no punctuation."
)

_EXPLAIN_FORMAT = """First reason, then decide. Respond in EXACTLY this format, and nothing outside these tags:
<strategy>your overall approach in 1-2 sentences: what you are optimizing and how you generally decide</strategy>
<thought_process>your step-by-step reasoning for THIS situation: where the pedestrians (A > V <) are and which way they walk, which cells are high discomfort (!), what the leader's proposed action would do, what the advisor suggests and what any gesture implies, and whether obeying or overriding is the rewarded choice</thought_process>
<decision>0 or 1</decision>
0 = obey the leader, 1 = override (execute the advisor's suggestion)."""


def _explain_prompt(base: str) -> str:
    """Turn a single-digit asking prompt into one that asks for <strategy>/<thought_process>/<decision>."""
    if _ANSWER_FORMAT_LINE in base:
        return base.replace(_ANSWER_FORMAT_LINE, _EXPLAIN_FORMAT)
    return base.rstrip() + "\n\n" + _EXPLAIN_FORMAT


_TAG_RE = {name: re.compile(rf"<{name}>(.*?)</{name}>", re.S | re.I)
           for name in ("strategy", "thought_process", "decision")}


def _parse_explain(text: str):
    """Return (strategy, thought_process, decision_int).
    fallback to scanning the text for 0 or 1 if the decision was not found
    """
    def _tag(name: str) -> str:
        m = _TAG_RE[name].search(text or "")
        return m.group(1).strip() if m else ""
    strategy = _tag("strategy")
    thought = _tag("thought_process")
    dec_raw = _tag("decision")
    try:
        decision = _parse_response(dec_raw) if dec_raw else _parse_response(text)
    except ValueError:
        decision = _parse_response(text)
    return strategy, thought, decision


class LLMValidatorSocial(RLModule):
    """Validator policy that uses an LLM to decide obey or override for the
    leader's proposed action, given the social rulebook prompt, the ASCII
    egocentric view, the advisor's suggestion, and the pedestrian gesture."""

    # Override in subclasses to swap models
    MODEL_NAME: str = MODEL_NAME
    SYSTEM_PROMPT: str = RULEBOOK_PROMPT
    LOG_TAG: str = "rulebook"
    # when true, append the explain tag to the llm names
    EXPLAIN: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy: LLMProxy | None = None
        self._cache: dict[tuple, int] = {}
        self._call_count = 0
        self._cache_hits = 0
        # make sure the results do not overwrite each other
        slug = _slugify(self.MODEL_NAME)
        self._log_path = LOG_DIR / f"llm_social_{self.LOG_TAG}__{slug}.jsonl"
        # reasoning format only
        self._explain_path = LOG_DIR / f"llm_social_explain_{self.LOG_TAG}__{slug}.jsonl"

    def _ensure_proxy(self) -> LLMProxy:
        if self._proxy is None:
            self._proxy = LLMProxy()
        return self._proxy

    @staticmethod
    def _write_jsonl(path: Path, record: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _decide(
            self,
            single_obs: torch.Tensor,
            proposer_action: int,
            advice: int,
            gesture: int,
    ) -> int:
        obs_arr = single_obs.detach().cpu().numpy() if hasattr(single_obs, "detach") else single_obs
        # Cache key based on the raw observation bytes and the decision context
        cache_key = (obs_arr.tobytes(), proposer_action, advice, gesture)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        grid_ascii = _render_egocentric(single_obs)
        system_prompt = _explain_prompt(self.SYSTEM_PROMPT) if self.EXPLAIN else self.SYSTEM_PROMPT
        query = _build_query(system_prompt, grid_ascii, proposer_action, advice, gesture)

        proxy = self._ensure_proxy()
        self._call_count += 1
        # LLM call
        response = proxy.generate(
            model=self.MODEL_NAME,
            system=system_prompt,
            query=query,
            temperature=0.0,
            session_id=f"llm-validator-{self.LOG_TAG}-{_slugify(self.MODEL_NAME)}-{self._call_count}",
        )

        strategy, thought = "", ""
        if not isinstance(response, dict) or "error" in response:
            print(f"[llm_validator] proxy error, defaulting to obey: {response}")
            decision = ValidatorAction.obey.value
            result_text = ""
        else:
            result_text = response.get("result", "")
            try:
                if self.EXPLAIN:
                    strategy, thought, decision = _parse_explain(result_text)
                else:
                    decision = _parse_response(result_text)
            except ValueError as e:
                print(f"[llm_validator] response parsing error: {e}, defaulting to obey")
                decision = ValidatorAction.obey.value
        # Cache the decision
        self._cache[cache_key] = decision

        action_name = _PROPOSER_ACTION_NAMES.get(proposer_action, f"unknown_{proposer_action}")
        advice_name = _ADVICE_NAMES.get(advice, f"unknown_{advice}")
        gesture_name = _GESTURE_NAMES.get(gesture, f"unknown_{gesture}")
        if self.EXPLAIN:
            # put the reasoning into a different dataset
            self._write_jsonl(self._log_path, {
                "model": self.MODEL_NAME, "call": self._call_count,
                "proposer_action": proposer_action, "advice": advice,
                "gesture": gesture, "grid": grid_ascii, "decision": decision,
            })
            self._write_jsonl(self._explain_path, {
                "model": self.MODEL_NAME, "call": self._call_count,
                "proposer_action": proposer_action, "action_name": action_name,
                "advice": advice, "advice_name": advice_name,
                "gesture": gesture, "gesture_name": gesture_name,
                "grid": grid_ascii,
                "strategy": strategy, "thought_process": thought, "decision": decision,
            })
        else:
            self._write_jsonl(self._log_path, {
                "model": self.MODEL_NAME, "call": self._call_count,
                "proposer_action": proposer_action, "advice": advice,
                "gesture": gesture, "grid": grid_ascii,
                "result": result_text, "decision": decision,
            })
        return decision

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """For each observation in the batch, query the LLM to get a validator decision."""
        env_obs = batch[SampleBatch.OBS]["env"]
        # One-hot context vectors, collapse to integer IDs
        proposer_action_ids = torch.argmax(batch[SampleBatch.OBS]["proposer_action"], dim=-1)
        advice_ids = torch.argmax(batch[SampleBatch.OBS]["advice"], dim=-1)
        gesture_ids = torch.argmax(batch[SampleBatch.OBS]["gesture"], dim=-1)

        batch_size = len(env_obs)
        actions = []
        # Process each observation in the batch
        for i in range(batch_size):
            actions.append(self._decide(
                env_obs[i],
                int(proposer_action_ids[i].item()),
                int(advice_ids[i].item()),
                int(gesture_ids[i].item()),
            ))

        return {SampleBatch.ACTIONS: batch_func(actions)}


class LLMValidatorSocialExplain(LLMValidatorSocial):
    """Same rulebook prompt, but the model first outputs <strategy>/<thought_process>/
    <decision>.
    The reasoning is logged to a SEPARATE dataset
    (logs/llm_social_explain_explain__<model>.jsonl) for analysis.
    """

    LOG_TAG = "explain"
    EXPLAIN = True
