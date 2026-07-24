"""Tests for llm_validator: ASCII rendering, query building, tagged
strategy/thought_process/decision parsing,"""

import json

import numpy as np
import pytest
import torch
import tree
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModuleSpec

from idg_social_nav.core import (
    CH_AGENT,
    CH_DISCOMFORT,
    CH_GOAL,
    CH_PED,
    CH_WALL,
    Advice,
    Gesture,
    ProposerAction,
)
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.llm_validator import (
    RULEBOOK_PROMPT,
    LLMValidatorSocial,
    _build_query,
    _parse_explain,
    _parse_response,
    _render_egocentric,
)


def _crafted_obs() -> np.ndarray:
    """3x5 validator view exercising every glyph and the priority order
    (wall > agent > goal > pedestrian > discomfort bands)."""
    arr = np.zeros((3, 5, 5), dtype=np.float32)
    # row 0: wall + the four ego-relative pedestrian facings
    arr[0, 0, CH_WALL] = 1.0
    arr[0, 1, CH_PED] = 0.25  # walking away -> A
    arr[0, 2, CH_PED] = 0.5   # to the right -> >
    arr[0, 3, CH_PED] = 0.75  # toward you -> V
    arr[0, 4, CH_PED] = 1.0   # to the left -> <
    # row 1: goal + the discomfort band boundaries
    arr[1, 0, CH_GOAL] = 1.0
    arr[1, 1, CH_DISCOMFORT] = 0.5   # at tau -> !
    arr[1, 2, CH_DISCOMFORT] = 0.15  # mild band edge -> ~
    arr[1, 3, CH_DISCOMFORT] = 0.1   # below the mild band -> .
    # row 2: every glyph beats the discomfort band underneath it
    arr[2, 0, CH_WALL] = 1.0
    arr[2, 0, CH_DISCOMFORT] = 0.9
    arr[2, 1, CH_AGENT] = 1.0
    arr[2, 1, CH_DISCOMFORT] = 0.9
    arr[2, 2, CH_GOAL] = 1.0
    arr[2, 2, CH_DISCOMFORT] = 0.9
    arr[2, 3, CH_PED] = 0.75
    arr[2, 3, CH_DISCOMFORT] = 0.9
    return arr


_GOLDEN = "\n".join([
    "# A > V <",
    "G ! ~ . .",
    "# ^ G V .",
])


class TestRenderEgocentric:
    def test_golden_render(self):
        assert _render_egocentric(_crafted_obs()) == _GOLDEN

    def test_accepts_torch_tensors(self):
        assert _render_egocentric(torch.tensor(_crafted_obs())) == _GOLDEN

    def test_never_contains_numeric_field_values(self):
        ascii_grid = _render_egocentric(_crafted_obs())
        assert set(ascii_grid) <= set("#^GAV<>!~. \n")


class TestBuildQuery:
    def test_contains_the_three_context_lines(self):
        query = _build_query(
            RULEBOOK_PROMPT, _GOLDEN,
            int(ProposerAction.forward), int(Advice.WAIT), int(Gesture.STOP),
        )
        assert "Leader proposes: forward" in query
        assert "Advisor suggests: wait" in query
        assert "Pedestrian gesture: stop" in query
        assert _GOLDEN in query
        assert query.rstrip().endswith("Answer:")

    def test_advice_and_gesture_names(self):
        query = _build_query(
            RULEBOOK_PROMPT, _GOLDEN,
            int(ProposerAction.turn_left), int(Advice.TURN_RIGHT), int(Gesture.GO),
        )
        assert "Leader proposes: turn_left" in query
        assert "Advisor suggests: turn_right" in query
        assert "Pedestrian gesture: go" in query


class TestParseResponse:
    def test_bare_digits(self):
        assert _parse_response("0") == 0
        assert _parse_response("1") == 1

    def test_digit_embedded_in_prose(self):
        assert _parse_response("The answer is 1.") == 1

    def test_first_digit_wins(self):
        assert _parse_response("0 (not 1)") == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_response("")

    def test_no_digit_raises(self):
        with pytest.raises(ValueError):
            _parse_response("obey the leader")


class TestParseExplain:
    def test_full_tagged_response(self):
        text = ("<strategy>maximize the reward table</strategy>"
                "<thought_process>the V ahead makes forward hazardous</thought_process>"
                "<decision>1</decision>")
        strategy, thought, decision = _parse_explain(text)
        assert strategy == "maximize the reward table"
        assert thought == "the V ahead makes forward hazardous"
        assert decision == 1

    def test_missing_tags_fall_back_to_scanning(self):
        strategy, thought, decision = _parse_explain("I choose 0 because it is safe")
        assert strategy == ""
        assert thought == ""
        assert decision == 0

    def test_rulebook_prompt_asks_for_tagged_answer(self):
        assert "<strategy>" in RULEBOOK_PROMPT
        assert "<thought_process>" in RULEBOOK_PROMPT
        assert "<decision>" in RULEBOOK_PROMPT
        assert "Reply with exactly one digit (0 or 1) and nothing else." not in RULEBOOK_PROMPT


class _FakeProxy:
    """Stub for llmproxy.LLMProxy: counts calls, returns a canned result."""

    def __init__(self, result: str = "1"):
        self.calls = 0
        self.result = result

    def generate(self, **kwargs):
        self.calls += 1
        return {"result": self.result}


def _build_module(module_class, tmp_path):
    env = SocialNavEnv(scenario="frontal_approach", randomize_variant=False)
    module = RLModuleSpec(
        module_class=module_class,
        observation_space=env.observation_spaces["validator"],
        action_space=env.action_spaces["validator"],
        inference_only=True,
    ).build()
    module._log_path = tmp_path / "log.jsonl"
    module._explain_path = tmp_path / "explain.jsonl"
    # a non-OpenAI model name keeps _generate on the proxy path, so the
    # _FakeProxy stub intercepts every call and no test ever hits the network
    module.MODEL_NAME = "stub-proxy-model"
    return env, module


def _validator_batch(env: SocialNavEnv) -> dict:
    env.reset(options={
        "scenario": "frontal_approach",
        "variant": {"ped_start_col": 7, "ped_delay": 0},
    })
    obs, *_ = env.step({"proposer": int(ProposerAction.forward)})
    return {SampleBatch.OBS: tree.map_structure(
        lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs["validator"])}


class TestLLMValidatorSocial:
    def test_cache_hit_avoids_second_call(self, tmp_path):
        env, module = _build_module(LLMValidatorSocial, tmp_path)
        module._proxy = fake = _FakeProxy(result="1")
        batch = _validator_batch(env)

        out = module.forward_inference(batch)
        assert int(out[SampleBatch.ACTIONS][0]) == 1
        assert fake.calls == 1
        assert module._cache_hits == 0

        out = module.forward_inference(batch)
        assert int(out[SampleBatch.ACTIONS][0]) == 1
        assert fake.calls == 1
        assert module._cache_hits == 1

    def test_miss_writes_one_jsonl_record(self, tmp_path):
        env, module = _build_module(LLMValidatorSocial, tmp_path)
        module._proxy = _FakeProxy(result="0")
        batch = _validator_batch(env)

        module.forward_inference(batch)
        module.forward_inference(batch)  # cache hit: no second record

        records = [json.loads(line)
                   for line in module._log_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["decision"] == 0
        assert records[0]["proposer_action"] == int(ProposerAction.forward)
        assert "grid" in records[0]

    def test_tagged_response_parses_and_logs_reasoning(self, tmp_path):
        env, module = _build_module(LLMValidatorSocial, tmp_path)
        module._proxy = _FakeProxy(result=(
            "<strategy>obey unless hazardous</strategy>"
            "<thought_process>the lane is clear</thought_process>"
            "<decision>0</decision>"))
        batch = _validator_batch(env)

        out = module.forward_inference(batch)
        assert int(out[SampleBatch.ACTIONS][0]) == 0

        records = [json.loads(line)
                   for line in module._explain_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["strategy"] == "obey unless hazardous"
        assert records[0]["thought_process"] == "the lane is clear"
        assert records[0]["decision"] == 0
