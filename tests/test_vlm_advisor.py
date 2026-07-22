"""Tests for vlm_advisor"""

import json

import numpy as np
import pytest

from idg_social_nav.core import (
    LEFT,
    Advice,
    Advisor,
    AdvisorContext,
    Gesture,
    PedestrianSnapshot,
    map_vlm_tokens_to_advice,
)
from idg_social_nav.scenarios import make_scenario
from idg_social_nav.vlm_advisor import (
    CachedAdvisor,
    LLMProxyBackend,
    VLMAdvisor,
    context_cache_key,
    parse_vlm_response,
)


def _context(gesture: Gesture = Gesture.NONE, visible: bool = True,
             frame_provider=None) -> AdvisorContext:
    cfg = make_scenario("frontal_approach", {"ped_start_col": 7, "ped_delay": 0})
    ped = PedestrianSnapshot(
        pos=(3, 7), facing=LEFT, gesture=gesture, visible=visible)
    return AdvisorContext(
        scenario_name=cfg.name,
        walls=cfg.walls,
        agent_pos=cfg.agent_start,
        agent_dir=cfg.agent_dir,
        goal_pos=cfg.goal_pos,
        pedestrians=[ped],
        step=0,
        frame_provider=frame_provider,
    )


class _FakeBackend:
    """A fake backend that returns canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []
        self.images = []

    def generate(self, prompt_text, image_png_bytes=None):
        self.calls += 1
        self.prompts.append(prompt_text)
        self.images.append(image_png_bytes)
        if self.responses:
            return self.responses.pop(0)
        return "no valid tokens here"


class _StubAdvisor(Advisor):
    def __init__(self, advice: Advice = Advice.TURN_RIGHT):
        self.advice = advice
        self.calls = 0

    def advise(self, context: AdvisorContext) -> Advice:
        self.calls += 1
        return self.advice


class TestPromptBuilding:
    def test_geometry_as_text(self, tmp_path):
        advisor = VLMAdvisor(_FakeBackend([]), mode="symbolic",
                             log_path=tmp_path / "log.jsonl")
        prompt = advisor.build_prompt(_context())
        assert "7x11 grid" in prompt
        assert "Robot at row 3, col 1, facing right" in prompt
        assert "Goal at row 3, col 9" in prompt
        assert "Pedestrian at row 3, col 7, facing left" in prompt
        assert "Manhattan distance 6" in prompt
        assert "in the robot's lane" in prompt
        assert "moving toward the robot" in prompt

    def test_symbolic_mode_includes_gesture_line(self, tmp_path):
        advisor = VLMAdvisor(_FakeBackend([]), mode="symbolic",
                             log_path=tmp_path / "log.jsonl")
        prompt = advisor.build_prompt(_context(gesture=Gesture.STOP))
        assert "Pedestrian gesture: stop" in prompt

    def test_pixel_mode_omits_gesture_line(self, tmp_path):
        advisor = VLMAdvisor(_FakeBackend([]), mode="pixel",
                             log_path=tmp_path / "log.jsonl")
        prompt = advisor.build_prompt(_context(gesture=Gesture.STOP))
        assert "Pedestrian gesture:" not in prompt
        assert "camera image" in prompt

    def test_no_visible_pedestrian_line(self, tmp_path):
        advisor = VLMAdvisor(_FakeBackend([]), mode="symbolic",
                             log_path=tmp_path / "log.jsonl")
        prompt = advisor.build_prompt(_context(visible=False))
        assert "No pedestrian is visible" in prompt


class TestParseVLMResponse:
    def test_exact_two_words(self):
        assert parse_vlm_response("left slow") == ("left", "slow")

    def test_messy_prose(self):
        assert parse_vlm_response(
            "I think you should go Left and Slow down here."
        ) == ("left", "slow")

    def test_case_insensitive(self):
        assert parse_vlm_response("STRAIGHT CONSTANT") == ("straight", "constant")

    def test_straight_does_not_match_right(self):
        assert parse_vlm_response("straight stop") == ("straight", "stop")

    def test_missing_speed_is_invalid(self):
        assert parse_vlm_response("turn left please") is None

    def test_empty_is_invalid(self):
        assert parse_vlm_response("") is None


class TestMapVLMTokens:
    @pytest.mark.parametrize("heading,speed,expected", [
        ("left", "slow", Advice.TURN_LEFT),
        ("left", "stop", Advice.WAIT),
        ("left", "constant", Advice.TURN_LEFT),
        ("straight", "slow", Advice.WAIT),
        ("straight", "stop", Advice.WAIT),
        ("straight", "constant", Advice.FORWARD),
        ("right", "slow", Advice.TURN_RIGHT),
        ("right", "stop", Advice.WAIT),
        ("right", "constant", Advice.TURN_RIGHT),
    ])
    def test_table(self, heading, speed, expected):
        assert map_vlm_tokens_to_advice(heading, speed) == expected

    def test_invalid_tokens_raise(self):
        with pytest.raises(ValueError):
            map_vlm_tokens_to_advice("backward", "slow")
        with pytest.raises(ValueError):
            map_vlm_tokens_to_advice("left", "fast")


class TestVLMAdvisorQuerying:
    def test_valid_first_answer(self, tmp_path):
        backend = _FakeBackend(["left constant"])
        advisor = VLMAdvisor(backend, mode="symbolic",
                             log_path=tmp_path / "log.jsonl")
        assert advisor.advise(_context()) == Advice.TURN_LEFT
        assert backend.calls == 1

    def test_reject_and_retry_appends_correction(self, tmp_path):
        backend = _FakeBackend(["gibberish", "right stop"])
        advisor = VLMAdvisor(backend, mode="symbolic",
                             log_path=tmp_path / "log.jsonl")
        assert advisor.advise(_context()) == Advice.WAIT
        assert backend.calls == 2
        assert "previous answer was invalid" in backend.prompts[1]

    def test_fallback_after_retries(self, tmp_path):
        backend = _FakeBackend([])  # never valid
        advisor = VLMAdvisor(backend, mode="symbolic", max_retries=2,
                             log_path=tmp_path / "log.jsonl")
        assert advisor.advise(_context()) == Advice.WAIT
        assert backend.calls == 3  # initial attempt + 2 retries

        records = [json.loads(line)
                   for line in (tmp_path / "log.jsonl").read_text().splitlines()]
        assert records[-1]["mapping_failure"] is True
        assert records[-1]["retries"] == 2
        assert records[-1]["tokens"] == ["straight", "slow"]

    def test_cached_wrapper_skips_backend(self, tmp_path):
        backend = _FakeBackend(["straight constant"])
        advisor = CachedAdvisor(
            VLMAdvisor(backend, mode="symbolic",
                       log_path=tmp_path / "log.jsonl"),
            cache_path=tmp_path / "cache.json")
        assert advisor.advise(_context()) == Advice.FORWARD
        assert advisor.advise(_context()) == Advice.FORWARD
        assert backend.calls == 1
        assert advisor._cache_hits == 1

    def test_pixel_mode_attaches_png(self, tmp_path):
        backend = _FakeBackend(["straight constant"])
        advisor = VLMAdvisor(backend, mode="pixel",
                             log_path=tmp_path / "log.jsonl")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        context = _context(frame_provider=lambda: frame)
        assert advisor.advise(context) == Advice.FORWARD
        assert backend.images[0] is not None
        assert backend.images[0][:4] == b"\x89PNG"

    def test_llmproxy_backend_rejects_pixel_mode(self):
        with pytest.raises(ValueError):
            LLMProxyBackend().generate("prompt", b"not-a-png")


class TestCachedAdvisor:
    def test_second_identical_context_is_a_hit(self, tmp_path):
        stub = _StubAdvisor(Advice.TURN_RIGHT)
        cached = CachedAdvisor(stub, tmp_path / "cache.json")
        context = _context(gesture=Gesture.GO)

        assert cached.advise(context) == Advice.TURN_RIGHT
        assert cached.advise(context) == Advice.TURN_RIGHT
        assert stub.calls == 1
        assert cached._cache_hits == 1
        assert cached._call_count == 2

    def test_gesture_changes_the_key(self, tmp_path):
        assert (context_cache_key(_context(gesture=Gesture.STOP))
                != context_cache_key(_context(gesture=Gesture.GO)))

    def test_file_roundtrip(self, tmp_path):
        path = tmp_path / "cache.json"
        CachedAdvisor(_StubAdvisor(Advice.TURN_LEFT), path).advise(_context())

        stub = _StubAdvisor(Advice.WAIT)
        reloaded = CachedAdvisor(stub, path)
        assert len(reloaded) == 1
        assert reloaded.advise(_context()) == Advice.TURN_LEFT
        assert stub.calls == 0

    def test_read_only_raises_on_miss(self, tmp_path):
        stub = _StubAdvisor()
        cached = CachedAdvisor(stub, tmp_path / "empty.json", read_only=True)
        with pytest.raises(KeyError):
            cached.advise(_context())
        assert stub.calls == 0
