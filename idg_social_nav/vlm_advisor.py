"""VLM advisor

The advisor sees the scene.
The geometry and locations are always represented as text.
The social cue is either shown as a rendered camera frame in pixel mode
or as an explicit gesture line in the symbolic mode.

The advisor answers with exactly two words from the fixed vocabulary
{left, straight, right} x {slow, stop, constant}, which are mapped onto
grid advice thru core.map_vlm_tokens_to_advice.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from idg_social_nav.core import (
    DIR_NAMES,
    DIR_OFFSET,
    Advice,
    Advisor,
    AdvisorContext,
    map_vlm_tokens_to_advice,
    nearest_gesture,
)
from idg_social_nav.paths import LOG_DIR

_HEADING_RE = re.compile(r"\b(left|straight|right)\b", re.IGNORECASE)
_SPEED_RE = re.compile(r"\b(slow|stop|constant)\b", re.IGNORECASE)

# fallback when the backend never produces valid tokens: straight slow -> WAIT
_FALLBACK_TOKENS = ("straight", "slow")

_SOCIAL_PREAMBLE = (
    "You are a social-navigation advisor for a mobile robot in a discrete grid world. "
    "Rows increase downward, columns increase rightward. "
    "The robot moves one cell at a time."
)

_SOCIAL_NORMS = (
    "Recommend the robot's next motion so it behaves in a socially acceptable manner: "
    "Do not enter a pedestrian's personal space, "
    "do not cut across or block a pedestrian's path, "
    "keep to the right when passing, "
    "respect any gesture the pedestrian makes "
    "(a raised palm means they assert right of way — clear their lane and wait, "
    "and a sweeping arm means they yield — proceed), "
    "and do not stop unnecessarily when the way is clear."
)

_ANSWER_FORMAT = (
    "Answer with EXACTLY two words: <heading> <speed>\n"
    "heading is one of {left, straight, right}, "
    "speed is one of {slow, stop, constant}."
)

_CORRECTION = (
    "Your previous answer was invalid. "
    "Reply with EXACTLY two words: a heading from {left, straight, right} "
    "followed by a speed from {slow, stop, constant}. Nothing else."
)


def parse_vlm_response(text: str) -> tuple[str, str] | None:
    """parse over the VLM's raw text response and extract the two-word answer.
    Returns (heading, speed) or None if the answer is invalid."""
    if not text:
        return None
    heading = _HEADING_RE.search(text)
    speed = _SPEED_RE.search(text)
    if heading is None or speed is None:
        return None
    return heading.group(1).lower(), speed.group(1).lower()


def _frame_to_png(frame: np.ndarray) -> bytes:
    import imageio.v3 as iio
    return iio.imwrite("<bytes>", np.asarray(frame, dtype=np.uint8), extension=".png")


def _load_api_key(env_var: str) -> str:
    """
    Load an API key from the environment or .env file.
    Raises ValueError if the key is not found.
    """
    key = os.getenv(env_var)
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        except ImportError:
            pass
        key = os.getenv(env_var)
    if not key:
        raise ValueError(
            f"Missing API key: set {env_var} in the environment or .env "
            "(see .env.example)")
    return key


# backends
class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, prompt_text: str, image_png_bytes: bytes | None = None) -> str:
        import requests
        key = _load_api_key(self.api_key_env)
        content: list[dict] = [{"type": "text", "text": prompt_text}]
        if image_png_bytes is not None:
            b64 = base64.standard_b64encode(image_png_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "low",
                },
            })
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 30,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicBackend:
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, prompt_text: str, image_png_bytes: bytes | None = None) -> str:
        import requests
        key = _load_api_key(self.api_key_env)
        content: list[dict] = []
        if image_png_bytes is not None:
            b64 = base64.standard_b64encode(image_png_bytes).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt_text})
        payload = {
            "model": self.model,
            "max_tokens": 30,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class LLMProxyBackend:
    """Vendored llmproxy backend. Text only: pixel mode raises ValueError."""

    def __init__(self, model: str = "us.anthropic.claude-3-haiku-20240307-v1:0"):
        self.model = model
        self._proxy = None
        self._call_count = 0

    def generate(self, prompt_text: str, image_png_bytes: bytes | None = None) -> str:
        if image_png_bytes is not None:
            raise ValueError("LLMProxyBackend is text-only; use mode='symbolic'.")
        if self._proxy is None:
            from idg_social_nav.llmproxy import LLMProxy
            self._proxy = LLMProxy()
        self._call_count += 1
        response = self._proxy.generate(
            model=self.model,
            system="You are a social-navigation advisor for a mobile robot.",
            query=prompt_text,
            temperature=0.0,
            session_id=f"vlm-advisor-{self._call_count}",
        )
        if not isinstance(response, dict) or "error" in response:
            raise RuntimeError(f"LLMProxy error: {response}")
        return response.get("result", "")



# Cache
def context_cache_key(context: AdvisorContext) -> str:
    """Return a string key for the context, designed for caching advice.
    The key is a pipe-separated string of:
    - scenario name
    - agent position (row, col)
    - agent direction (0-3)
    - visible pedestrians, each as (row, col)|facing|gesture, separated by semicolons.
    """
    peds = ";".join(
        f"{tuple(p.pos)}|{int(p.facing)}|{int(p.gesture)}"
        for p in context.pedestrians if p.visible
    )
    agent_pos = (int(context.agent_pos[0]), int(context.agent_pos[1]))
    return f"{context.scenario_name}|{agent_pos}|{int(context.agent_dir)}|" + peds


class _JsonCache:
    """str key -> int advice mapping persisted as a JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self.data: dict[str, int] = {}
        if self.path.exists():
            try:
                self.data = {
                    str(k): int(v)
                    for k, v in json.loads(self.path.read_text()).items()
                }
            except (OSError, ValueError):
                self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)


class CachedAdvisor(Advisor):
    """Wraps any advisor with a persistent JSON cache.

    read_only=True freezes the cache: a miss raises KeyError 
    instead of querying the base advisor.
    """

    def __init__(self, base: Advisor, cache_path, read_only: bool = False):
        self.base = base
        self.read_only = read_only
        self._store = _JsonCache(cache_path)
        self._call_count = 0
        self._cache_hits = 0

    @property
    def cache_path(self) -> Path:
        return self._store.path

    def __len__(self) -> int:
        return len(self._store.data)

    def advise(self, context: AdvisorContext) -> Advice:
        self._call_count += 1
        key = context_cache_key(context)
        if key in self._store.data:
            self._cache_hits += 1
            return Advice(self._store.data[key])
        if self.read_only:
            raise KeyError(f"Advice cache miss in read-only mode: {key}")
        advice = Advice(self.base.advise(context))
        self._store.data[key] = int(advice)
        self.save()
        return advice

    def save(self) -> None:
        self._store.save()

    def reset(self) -> None:
        self.base.reset()




# vlm advisor
def _in_agent_lane(agent_pos, agent_dir, ped_pos) -> bool:
    """Pedestrian sits on the line of cells straight ahead of the agent."""
    fr, fc = DIR_OFFSET[agent_dir]
    dr = ped_pos[0] - agent_pos[0]
    dc = ped_pos[1] - agent_pos[1]
    forward = dr * fr + dc * fc
    lateral = abs(dr * fc - dc * fr)
    return forward > 0 and lateral == 0


def _moving_toward(agent_pos, ped_pos, ped_facing) -> bool:
    """
    Returns True if the pedestrian is moving toward the agent.
    The pedestrian is moving toward the agent if the projection of the
    vector from the pedestrian to the agent onto the pedestrian's facing
    is positive
    """
    fr, fc = DIR_OFFSET[ped_facing]
    dr = agent_pos[0] - ped_pos[0]
    dc = agent_pos[1] - ped_pos[1]
    return (dr * fr + dc * fc) > 0


class VLMAdvisor(Advisor):
    """
    VLM advisor: queries a vision-language model for advice.
    The advisor sees the scene as text and optionally a camera image.
    The model is expected to answer with exactly two words from the fixed vocabulary
    {left, straight, right} x {slow, stop, constant}, which are mapped onto grid advice thru core.map_vlm_tokens_to_advice.
    """

    def __init__(self, backend, mode: str = "symbolic",
                 max_retries: int = 2, log_path=None):
        assert mode in ("symbolic", "pixel")
        self.backend = backend
        self.mode = mode
        self.max_retries = max_retries
        self._log_path = (Path(log_path) if log_path is not None
                          else LOG_DIR / "vlm_advisor.jsonl")
        self._call_count = 0


    def build_prompt(self, context: AdvisorContext) -> str:
        h, w = context.walls.shape
        ar, ac = int(context.agent_pos[0]), int(context.agent_pos[1])
        lines = [
            _SOCIAL_PREAMBLE,
            "",
            f"SCENE ({h}x{w} grid; the outer boundary cells are walls):",
            f"- Robot at row {ar}, col {ac}, facing {DIR_NAMES[context.agent_dir]}.",
            f"- Goal at row {context.goal_pos[0]}, col {context.goal_pos[1]}.",
        ]
        visible = [p for p in context.pedestrians if p.visible]
        for p in visible:
            pr, pc = p.pos
            dist = abs(pr - ar) + abs(pc - ac)
            lane = ("in the robot's lane"
                    if _in_agent_lane((ar, ac), context.agent_dir, p.pos)
                    else "not in the robot's lane")
            toward = ("moving toward the robot"
                      if _moving_toward((ar, ac), p.pos, p.facing)
                      else "not moving toward the robot")
            lines.append(
                f"- Pedestrian at row {pr}, col {pc}, facing {DIR_NAMES[p.facing]}, "
                f"Manhattan distance {dist}, {lane}, {toward}.")
        if not visible:
            lines.append("- No pedestrian is visible.")

        if self.mode == "symbolic":
            gesture = nearest_gesture(
                [(tuple(p.pos), p.gesture)
                 for p in context.pedestrians if p.visible],
                context.agent_pos)
            lines.append(f"Pedestrian gesture: {gesture.name.lower()}")
        else:
            lines.append(
                "- A camera image of the scene is attached. "
                "read the pedestrian's body language and any hand signal from the image.")

        lines += ["", _SOCIAL_NORMS, "", _ANSWER_FORMAT]
        return "\n".join(lines)

    # query the backend with retries and correction prompts
    def _query(self, prompt: str, image: bytes | None):
        """Returns (tokens | None, raw_response, retries, error)."""
        raw, error = "", None
        current = prompt
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.backend.generate(current, image)
                error = None
            except ValueError:
                raise  # configuration error (missing key, pixel misuse)
            except Exception as exc:  # backend or network failure
                raw, error = "", str(exc)
            tokens = parse_vlm_response(raw)
            if tokens is not None:
                return tokens, raw, attempt, error
            current = current + "\n\n" + _CORRECTION
        return None, raw, self.max_retries, error

    def _log(self, record: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def advise(self, context: AdvisorContext) -> Advice:
        self._call_count += 1
        prompt = self.build_prompt(context)
        image = None
        if self.mode == "pixel":
            if context.frame_provider is None:
                raise ValueError(
                    "pixel mode requires context.frame_provider")
            image = _frame_to_png(context.frame_provider())

        tokens, raw, retries, error = self._query(prompt, image)
        mapping_failure = tokens is None
        heading, speed = _FALLBACK_TOKENS if mapping_failure else tokens
        advice = map_vlm_tokens_to_advice(heading, speed)

        self._log({
            "mode": self.mode,
            "scenario": context.scenario_name,
            "step": int(context.step),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "raw_response": raw,
            "tokens": [heading, speed],
            "advice": int(advice),
            "retries": retries,
            "mapping_failure": mapping_failure,
            "error": error,
        })

        return advice
