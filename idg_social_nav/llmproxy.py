"""Minimal client for the Tufts LLM proxy (text generation only).

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class ClientConfig:
    endpoint: str
    api_key: str
    timeout: float = 118.0  

    @staticmethod
    def from_env() -> ClientConfig:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
        endpoint = os.getenv("LLMPROXY_ENDPOINT")
        api_key = os.getenv("LLMPROXY_API_KEY")
        if not endpoint or not api_key:
            raise ValueError(
                "LLMProxy configuration error:\n"
                "Missing LLMPROXY_ENDPOINT or LLMPROXY_API_KEY.\n\n"
                "Make sure your .env file is in the SAME DIRECTORY where "
                "you run python (see .env.example)."
            )
        return ClientConfig(endpoint=endpoint, api_key=api_key)


def _build_session() -> requests.Session:
    """Session with retries and connection pooling."""
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class LLMProxy:
    def __init__(self) -> None:
        self.config = ClientConfig.from_env()
        self.session = _build_session()

    def _post_json(self, request_type: str, payload: dict[str, Any]) -> dict:
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        try:
            resp = self.session.post(
                self.config.endpoint,
                headers={
                    "x-api-key": self.config.api_key,
                    "request_type": request_type,
                },
                json=clean_payload,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {e}", "status_code": None}

        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return {"error": "Invalid JSON in response",
                        "status_code": resp.status_code}
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        return {"error": f"HTTP {resp.status_code}: {detail}",
                "status_code": resp.status_code}

    def generate(
        self,
        model: str,
        system: str,
        query: str,
        temperature: float | None = None,
        lastk: int | None = None,
        session_id: str | None = "GenericSession",
        rag_threshold: float | None = 0.5,
        rag_usage: bool | None = False,
        rag_k: int | None = 5,
    ) -> dict:
        """Text generation; the server returns the text under "result"."""
        return self._post_json("call", {
            "model": model,
            "system": system,
            "query": query,
            "temperature": temperature,
            "lastk": lastk,
            "session_id": session_id,
            "rag_threshold": rag_threshold,
            "rag_usage": rag_usage,
            "rag_k": rag_k,
        })
