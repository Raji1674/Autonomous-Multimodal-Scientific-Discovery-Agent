"""LLM interface, a scripted FakeLLM for tests, and a JSON-repair helper."""
from __future__ import annotations

import json
import re
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when an LLM call fails permanently."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class FakeLLM:
    """Returns queued responses in order and records every call (no network)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self._responses:
            raise LLMError("FakeLLM has no queued responses left")
        return self._responses.pop(0)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def complete_json(client: LLMClient, system: str, user: str, retries: int = 2) -> dict | list:
    """Call the LLM and parse JSON, re-asking up to ``retries`` times on bad JSON."""
    prompt = user
    last_err = ""
    for _ in range(retries + 1):
        raw = client.complete(system, prompt, json_mode=True)
        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            last_err = str(exc)
            prompt = (f"{user}\n\nYour previous reply was not valid JSON ({last_err}). "
                      "Reply with ONLY valid JSON, no prose, no code fences.")
    raise LLMError(f"No valid JSON after {retries + 1} attempts: {last_err}")
