"""Real LLM providers through OpenAI-compatible chat endpoints (free-tier friendly).

NOT VERIFIED AGAINST LIVE SERVICES in this repo's tests: endpoints and model names
must be checked against each provider's current docs before use.
"""
from __future__ import annotations

import time
from typing import Callable

import requests

from src.llm.client import LLMError
from src.utils.config import ModelConfig, get_secret

PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}


class OpenAICompatClient:
    """Chat-completions client with exponential backoff on rate limits (429) and 5xx."""

    def __init__(self, url: str, api_key: str, model: str, temperature: float = 0.2,
                 max_retries: int = 4, session=None, sleep: Callable[[float], None] = time.sleep):
        self.url, self.api_key, self.model = url, api_key, model
        self.temperature, self.max_retries = temperature, max_retries
        self.session = session or requests.Session()
        self._sleep = sleep
        self.calls: list[dict] = []  # token usage per call, for cost tracking

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            resp = self.session.post(self.url, json=body, headers=headers, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(f"LLM request failed after retries (HTTP {resp.status_code})")
                self._sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"LLM request rejected (HTTP {resp.status_code})")
            data = resp.json()
            self.calls.append({"model": self.model, "usage": data.get("usage", {}), "cost": 0.0})
            return data["choices"][0]["message"]["content"]
        raise LLMError("unreachable")


def make_client(cfg: ModelConfig) -> OpenAICompatClient:
    """Build a client from config; the API key comes from the environment."""
    if cfg.provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {cfg.provider!r}; choose one of {sorted(PROVIDERS)}")
    if cfg.model == "set-me":
        raise ValueError("Set a real model name in configs/models.yaml (check the provider's docs)")
    url, key_name = PROVIDERS[cfg.provider]
    return OpenAICompatClient(url, get_secret(key_name), cfg.model, cfg.temperature)
