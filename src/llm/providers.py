
"""Real LLM providers through OpenAI-compatible chat endpoints."""
from __future__ import annotations

import time
from typing import Callable
import base64
import mimetypes
import requests
from pathlib import Path
from src.llm.client import LLMError
from src.utils.config import ModelConfig, get_secret

PROVIDERS = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
    ),
    "groq": (
        "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
    ),
}


class OpenAICompatClient:
    """Chat-completions client with retry handling."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_retries: int = 4,
        session=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._sleep = sleep
        self.calls: list[dict] = []

    def complete(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(self.max_retries + 1):
            resp = self.session.post(
                self.url,
                json=body,
                headers=headers,
                timeout=60,
            )

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(
                        f"LLM request failed after retries "
                        f"(HTTP {resp.status_code})"
                    )
                retry_after = (
                    resp.headers.get("retry-after")
                    if resp.status_code == 429 else None
                )
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt
                except (TypeError, ValueError):
                    delay = 2 ** attempt
                self._sleep(delay)
                continue

            if resp.status_code >= 400:
                try:
                    error_data = resp.json()
                    error_message = error_data.get("error", {}).get(
                        "message", "No error details"
                    )
                except (ValueError, AttributeError):
                    error_message = resp.text[:1000] or "Empty response body" # Truncate to first 1000 chars

                raise LLMError(
                    f"LLM request rejected (HTTP {resp.status_code}): "
                    f"{error_message}"
                )

            data = resp.json()
            self.calls.append({
                "model": self.model,
                "usage": data.get("usage", {}),
                "cost": 0.0,
            })
            return data["choices"][0]["message"]["content"]

        raise LLMError("unreachable")
    
    def complete_with_images(
        self,
        system: str,
        user: str,
        image_paths: list[str],
        *,
        json_mode: bool = False,
    ) -> str:
        """Send text and images to a vision-capable OpenAI-compatible model."""

        if not image_paths:
            raise ValueError("At least one image is required")

        if len(image_paths) > 3:
            raise ValueError("Maximum 3 images per request")

        content = [{"type": "text", "text": user}]

        for image_path in image_paths:
            path = Path(image_path)

            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")

            mime_type, _ = mimetypes.guess_type(str(path))

            if mime_type not in ("image/png", "image/jpeg", "image/webp"):
                raise ValueError(f"Unsupported image format: {path}")

            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{encoded}"

            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(self.max_retries + 1):
            resp = self.session.post(
                self.url,
                json=body,
                headers=headers,
                timeout=60,
            )

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(
                        f"Vision request failed after retries "
                        f"(HTTP {resp.status_code})"
                    )

                retry_after = (
                    resp.headers.get("retry-after")
                    if resp.status_code == 429 else None
                )

                try:
                    delay = float(retry_after) if retry_after else 2**attempt
                except (TypeError, ValueError):
                    delay = 2**attempt

                self._sleep(delay)
                continue

            if resp.status_code >= 400:
                raise LLMError(
                    f"Vision request rejected (HTTP {resp.status_code}): "
                    f"{resp.text[:1000]}"
                )

            data = resp.json()
            self.calls.append({
                "model": self.model,
                "usage": data.get("usage", {}),
                "cost": 0.0,
            })

            return data["choices"][0]["message"]["content"]

        raise LLMError("Vision request failed")


def make_client(cfg: ModelConfig) -> OpenAICompatClient:
    """Build a client using the configured provider and API key."""
    if cfg.provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {cfg.provider!r}; "
            f"choose one of {sorted(PROVIDERS)}"
        )

    if cfg.model == "set-me":
        raise ValueError("Set a real model name in configs/models.yaml")

    url, key_name = PROVIDERS[cfg.provider]
    return OpenAICompatClient(
        url,
        get_secret(key_name),
        cfg.model,
        cfg.temperature,
    )
