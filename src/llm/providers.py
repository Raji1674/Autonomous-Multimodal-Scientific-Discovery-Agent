
"""Production LLM providers using official Gemini SDK and compatible APIs."""
from __future__ import annotations

import base64
import json
import time
from typing import Callable

import requests
from google import genai

from src.llm.client import LLMError
from src.utils.config import ModelConfig, get_secret


PROVIDERS = {
    "groq": (
        "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1/chat/completions",
        "OPENROUTER_API_KEY",
    ),
}


class GeminiClient:
    """Gemini client using Google's official GenAI Interactions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        client=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._sleep = sleep
        self.client = client or genai.Client(api_key=api_key)
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
    ) -> str:
        prompt = f"{system}\n\n{user}" if system else user

        if json_mode:
            prompt += "\n\nReturn only valid JSON. Do not use Markdown fences."

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                )

                output = response.output_text
                if not output:
                    raise LLMError("Gemini returned an empty response")

                self.calls.append({
                    "model": self.model,
                    "usage": {},
                    "cost": 0.0,
                })
                return output

            except LLMError:
                raise
            except Exception as exc:
                status = getattr(exc, "code", None)
                message = str(exc).lower()

                retryable = (
                    status == 429
                    or status == 500
                    or status == 502
                    or status == 503
                    or status == 504
                    or "429" in message
                    or "500" in message
                    or "502" in message
                    or "503" in message
                    or "504" in message
                    or "timeout" in message
                )

                if not retryable or attempt == self.max_retries:
                    raise LLMError(
                        f"Gemini request failed: {type(exc).__name__}: {exc}"
                    ) from exc

                self._sleep(2 ** attempt)

        raise LLMError("Gemini request failed after retries")

    def describe_image(
        self,
        system: str,
        user: str,
        *,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """Interpret a scientific image using Gemini's vision capability."""
        if not image_bytes:
            raise LLMError("Image data cannot be empty")

        if mime_type not in {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }:
            raise LLMError(f"Unsupported image MIME type: {mime_type}")

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = f"{system}\n\n{user}" if system else user

        image_input = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": encoded_image,
                "mime_type": mime_type,
            },
        ]

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.interactions.create(
                    model=self.model,
                    input=image_input,
                )

                output = response.output_text
                if not output:
                    raise LLMError("Gemini returned an empty response")

                self.calls.append({
                    "model": self.model,
                    "usage": {},
                    "cost": 0.0,
                })
                return output

            except LLMError:
                raise
            except Exception as exc:
                message = str(exc).lower()
                status = getattr(exc, "code", None)

                retryable = (
                    status in (429, 500, 502, 503, 504)
                    or any(
                        code in message
                        for code in ("429", "500", "502", "503", "504", "timeout")
                    )
                )

                if not retryable or attempt == self.max_retries:
                    raise LLMError(
                        f"Gemini image request failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                self._sleep(2 ** attempt)

        raise LLMError("Gemini image request failed after retries")


class OpenAICompatClient:
    """OpenAI-compatible client for Groq and OpenRouter."""

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
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
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

        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.url,
                    json=body,
                    headers=headers,
                    timeout=60,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"LLM network request failed: {exc}") from exc
                self._sleep(2 ** attempt)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(
                        f"LLM request failed after retries (HTTP {resp.status_code})"
                    )
                self._sleep(2 ** attempt)
                continue

            if resp.status_code >= 400:
                raise LLMError(
                    f"LLM request rejected (HTTP {resp.status_code})"
                )

            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError("Invalid response from LLM provider") from exc

            self.calls.append({
                "model": self.model,
                "usage": data.get("usage", {}),
                "cost": 0.0,
            })
            return content

        raise LLMError("LLM request failed")


def make_client(cfg: ModelConfig) -> GeminiClient | OpenAICompatClient:
    """Build a configured provider client using environment API keys."""

    if cfg.model == "set-me":
        raise ValueError(
            "Set a real model name in configs/models.yaml"
        )

    if cfg.provider == "gemini":
        return GeminiClient(
            api_key=get_secret("GEMINI_API_KEY"),
            model=cfg.model,
            temperature=cfg.temperature,
        )

    if cfg.provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {cfg.provider!r}; "
            f"choose one of ['gemini', {', '.join(PROVIDERS)}]"
        )

    url, key_name = PROVIDERS[cfg.provider]

    return OpenAICompatClient(
        url,
        get_secret(key_name),
        cfg.model,
        cfg.temperature,
    )
