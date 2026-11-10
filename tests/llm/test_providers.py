
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import base64
from src.llm.client import LLMError
from src.llm.providers import GeminiClient, make_client
from src.utils.config import ModelConfig


def make_fake_sdk(response_text="Hello from Gemini"):
    fake = Mock()
    fake.interactions.create.return_value = SimpleNamespace(
        output_text=response_text
    )
    return fake


def test_gemini_returns_generated_text():
    sdk = make_fake_sdk()
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
    )

    result = client.complete("Be helpful", "Say hello")

    assert result == "Hello from Gemini"
    sdk.interactions.create.assert_called_once_with(
        model="gemini-3.5-flash-lite",
        input="Be helpful\n\nSay hello",
    )


def test_gemini_json_mode_requests_json():
    sdk = make_fake_sdk('{"approved": true}')
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
    )

    result = client.complete(
        "Return structured data",
        "Is this approved?",
        json_mode=True,
    )

    assert result == '{"approved": true}'
    args = sdk.interactions.create.call_args.kwargs
    assert "Return only valid JSON" in args["input"]


def test_gemini_rejects_empty_response():
    sdk = make_fake_sdk("")
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
    )

    with pytest.raises(LLMError, match="empty response"):
        client.complete("Be helpful", "Say hello")


def test_gemini_retries_server_error():
    sdk = make_fake_sdk()
    sdk.interactions.create.side_effect = [
        RuntimeError("503 service unavailable"),
        SimpleNamespace(output_text="Recovered"),
    ]

    sleeps = []
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
        max_retries=1,
        sleep=sleeps.append,
    )

    assert client.complete("System", "User") == "Recovered"
    assert sdk.interactions.create.call_count == 2
    assert sleeps == [1]


def test_gemini_raises_after_retry_limit():
    sdk = make_fake_sdk()
    sdk.interactions.create.side_effect = RuntimeError(
        "503 service unavailable"
    )

    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
        max_retries=1,
        sleep=lambda _: None,
    )

    with pytest.raises(LLMError, match="Gemini request failed"):
        client.complete("System", "User")

    assert sdk.interactions.create.call_count == 2


def test_make_client_selects_gemini(monkeypatch):
    monkeypatch.setattr(
        "src.llm.providers.get_secret",
        lambda name: "test-key",
    )

    config = ModelConfig(
        provider="gemini",
        model="gemini-3.5-flash-lite",
        temperature=0.2,
    )

    client = make_client(config)

    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-3.5-flash-lite"


def test_make_client_rejects_unknown_provider():
    config = ModelConfig(
        provider="unknown",
        model="some-model",
        temperature=0.2,
    )

    with pytest.raises(ValueError, match="Unknown provider"):
        make_client(config)


def test_gemini_describes_image():
    sdk = make_fake_sdk("A bar chart comparing model accuracy.")
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
    )

    image = b"fake-png-image"

    result = client.describe_image(
        system="You are a scientific research assistant.",
        user="Describe this chart.",
        image_bytes=image,
        mime_type="image/png",
    )

    assert result == "A bar chart comparing model accuracy."

    args = sdk.interactions.create.call_args.kwargs
    assert args["model"] == "gemini-3.5-flash-lite"
    assert args["input"][0]["type"] == "text"
    assert args["input"][1]["type"] == "image"
    assert args["input"][1]["data"] == base64.b64encode(image).decode("utf-8")
    assert args["input"][1]["mime_type"] == "image/png"


def test_gemini_image_rejects_empty_response():
    sdk = make_fake_sdk("")
    client = GeminiClient(
        api_key="test-key",
        model="gemini-3.5-flash-lite",
        client=sdk,
    )

    with pytest.raises(LLMError, match="empty response"):
        client.describe_image(
            system="Describe images.",
            user="Describe this figure.",
            image_bytes=b"fake-image",
            mime_type="image/png",
        )