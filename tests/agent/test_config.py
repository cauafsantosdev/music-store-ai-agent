"""GroqCloud configuration and real SDK serialization without external requests."""

import asyncio
import importlib
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr, ValidationError

from music_store_agent.agent.config import AgentSettings, load_settings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_live_model


def test_default_model_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the requested Groq model with conservative execution defaults.

    Args:
        monkeypatch: Scoped removal of optional model environment overrides.
    """
    for name in ("AI_MODEL", "AI_PROVIDER_TIMEOUT_SECONDS", "AI_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    settings = AgentSettings(groq_api_key=None)
    assert settings.ai_model == "openai/gpt-oss-120b"
    assert settings.ai_provider_timeout_seconds == 30
    assert settings.ai_max_output_tokens == 2048


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_or_blank_keys_are_rejected(key: str | None) -> None:
    """Reject unavailable credentials without echoing input or making a request.

    Args:
        key: Missing or blank explicit credential.
    """
    with pytest.raises(AgentFailure) as raised:
        create_live_model(AgentSettings(groq_api_key=key))
    assert raised.value.code == ErrorCode.MISSING_API_KEY
    assert "GROQ_API_KEY" in str(raised.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ai_model", "another-model"),
        ("ai_max_model_calls", 0),
        ("ai_max_tool_calls", 1000),
        ("ai_max_output_tokens", 65000),
        ("ai_demo_schema", "public"),
        ("ai_provider_timeout_seconds", 0),
        ("ai_provider_timeout_seconds", 61),
    ],
)
def test_invalid_settings_are_rejected(field: str, value: object) -> None:
    """Reject other models, unsafe namespaces, and unbounded execution settings.

    Args:
        field: Configuration field under test.
        value: Invalid explicit value.
    """
    with pytest.raises(ValidationError):
        AgentSettings(**{field: value})


def test_missing_key_fails_only_when_creating_live_model() -> None:
    """Keep package imports and settings usable without credentials or requests."""
    settings = AgentSettings(groq_api_key=None)
    for module in ("music_store_agent.agent", "music_store_agent.agent.factory"):
        importlib.import_module(module)
    with pytest.raises(AgentFailure) as raised:
        create_live_model(settings)
    assert raised.value.code == ErrorCode.MISSING_API_KEY


def test_invalid_environment_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid echoing secret or raw configuration values in startup errors.

    Args:
        monkeypatch: Scoped environment settings.
    """
    monkeypatch.setenv("AI_MODEL", "private-marker")
    with pytest.raises(AgentFailure) as raised:
        load_settings()
    assert raised.value.code == ErrorCode.CONFIGURATION
    assert "private-marker" not in str(raised.value)


def test_live_client_request_configuration_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real Groq request serialization using an offline HTTP response.

    Args:
        monkeypatch: Scoped HTTP transport response, preventing external requests.
    """
    secret = "offline-only-key-not-real"
    settings = AgentSettings(groq_api_key=SecretStr(secret))
    model = create_live_model(settings)
    requests: list[dict[str, object]] = []

    def respond(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Inspect synthetic wire settings and return a minimal supported response.

        Args:
            self: Actual SDK HTTP transport, with network execution replaced.
            request: Serialized SDK request containing only synthetic test values.

        Returns:
            A successful Groq-compatible completion without a network connection.
        """
        assert request.url.host == "api.groq.com"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == settings.ai_model
        assert payload["max_completion_tokens"] == settings.ai_max_output_tokens
        assert "max_tokens" not in payload
        assert payload["service_tier"] == "on_demand"
        assert "parallel_tool_calls" not in payload
        assert not {"top_k", "frequency_penalty", "presence_penalty"} & payload.keys()
        assert not payload.get("reasoning_format")
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "offline-response",
                "object": "chat.completion",
                "created": 0,
                "model": settings.ai_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    try:
        assert isinstance(model, ChatGroq)
        assert model.model_name == "openai/gpt-oss-120b"
        assert model.max_retries == model.client._client.max_retries == 0
        assert model.request_timeout == settings.ai_provider_timeout_seconds
        assert model.client._client.timeout == settings.ai_provider_timeout_seconds
        assert secret not in repr(settings)
        assert "groq_api_key" not in settings.model_dump()
        assert (
            model.invoke([HumanMessage(content="Synthetic diagnostic.")]).text == "OK"
        )
        assert len(requests) == 1
    finally:
        model.client._client.close()
        asyncio.run(model.async_client._client.close())
