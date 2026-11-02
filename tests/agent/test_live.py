"""An explicitly opted-in GroqCloud smoke test using only confirmed synthetic data."""

import asyncio
import json
import logging
import os
import time
from typing import NoReturn

import httpx
import pytest
from groq import RateLimitError
from groq.resources.chat.completions import Completions
from langsmith import tracing_context
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import provider_failure
from music_store_agent.agent.factory import create_live_model, create_store_agent


def _exception_categories(error: BaseException) -> list[str]:
    """Collect exception type identifiers without messages or request/response data.

    Args:
        error: Provider or transport failure, possibly wrapped by the integration.

    Returns:
        At most four safe module-qualified exception class names.
    """
    categories: list[str] = []
    current: BaseException | None = error
    for _ in range(4):
        if current is None:
            break
        categories.append(f"{type(current).__module__}.{type(current).__name__}")
        current = current.__cause__ or current.__context__
    return categories


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROQ") != "1", reason="Explicit live GroqCloud opt-in required."
)
def test_live_groq_function_calling(
    demo_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gate real tool calling on a minimal response, using at most three API calls.

    Args:
        demo_engine: Freshly initialized and fingerprint-verified synthetic dataset.
        monkeypatch: Scoped transport instrumentation counting outbound attempts only.
        caplog: Suppressed provider logs, which could contain raw diagnostic payloads.
    """
    if not os.getenv("GROQ_API_KEY", "").strip():
        pytest.skip("GROQ_API_KEY is not explicitly configured.")
    settings = AgentSettings(
        ai_max_model_calls=2, ai_max_tool_calls=1, ai_recursion_limit=16
    )
    verify_demo(demo_engine)
    model = create_live_model(settings)
    # Instrument the locked SDK's transport without inspecting URLs, keys, or bodies.
    transport = model.client._client._client
    original_send = transport.send
    calls = 0

    def counted_send(*args: object, **kwargs: object) -> object:
        """Count real transport attempts without inspecting submitted data.

        Args:
            args: Uninspected positional transport arguments.
            kwargs: Uninspected keyword transport arguments.

        Returns:
            The unchanged transport response.

        Raises:
            Exception: The original transport failure, handled without logging it.
        """
        nonlocal calls
        calls += 1
        if calls > 3:
            pytest.fail("Live API call budget exceeded.", pytrace=False)
        return original_send(*args, **kwargs)

    monkeypatch.setattr(transport, "send", counted_send)
    started = time.monotonic()
    try:
        with caplog.at_level(logging.CRITICAL), tracing_context(enabled=False):
            failure_report: dict[str, object] | None = None
            basic = None
            try:
                basic = model.invoke(
                    "Responda apenas OK. Mensagem sintética de diagnóstico."
                )
            except Exception as error:
                # Deliberately omit exception text, objects, and payloads from output.
                failure_report = {
                    "stage": "minimal_response",
                    "code": provider_failure(error).code.value,
                    "exception_categories": _exception_categories(error),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "external_api_calls": calls,
                    "tool_calling": "not_attempted",
                }
            # Raise only after leaving the provider exception handler. Pytest can
            # otherwise render the original exception chain even with pytrace=False.
            if failure_report is not None:
                pytest.fail(json.dumps(failure_report), pytrace=False)
            if basic is None or not basic.text.strip() or basic.tool_calls:
                pytest.fail("Minimal text response validation failed.", pytrace=False)
            print(
                json.dumps(
                    {
                        "stage": "minimal_response",
                        "code": "ok",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "external_api_calls": calls,
                    }
                )
            )
            conversation = create_store_agent(
                demo_engine, settings=settings, model=model
            ).new_conversation()
            result = conversation.send(
                "Use get_product_details para consultar o produto fictício 101 "
                "e informe sua disponibilidade em português."
            )
        report = {
            "stage": "tool_calling",
            "code": result.code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "external_api_calls": calls,
            "tool_calls": result.tool_calls,
        }
        if result.code != "ok" or not result.text.strip() or result.tool_calls != 1:
            pytest.fail(json.dumps(report), pytrace=False)
        print(json.dumps(report))
    finally:
        model.client._client.close()
        asyncio.run(model.async_client._client.close())


def test_live_failure_report_cannot_chain_raw_provider_error(
    demo_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise live-test failure reporting offline without leaking an exception chain.

    Args:
        demo_engine: Verified synthetic PostgreSQL dataset.
        monkeypatch: Scoped fake credential and SDK failure, with no external calls.
        caplog: Captured diagnostic logs.
        capsys: Captured terminal streams.
    """

    def rejected(*args: object, **kwargs: object) -> NoReturn:
        """Raise a synthetic quota error before any SDK transport call.

        Args:
            args: Uninspected SDK positional arguments.
            kwargs: Uninspected SDK keyword arguments.

        Raises:
            RateLimitError: Always, with synthetic details that must remain private.
        """
        raise RateLimitError(
            "private-response-body-marker",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://api.groq.com")
            ),
            body={"error": {"message": "private-response-body-marker"}},
        )

    monkeypatch.setenv("GROQ_API_KEY", "offline-only-not-real")
    monkeypatch.setattr(Completions, "create", rejected)
    with pytest.raises(pytest.fail.Exception) as raised:
        test_live_groq_function_calling(demo_engine, monkeypatch, caplog)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    report = json.loads(str(raised.value))
    assert report["code"] == "rate_limit"
    assert report["external_api_calls"] == 0
    assert report["tool_calling"] == "not_attempted"
    assert "private-response-body-marker" not in str(raised.value)
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
