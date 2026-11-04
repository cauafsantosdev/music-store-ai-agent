"""Separately opted-in live evaluations and offline tests of their safety envelope."""

import json
import os

import httpx
import pytest
from sqlalchemy import Engine

from music_store_agent.agent.memory import ConversationStore
from music_store_agent.evaluation.live import run_live


@pytest.mark.live_eval
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROQ_EVAL") != "1",
    reason="Separate live evaluation opt-in required.",
)
def test_live_agent_evaluation(
    demo_engine: Engine, state_store: ConversationStore, request: pytest.FixtureRequest
) -> None:
    """Measure real model behavior only on verified, disposable synthetic schemas.

    Args:
        demo_engine: Fingerprint-verified operational test schema.
        state_store: Isolated native checkpoints, never developer conversation state.
        request: Test metadata receiving only the sanitized evaluation summary.
    """
    if not os.getenv("GROQ_API_KEY", "").strip():
        pytest.skip("Explicit credentials are required for live evaluation.")
    report = run_live(demo_engine, state_store)
    request.node.user_properties.append(("live_evaluation", report))
    if (
        report.get("failed")
        or report.get("provider_failures")
        or report.get("evaluation_errors")
        or report.get("not_run")
    ):
        pytest.fail(
            "Live evaluation contains failed, provider-failed or incomplete cases.",
            pytrace=False,
        )


@pytest.mark.parametrize("status", [401, 402, 429])
def test_live_evaluation_stops_immediately_on_provider_rejection(
    demo_engine: Engine,
    state_store: ConversationStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    """Exercise actual SDK failures offline and verify one counted wire attempt only.

    Args:
        demo_engine: Verified isolated data.
        state_store: Disposable native checkpoint storage.
        monkeypatch: Scoped credential, flag and synthetic provider transport.
        caplog: Captured logs checked for raw diagnostic exclusion.
        status: Authentication, quota or rate-limit rejection.
    """
    monkeypatch.setenv("RUN_LIVE_GROQ_EVAL", "1")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-eval-key-canary")
    attempts: list[bool] = []

    def rejected(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Return synthetic SDK rejection data without an outbound network request.

        Args:
            transport: SDK transport.
            request: Synthetic chat request, never logged.

        Returns:
            A controlled response handled by the actual Groq SDK.
        """
        attempts.append(True)
        return httpx.Response(
            status,
            request=request,
            json={"error": {"message": "private-eval-provider-canary"}},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", rejected)
    report = run_live(demo_engine, state_store)
    assert attempts == [True]
    assert report["outbound_model_requests"] == report["provider_failures"] == 1
    assert report["conversations"] == report["evaluated_turns"] == 1
    assert report["passed"] == report["failed"] == 0 and report["not_run"] == 4
    assert (
        report["cases"][0]["reason"]
        == {401: "authentication", 402: "quota_exhausted", 429: "rate_limit"}[status]
    )
    assert "private-eval-provider-canary" not in json.dumps(report) + caplog.text
    assert "synthetic-eval-key-canary" not in json.dumps(report) + caplog.text
    assert "example.test" not in json.dumps(report)


def test_live_evaluation_enforces_wire_budget_before_dispatch(
    demo_engine: Engine,
    state_store: ConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop a real graph and SDK cycle before an over-budget transport attempt.

    Args:
        demo_engine: Verified isolated data.
        state_store: Disposable native checkpoint storage.
        monkeypatch: Controlled looping SDK responses and reduced safety budget.
    """
    monkeypatch.setenv("RUN_LIVE_GROQ_EVAL", "1")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-eval-key-canary")
    monkeypatch.setattr("music_store_agent.evaluation.live.MAX_REQUESTS", 2)
    attempts: list[bool] = []

    def looping(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Return repeated native tool calls through actual ChatGroq serialization.

        Args:
            transport: SDK transport.
            request: Synthetic chat request.

        Returns:
            A controlled function call, without a network attempt.
        """
        attempts.append(True)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"loop-{len(attempts)}",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"wire-{len(attempts)}",
                                    "type": "function",
                                    "function": {
                                        "name": "get_product_details",
                                        "arguments": '{"product_id":101}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", looping)
    report = run_live(demo_engine, state_store)
    assert len(attempts) == report["outbound_model_requests"] == 2
    assert report["failed"] == 1 and report["provider_failures"] == 0
    assert report["cases"][0]["reason"] == "execution_limit"
