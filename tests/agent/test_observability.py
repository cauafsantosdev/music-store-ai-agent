"""Privacy-safe request, committed-turn and streaming delivery observability."""

import asyncio
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import Engine
from starlette.types import Message, Receive, Scope, Send
from test_api import new_session

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api.main import create_app
from music_store_agent.api.middleware import MAX_BODY_BYTES, RequestGuard
from music_store_agent.observability import (
    FIELDS,
    LOGGER,
    REQUEST_ID,
    SafeJSONFormatter,
    emit,
)


@pytest.fixture
def telemetry(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the dedicated logger without changing its production propagation policy.

    Args:
        caplog: Pytest's in-memory handler, without a persistent log file.

    Yields:
        Captured records containing only explicit scalar metrics.
    """
    LOGGER.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        LOGGER.removeHandler(caplog.handler)


def records(telemetry: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    """Read the actual JSON formatter output without inspecting request payloads.

    Args:
        telemetry: Captured dedicated logger events.

    Returns:
        Structured allowlisted records.
    """
    formatter = SafeJSONFormatter()
    return [
        json.loads(formatter.format(record))
        for record in telemetry.records
        if record.name == LOGGER.name
    ]


def test_ids_are_generated_for_success_guard_errors_and_unknown_paths(
    telemetry: pytest.LogCaptureFixture,
) -> None:
    """Ignore arbitrary caller IDs and log templates instead of paths or query values.

    Args:
        telemetry: Correlation and request-level metrics capture.
    """
    identifiers: set[str] = set()
    with TestClient(create_app()) as client:
        for caller_id in (str(uuid4()), "sensitive-header-marker" * 100):
            responses = (
                client.get("/health", headers={"X-Request-ID": caller_id}),
                client.get("/sensitive-path-marker?email=sensitive-query-marker"),
                client.post("/health"),
                client.post(
                    "/api/v1/conversations",
                    content="x" * (MAX_BODY_BYTES + 1),
                    headers={"Authorization": "Bearer sensitive-capability-marker"},
                ),
            )
            assert [response.status_code for response in responses] == [
                200,
                404,
                405,
                413,
            ]
            for response in responses:
                identifier = response.headers["x-request-id"]
                assert len(identifier) == 36 and UUID(identifier).version == 4
                assert identifier != caller_id and identifier not in identifiers
                identifiers.add(identifier)
    events = records(telemetry)
    assert len(events) == 8 and {event["request_id"] for event in events} == identifiers
    assert all(
        event["duration_ms"] >= 0 and event["delivery"] == "completed"
        for event in events
    )
    assert all(set(event) <= FIELDS | {"timestamp", "level"} for event in events)
    assert "sensitive-" not in json.dumps(events)
    assert REQUEST_ID.get() is None


def test_worker_logs_only_committed_tool_count_without_sensitive_values(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    telemetry: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Correlate a real tool turn and verify success is logged after durable commit.

    Args:
        demo_engine: Verified disposable operational data.
        state_store: Transaction-bound PostgresSaver pool.
        offline_settings: Explicitly credential-free inference.
        telemetry: Captured safe events.
        monkeypatch: Read-only observation at the success logging boundary.
    """
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="sensitive-intermediate-marker",
                tool_calls=[
                    {
                        "name": "get_product_pricing",
                        "args": {"product_id": 101, "payment_method": "pix"},
                        "id": "sensitive-tool-id-marker",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="sensitive-answer-marker",
                additional_kwargs={"reasoning_content": "sensitive-reasoning-marker"},
            ),
        ]
    )
    application = create_app(
        settings=offline_settings, model=model, engine=demo_engine, store=state_store
    )
    committed: list[bool] = []
    with TestClient(application) as client:
        identifier, headers = new_session(client)

        def observe(event: str, **metrics: object) -> None:
            """Read through a separate connection when the committed event is emitted.

            Args:
                event: Fixed event name.
                metrics: Explicit safe scalar metrics.
            """
            if event == "agent_turn" and metrics["outcome"] == "ok":
                with state_store.transaction() as connection:
                    checkpoint = state_store.saver(connection).get_tuple(
                        {"configurable": {"thread_id": identifier}}
                    )
                    assert checkpoint is not None
                    assert (
                        checkpoint.checkpoint["channel_values"]["messages"][-1].text
                        == "sensitive-answer-marker"
                    )
                    committed.append(True)
            emit(event, **metrics)

        monkeypatch.setattr("music_store_agent.agent.service.emit", observe)
        response = client.post(
            f"/api/v1/conversations/{identifier}/stream",
            headers=headers,
            json={"message": "sensitive-user-message-marker"},
        )
        assert response.status_code == 200 and committed == [True]
        reference = response.headers["x-request-id"]
        events = [
            event for event in records(telemetry) if event["request_id"] == reference
        ]
        assert [event["event"] for event in events] == [
            "agent_turn",
            "sse_delivery",
            "http_request",
        ]
        assert events[0]["outcome"] == events[1]["outcome"] == "ok"
        assert events[0]["tool_calls"] == events[1]["tool_calls"] == 1
        assert events[-1]["route"] == "/api/v1/conversations/{conversation_id}/stream"
        serialized = json.dumps(records(telemetry))
        for secret in (
            "sensitive-",
            identifier,
            headers["Authorization"],
            "quoted_price",
            "reasoning_content",
        ):
            assert secret not in serialized
        assert all(event["duration_ms"] >= 0 for event in events)
        assert len(model.seen) == 2


@pytest.mark.parametrize(
    "provider_status,endpoint,outcome,http_status",
    [
        (401, "stream", "authentication", 200),
        (429, "messages", "rate_limit", 429),
        (504, "stream", "provider_timeout", 200),
        (503, "stream", "provider_unavailable", 200),
    ],
)
def test_provider_outcomes_remain_distinct_from_http_stream_status(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    telemetry: pytest.LogCaptureFixture,
    provider_status: int,
    endpoint: str,
    outcome: str,
    http_status: int,
) -> None:
    """Measure actual provider classification without raw exception or turn retention.

    Args:
        demo_engine: Verified isolated data.
        state_store: Transactional checkpoint pool.
        offline_settings: Explicit controlled settings.
        telemetry: Safe record capture.
        provider_status: Controlled model failure status.
        endpoint: Completed-turn REST or SSE endpoint.
        outcome: Expected existing sanitized domain classification.
        http_status: Actual transport status, including SSE's already-sent 200.
    """
    model = ScriptedModel(
        responses=[AIMessage(content="unused")], failure_code=provider_status
    )
    application = create_app(
        settings=offline_settings, model=model, engine=demo_engine, store=state_store
    )
    with TestClient(application) as client:
        identifier, headers = new_session(client)
        response = client.post(
            f"/api/v1/conversations/{identifier}/{endpoint}",
            headers=headers,
            json={"message": "Consulta fictícia."},
        )
        assert response.status_code == http_status
        events = [
            event
            for event in records(telemetry)
            if event["request_id"] == response.headers["x-request-id"]
        ]
        assert events[0]["event"] == "agent_turn" and events[0]["outcome"] == outcome
        assert events[0]["tool_calls"] == 0 and events[-1]["status_code"] == http_status
        if endpoint == "stream":
            assert (
                events[1]["outcome"] == outcome and events[1]["delivery"] == "completed"
            )
            assert "event: message" not in response.text
        assert (
            client.get(f"/api/v1/conversations/{identifier}", headers=headers).json()[
                "messages"
            ]
            == []
        )
    assert "private-diagnostic-marker" not in json.dumps(records(telemetry))


def test_concurrent_conversations_keep_worker_request_context_isolated(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    telemetry: pytest.LogCaptureFixture,
) -> None:
    """Retain request-local correlation across simultaneous thread-offloaded turns.

    Args:
        demo_engine: Shared verified synthetic data.
        state_store: Real checkpoint pool with independent conversation locks.
        offline_settings: Credential-free settings.
        telemetry: Safe captured records.
    """
    model = ScriptedModel(responses=[AIMessage(content="Resposta fictícia.")])
    with TestClient(
        create_app(
            settings=offline_settings,
            model=model,
            engine=demo_engine,
            store=state_store,
        )
    ) as client:
        sessions = [new_session(client), new_session(client)]

        def send(session: tuple[str, dict[str, str]]) -> str:
            """Submit one independent turn without printing its capability.

            Args:
                session: Authorized conversation handle.

            Returns:
                The generated non-secret correlation identifier.
            """
            identifier, headers = session
            response = client.post(
                f"/api/v1/conversations/{identifier}/messages",
                headers=headers,
                json={"message": "Consulta fictícia independente."},
            )
            assert response.status_code == 200
            return response.headers["x-request-id"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            references = set(pool.map(send, sessions))
    turns = [event for event in records(telemetry) if event["event"] == "agent_turn"]
    assert len(references) == len(turns) == 2
    assert {event["request_id"] for event in turns} == references
    assert REQUEST_ID.get() is None


def test_incomplete_asgi_delivery_is_not_logged_as_completed(
    telemetry: pytest.LogCaptureFixture,
) -> None:
    """Keep HTTP 200 separate from interrupted delivery after response headers.

    Args:
        telemetry: Captured safe request metrics.
    """

    async def broken(scope: Scope, receive: Receive, send: Send) -> None:
        """Raise a synthetic failure after one partial response frame.

        Args:
            scope: Uninspected synthetic scope.
            receive: Unused receive callback.
            send: Captured response callback.

        Raises:
            RuntimeError: Always, after headers, without diagnostic logging.
        """
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"partial", "more_body": True}
        )
        raise RuntimeError("sensitive-programming-marker")

    async def receive() -> Message:
        """Return a synthetic disconnect without retaining any request data.

        Returns:
            An ASGI disconnect frame.
        """
        return {"type": "http.disconnect"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        """Capture response frames without network access.

        Args:
            message: Current ASGI frame.
        """
        sent.append(message)

    asyncio.run(
        RequestGuard(broken)(
            {"type": "http", "method": "GET", "path": "/unknown"}, receive, send
        )
    )
    event = records(telemetry)[0]
    assert event["status_code"] == 200 and event["delivery"] == "interrupted"
    assert "sensitive-programming-marker" not in json.dumps(event)
    assert len(sent) == 2 and REQUEST_ID.get() is None


def test_formatter_drops_arbitrary_payloads_and_logging_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep diagnostic text and unexpected logger errors outside business outcomes.

    Args:
        monkeypatch: Scoped failed logging backend.
    """
    record = logging.LogRecord(
        LOGGER.name,
        logging.INFO,
        "private-path-marker",
        1,
        "private-message-marker %s",
        ("private-argument-marker",),
        (RuntimeError, RuntimeError("private-exception-marker"), None),
    )
    record.safe_fields = {
        "event": "http_request",
        "duration_ms": 0,
        "Authorization": "private-token-marker",
        "prompt": "private-prompt-marker",
    }
    encoded = SafeJSONFormatter().format(record)
    assert "private-" not in encoded
    assert set(json.loads(encoded)) == {"event", "duration_ms", "timestamp", "level"}

    def unavailable(*args: object, **kwargs: object) -> None:
        """Represent a failed logging backend without altering application outcomes.

        Args:
            args: Uninspected logging inputs.
            kwargs: Uninspected extra fields.

        Raises:
            RuntimeError: Always, representing a failed telemetry sink.
        """
        raise RuntimeError("private-logger-marker")

    monkeypatch.setattr(LOGGER, "info", unavailable)
    emit("agent_turn", duration_ms=1, outcome="ok", tool_calls=1)
