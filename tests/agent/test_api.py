"""HTTP contracts over real isolated PostgreSQL state and controlled inference."""

import hashlib
import json
from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from conftest import ScriptedModel
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import Engine, update

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.agent.service import AgentReply, StoreAgent
from music_store_agent.api.main import create_app
from music_store_agent.api.middleware import MAX_BODY_BYTES
from music_store_agent.database.models import Product


@pytest.fixture
def api_model() -> ScriptedModel:
    """Provide distinct synthetic response IDs for repeated HTTP turns.

    Returns:
        A controlled native LangChain model without external generation.
    """
    return ScriptedModel(
        responses=[
            AIMessage(content="Atendimento fictício.", id=f"reply-{index}")
            for index in range(10)
        ]
    )


@pytest.fixture
def api_app(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    api_model: ScriptedModel,
) -> FastAPI:
    """Build the real API and agent service using isolated PostgreSQL dependencies.

    Args:
        demo_engine: Verified synthetic operational schema.
        state_store: Initialized disposable checkpoint schema.
        offline_settings: Provider-free configuration.
        api_model: Controlled model replacing only inference.

    Returns:
        The production application factory with explicit test dependencies.
    """
    return create_app(
        settings=offline_settings,
        model=api_model,
        engine=demo_engine,
        store=state_store,
    )


@pytest.fixture
def api_client(api_app: FastAPI) -> Iterator[TestClient]:
    """Run the real application lifespan around synchronous HTTP integration tests.

    Args:
        api_app: Application using real synthetic data and checkpoint storage.

    Yields:
        A client that closes application resources on exit.
    """
    with TestClient(api_app) as client:
        yield client


def new_session(client: TestClient) -> tuple[str, dict[str, str]]:
    """Create an actual HTTP session without logging its returned capability.

    Args:
        client: Application HTTP client.

    Returns:
        Conversation identifier and authorization headers.
    """
    response = client.post("/api/v1/conversations")
    assert response.status_code == 201
    data = response.json()
    return data["conversation_id"], {"Authorization": f"Bearer {data['access_token']}"}


def parse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse complete JSON SSE frames without interpreting model token fragments.

    Args:
        body: Complete synthetic SSE response.

    Returns:
        Event names paired with their JSON payloads.
    """
    events = []
    for frame in body.strip().split("\n\n"):
        name, data = frame.split("\n")
        assert name.startswith("event: ") and data.startswith("data: ")
        events.append((name[7:], json.loads(data[6:])))
    return events


def test_http_creation_uses_existing_capabilities_and_no_generation(
    api_client: TestClient,
    api_app: FastAPI,
    api_model: ScriptedModel,
    state_store: ConversationStore,
) -> None:
    """Return only the created UUID and capability with no-store cache controls.

    Args:
        api_client: Real HTTP client.
        api_app: Running application.
        api_model: Controlled inference spy.
        state_store: Real checkpoint metadata storage.
    """
    response = api_client.post("/api/v1/conversations", json={})
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    assert set(data) == {"conversation_id", "access_token"}
    identifier = UUID(data["conversation_id"])
    assert identifier.version == 4
    assert len(data["access_token"]) >= 43
    with state_store.transaction() as connection:
        row = connection.execute(
            "SELECT access_token_hash FROM conversations WHERE conversation_id = %s",
            (identifier,),
        ).fetchone()
    assert (
        bytes(row["access_token_hash"])
        == hashlib.sha256(data["access_token"].encode()).digest()
    )
    assert api_model.seen == []
    service = api_app.state.resources.service()
    assert api_app.state.resources.service() is service


def test_http_tool_execution_history_and_follow_up(
    api_client: TestClient,
    api_model: ScriptedModel,
) -> None:
    """Invoke real pricing and preserve native state while exposing display text only.

    Args:
        api_client: Real HTTP client.
        api_model: Controlled inference with actual PostgreSQL tool execution.
    """
    api_model.responses = [
        AIMessage(
            content="",
            additional_kwargs={"opaque_signature": "hidden-marker"},
            tool_calls=[
                {
                    "name": "get_product_pricing",
                    "args": {"product_id": 101, "payment_method": "pix"},
                    "id": "http-tool",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="Preço verificado: R$ 1.104,93.",
            additional_kwargs={"reasoning_content": "hidden-reasoning-marker"},
        ),
        AIMessage(content="O mesmo produto continua no contexto."),
    ]
    identifier, headers = new_session(api_client)
    target = f"/api/v1/conversations/{identifier}"
    reply = api_client.post(
        target + "/messages",
        headers=headers,
        json={"message": "Consulte o preço do produto fictício 101 com PIX."},
    )
    assert reply.status_code == 200
    assert reply.json() == {
        "text": "Preço verificado: R$ 1.104,93.",
        "code": "ok",
        "tool_calls": 1,
    }
    evidence = next(
        item for item in api_model.seen[-1] if isinstance(item, ToolMessage)
    )
    quote = json.loads(evidence.content)["data"]["quote"]
    assert quote["quoted_price"] == "1104.93"
    assert quote["discount_source"] == "promotion" and quote["pix_price"] is None
    history = api_client.get(target, headers=headers)
    assert history.status_code == 200
    assert [item["role"] for item in history.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert all(set(item) == {"role", "text"} for item in history.json()["messages"])
    assert (
        "hidden-marker" not in history.text
        and "hidden-reasoning-marker" not in history.text
    )
    assert "http-tool" not in history.text and "access_token" not in history.text
    assert (
        api_client.post(
            target + "/messages",
            headers=headers,
            json={"message": "E o mesmo produto?"},
        ).status_code
        == 200
    )
    assert (
        len([item for item in api_model.seen[-1] if isinstance(item, HumanMessage)])
        == 2
    )
    assert any(
        isinstance(item, AIMessage)
        and item.additional_kwargs.get("opaque_signature") == "hidden-marker"
        for item in api_model.seen[-1]
    )


@pytest.mark.parametrize("operation", ["history", "messages", "stream", "delete"])
def test_unknown_and_cross_session_capabilities_are_indistinguishable(
    api_client: TestClient,
    api_model: ScriptedModel,
    operation: str,
) -> None:
    """Keep every protected route indistinguishable for invalid and unknown access.

    Args:
        api_client: Real HTTP client.
        api_model: Provider call spy.
        operation: Protected route under test.
    """
    identifier, headers = new_session(api_client)
    other, other_headers = new_session(api_client)
    suffix = "" if operation in ("history", "delete") else f"/{operation}"
    method = (
        "GET"
        if operation == "history"
        else "DELETE"
        if operation == "delete"
        else "POST"
    )
    data = {"json": {"message": "Olá."}} if method == "POST" else {}
    results = []
    for selected, credentials in (
        (str(uuid4()), headers),
        (identifier, other_headers),
        (identifier, {"Authorization": "Bearer wrong-token"}),
    ):
        response = api_client.request(
            method,
            f"/api/v1/conversations/{selected}{suffix}",
            headers=credentials,
            **data,
        )
        assert response.status_code == 404
        assert response.headers["cache-control"] == "no-store"
        results.append(response.json())
    assert results[0] == results[1] == results[2]
    assert results[0]["error"]["code"] == "session_unverified"
    assert (
        api_client.get(f"/api/v1/conversations/{identifier}", headers=headers).json()[
            "messages"
        ]
        == []
    )
    assert (
        api_client.get(f"/api/v1/conversations/{other}", headers=other_headers).json()[
            "messages"
        ]
        == []
    )
    assert api_model.seen == []


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic private-marker", "Bearer", "Bearer one two", "Bearer invalid$token"],
)
def test_missing_or_malformed_bearer_headers_are_sanitized(
    api_client: TestClient,
    api_model: ScriptedModel,
    authorization: str | None,
) -> None:
    """Reject malformed authorization without echoing header values.

    Args:
        api_client: Real HTTP client.
        api_model: Provider call spy.
        authorization: Missing or malformed Authorization header.
    """
    identifier, _ = new_session(api_client)
    headers = {} if authorization is None else {"Authorization": authorization}
    response = api_client.get(f"/api/v1/conversations/{identifier}", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "private-marker" not in response.text
    assert api_model.seen == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": None},
        {"message": 10},
        {"message": ["override"]},
        {"message": "x" * 2001},
        {"message": "private-customer@example.org"},
        {"message": "+55 (67) 99999-9999"},
        {"message": "Olá.", "engine": "private-marker"},
    ],
)
@pytest.mark.parametrize("endpoint", ["messages", "stream"])
def test_invalid_messages_fail_before_generation(
    api_client: TestClient,
    api_model: ScriptedModel,
    payload: dict[str, object],
    endpoint: str,
) -> None:
    """Apply strict public validation and existing synthetic-input protections.

    Args:
        api_client: Real HTTP client.
        api_model: Provider call spy.
        payload: Invalid or non-demonstration input.
        endpoint: Ordinary or SSE message route.
    """
    identifier, headers = new_session(api_client)
    response = api_client.post(
        f"/api/v1/conversations/{identifier}/{endpoint}", headers=headers, json=payload
    )
    assert response.status_code == 422
    assert "private-marker" not in response.text
    assert "example.org" not in response.text and "99999" not in response.text
    assert "2001" not in response.text
    assert api_model.seen == []


def test_request_size_queries_uuid_and_json_are_rejected_safely(
    api_client: TestClient,
    api_model: ScriptedModel,
) -> None:
    """Reject unsafe transport shapes before Groq without reflecting input details.

    Args:
        api_client: Real HTTP client.
        api_model: Provider call spy.
    """
    identifier, headers = new_session(api_client)
    base = f"/api/v1/conversations/{identifier}"
    large = api_client.post(
        base + "/messages", headers=headers, content=b"x" * (MAX_BODY_BYTES + 1)
    )
    assert large.status_code == 413
    query = api_client.get(base + "?access_token=private-query-marker")
    assert query.status_code == 422 and "private-query-marker" not in query.text
    malformed = api_client.get(
        "/api/v1/conversations/private-uuid-marker", headers=headers
    )
    assert malformed.status_code == 422 and "private-uuid-marker" not in malformed.text
    broken_json = api_client.post(
        base + "/messages",
        headers={**headers, "Content-Type": "application/json"},
        content='{"message": "private-body-marker",',
    )
    assert (
        broken_json.status_code == 422 and "private-body-marker" not in broken_json.text
    )
    creation = api_client.post(
        "/api/v1/conversations", json={"settings": "private-setting-marker"}
    )
    assert creation.status_code == 422 and "private-setting-marker" not in creation.text
    assert api_model.seen == []


def test_valid_maximum_message_accepts_json_escaped_non_bmp_characters(
    api_client: TestClient,
    api_model: ScriptedModel,
) -> None:
    """Keep the byte limit compatible with the service's character-count contract.

    Args:
        api_client: Real HTTP client backed by isolated PostgreSQL state.
        api_model: Controlled inference spy.
    """
    identifier, headers = new_session(api_client)
    message = "\U0001f3b5" * 2000
    encoded = json.dumps({"message": message}, ensure_ascii=True)
    assert len(encoded.encode()) > 16_384
    response = api_client.post(
        f"/api/v1/conversations/{identifier}/messages",
        headers={**headers, "Content-Type": "application/json"},
        content=encoded,
    )
    assert response.status_code == 200
    assert api_model.seen[0][-1].content == message


@pytest.mark.parametrize(
    "code,status",
    [
        ("database", 503),
        ("provider_timeout", 504),
        ("provider_unavailable", 503),
        ("rate_limit", 429),
        ("quota_exhausted", 429),
        ("execution_limit", 429),
        ("configuration", 503),
        ("missing_api_key", 503),
        ("authentication", 503),
        ("state_uninitialized", 503),
        ("empty_response", 503),
        ("private-unknown-code", 500),
        ("not_found", 200),
        ("order_unverified", 200),
    ],
)
def test_non_ok_service_replies_are_explicit_http_outcomes(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    status: int,
) -> None:
    """Map service results without trusting arbitrary failure text or error codes.

    Args:
        api_client: Real HTTP client.
        monkeypatch: Scoped service-result boundary substitution.
        code: Controlled existing or unknown failure identifier.
        status: Expected HTTP response status.
    """
    identifier, headers = new_session(api_client)

    def failed(
        self: StoreAgent,
        conversation_id: UUID | str,
        access_token: object,
        message: str,
    ) -> AgentReply:
        """Return a controlled result without changing persistence or calling a model.

        Args:
            self: Actual application service.
            conversation_id: Requested session.
            access_token: Uninspected capability.
            message: Synthetic user input.

        Returns:
            A controlled non-ok result with diagnostic text that must be discarded.
        """
        return AgentReply(text="private-diagnostic-marker", code=code)

    monkeypatch.setattr(StoreAgent, "send_message", failed)
    response = api_client.post(
        f"/api/v1/conversations/{identifier}/messages",
        headers=headers,
        json={"message": "Olá."},
    )
    assert response.status_code == status
    assert "private-diagnostic-marker" not in response.text
    assert "private-unknown-code" not in response.text


def test_real_provider_failure_is_sanitized_and_history_rolls_back(
    api_client: TestClient,
    api_model: ScriptedModel,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate an actual graph failure while keeping failed turns out of history.

    Args:
        api_client: Real HTTP client.
        api_model: Controlled provider error from inside the actual graph.
        caplog: Captured logs.
        capsys: Captured output streams.
    """
    identifier, headers = new_session(api_client)
    api_model.failure_code = 429
    base = f"/api/v1/conversations/{identifier}"
    response = api_client.post(
        base + "/messages", headers=headers, json={"message": "Olá."}
    )
    assert response.status_code == 429
    assert api_client.get(base, headers=headers).json()["messages"] == []
    assert len(api_model.seen) == 1
    assert "private-diagnostic-marker" not in response.text + caplog.text
    output = capsys.readouterr()
    assert "private-diagnostic-marker" not in output.out + output.err


def test_sse_success_has_one_complete_reply_and_no_duplicate_turn(
    api_client: TestClient,
    api_model: ScriptedModel,
) -> None:
    """Deliver JSON status/message/done frames without fake model-token fragments.

    Args:
        api_client: Real HTTP client.
        api_model: Controlled inference spy.
    """
    identifier, headers = new_session(api_client)
    base = f"/api/v1/conversations/{identifier}"
    response = api_client.post(
        base + "/stream", headers=headers, json={"message": "Olá."}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["status", "message", "done"]
    assert events[0][1] == {"status": "processing"}
    assert events[1][1] == {
        "text": "Atendimento fictício.",
        "code": "ok",
        "tool_calls": 0,
    }
    assert events[2][1] == {"status": "completed"}
    assert len(api_model.seen) == 1
    assert len(api_client.get(base, headers=headers).json()["messages"]) == 2


def test_sse_failure_has_error_and_failed_completion(
    api_client: TestClient,
    api_model: ScriptedModel,
) -> None:
    """Keep post-header provider failures explicit rather than fabricated success.

    Args:
        api_client: Real HTTP client.
        api_model: Controlled provider failure inside the real graph.
    """
    identifier, headers = new_session(api_client)
    api_model.failure_code = 503
    response = api_client.post(
        f"/api/v1/conversations/{identifier}/stream",
        headers=headers,
        json={"message": "Olá."},
    )
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["status", "error", "done"]
    assert events[1][1]["error"]["code"] == "provider_unavailable"
    assert events[1][1]["status_code"] == 503
    assert events[2][1] == {"status": "failed"}
    assert "private-diagnostic-marker" not in response.text
    assert len(api_model.seen) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_programming_errors_are_sanitized_without_false_success(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return 500 or an SSE internal-error event without raw exception logging.

    Args:
        api_client: Real HTTP client.
        monkeypatch: Controlled unexpected service error.
        stream: Whether headers have already started through SSE.
        caplog: Captured server logs.
    """
    identifier, headers = new_session(api_client)

    def broken(
        self: StoreAgent,
        conversation_id: UUID | str,
        access_token: object,
        message: str,
    ) -> AgentReply:
        """Raise a diagnostic canary at the service boundary.

        Args:
            self: Actual application service.
            conversation_id: Target session identifier.
            access_token: Uninspected capability.
            message: Synthetic input.

        Raises:
            RuntimeError: Always, with a diagnostic value that must stay private.
        """
        raise RuntimeError("private-programming-error-marker")

    monkeypatch.setattr(StoreAgent, "send_message", broken)
    endpoint = "stream" if stream else "messages"
    response = api_client.post(
        f"/api/v1/conversations/{identifier}/{endpoint}",
        headers=headers,
        json={"message": "Olá."},
    )
    if stream:
        events = parse_events(response.text)
        assert events[1][0] == "error" and events[1][1]["status_code"] == 500
    else:
        assert response.status_code == 500
    assert "private-programming-error-marker" not in response.text + caplog.text


def test_authorized_deletion_removes_checkpoints_and_preserves_other_sessions(
    api_client: TestClient,
    state_store: ConversationStore,
) -> None:
    """Delete only authorized metadata and native checkpoint rows through HTTP.

    Args:
        api_client: Real HTTP client.
        state_store: Real checkpoint storage.
    """
    identifier, headers = new_session(api_client)
    other, other_headers = new_session(api_client)
    base = f"/api/v1/conversations/{identifier}"
    assert (
        api_client.post(
            base + "/messages", headers=headers, json={"message": "Olá."}
        ).status_code
        == 200
    )
    assert api_client.delete(base, headers=other_headers).status_code == 404
    deleted = api_client.delete(base, headers=headers)
    assert deleted.status_code == 204 and deleted.content == b""
    assert api_client.get(base, headers=headers).status_code == 404
    assert (
        api_client.get(
            f"/api/v1/conversations/{other}", headers=other_headers
        ).status_code
        == 200
    )
    with state_store.transaction() as connection:
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert (
                connection.execute(
                    f"SELECT count(*) AS count FROM {table} WHERE thread_id = %s",
                    (identifier,),
                ).fetchone()["count"]
                == 0
            )


def test_readiness_checks_real_data_without_model_generation(
    api_client: TestClient,
    api_app: FastAPI,
    api_model: ScriptedModel,
) -> None:
    """Distinguish readiness from liveness without constructing the agent runtime.

    Args:
        api_client: Real HTTP client.
        api_app: Application with lazy resources.
        api_model: Controlled generation spy.
    """
    assert api_client.get("/health").json() == {"status": "ok"}
    ready = api_client.get("/ready")
    assert ready.status_code == 200 and ready.json() == {"status": "ready"}
    assert api_app.state.resources._service is None
    assert api_model.seen == []


def test_readiness_and_creation_fail_closed_after_demo_tampering(
    api_client: TestClient,
    api_model: ScriptedModel,
    demo_engine: Engine,
) -> None:
    """Reject unverified data without exposing its contents or forwarding a prompt.

    Args:
        api_client: Real HTTP client.
        api_model: Controlled generation spy.
        demo_engine: Isolated synthetic database modified only for this test.
    """
    with demo_engine.begin() as connection:
        connection.execute(
            update(Product)
            .where(Product.product_id == 101)
            .values(description="private-data-canary")
        )
    for response in (
        api_client.get("/ready"),
        api_client.post("/api/v1/conversations"),
    ):
        assert response.status_code == 503
        assert "private-data-canary" not in response.text
    assert api_client.get("/health").status_code == 200
    assert api_model.seen == []


def test_missing_key_does_not_break_liveness_or_leak_configuration(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Report missing provider configuration without generating or breaking health.

    Args:
        demo_engine: Verified synthetic database.
        state_store: Initialized checkpoint schema.
        offline_settings: Explicit missing provider key.
    """
    with TestClient(
        create_app(settings=offline_settings, engine=demo_engine, store=state_store)
    ) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["error"]["code"] == "missing_api_key"
        assert client.post("/api/v1/conversations").status_code == 503


def test_uninitialized_state_requires_explicit_cli_initialization(
    demo_engine: Engine,
    state_namespace: str,
    offline_settings: AgentSettings,
) -> None:
    """Report missing state without silently migrating checkpoint tables.

    Args:
        demo_engine: Verified synthetic operational schema.
        state_namespace: Uninitialized disposable state namespace.
        offline_settings: Provider-free configuration.
    """
    with ConversationStore(state_namespace) as store:
        application = create_app(
            settings=offline_settings,
            engine=demo_engine,
            store=store,
            model=ScriptedModel(responses=[AIMessage(content="unused")]),
        )
        with TestClient(application) as client:
            assert client.get("/health").status_code == 200
            response = client.get("/ready")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "state_uninitialized"
            with pytest.raises(AgentFailure) as raised:
                store.check_ready()
            assert raised.value.code == ErrorCode.STATE_UNINITIALIZED


def test_real_execution_limits_remain_effective_over_http(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Stop actual graph loops and leave failed turns out of persisted history.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Initialized checkpoint store.
        offline_settings: Controlled settings with a stricter model-call budget.
    """
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_product_details",
                        "args": {"product_id": 101},
                        "id": "loop",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    settings = offline_settings.model_copy(update={"ai_max_model_calls": 1})
    with TestClient(
        create_app(
            settings=settings, engine=demo_engine, store=state_store, model=model
        )
    ) as client:
        identifier, headers = new_session(client)
        response = client.post(
            f"/api/v1/conversations/{identifier}/messages",
            headers=headers,
            json={"message": "Consulte o produto fictício 101."},
        )
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "execution_limit"
        assert len(model.seen) == 1
        assert (
            client.get(f"/api/v1/conversations/{identifier}", headers=headers).json()[
                "messages"
            ]
            == []
        )
