"""HTTP lifespan, restart, ASGI streaming and nonblocking execution integration."""

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from threading import Event

import httpx
import pytest
from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr
from sqlalchemy import Engine, event
from starlette.types import Message
from test_api import new_session, parse_events
from test_memory import BlockingModel

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_live_model
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api import dependencies
from music_store_agent.api.main import create_app
from music_store_agent.api.middleware import MAX_BODY_BYTES


def test_fresh_import_and_liveness_need_neither_database_nor_key() -> None:
    """Import and serve liveness with unavailable dependencies in a fresh process."""
    environment = dict(os.environ)
    environment.pop("GROQ_API_KEY", None)
    environment.update(
        PGHOST="unavailable.invalid", PGCONNECT_TIMEOUT="1", AI_MODEL="invalid-model"
    )
    script = (
        "from fastapi.testclient import TestClient\n"
        "from music_store_agent.api.main import app\n"
        "with TestClient(app) as client:\n"
        "    assert client.get('/health').json() == {'status': 'ok'}\n"
        "print('health-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, "Fresh liveness probe failed; diagnostics redacted."
    assert result.stdout.strip() == "health-ok"


def test_lifespan_reuses_and_closes_owned_engine_pool_and_provider_clients(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Own resources once per lifespan without generating or migrating at startup.

    Args:
        demo_engine: Isolated operational engine returned by the engine factory.
        state_store: Initialized checkpoint schema, retained by the fixture.
        offline_settings: Validated configuration with a synthetic test credential.
        monkeypatch: Scoped factory instrumentation around real resource creation.
    """
    settings = offline_settings.model_copy(
        update={"groq_api_key": SecretStr("offline-only-not-real")}
    )
    model = create_live_model(settings)
    opened: list[ConversationStore] = []
    calls: list[str] = []

    def make_engine(*, schema: str) -> Engine:
        """Track engine ownership without accessing development data.

        Args:
            schema: Validated requested operational namespace.

        Returns:
            The isolated real operational engine.
        """
        calls.append("engine")
        return demo_engine

    def make_store(schema: str) -> ConversationStore:
        """Open a real pool without implicit initialization or migrations.

        Args:
            schema: Already initialized test namespace.

        Returns:
            A real owned application pool.
        """
        calls.append("store")
        store = ConversationStore(schema)
        store.check_ready()
        opened.append(store)
        return store

    def make_model(selected: AgentSettings) -> ChatGroq:
        """Return a real SDK client without performing any provider request.

        Args:
            selected: Validated application configuration.

        Returns:
            The actual offline-credential Groq model.
        """
        calls.append("model")
        return model

    disposed: list[bool] = []

    def on_dispose(engine: Engine) -> None:
        """Record explicit engine disposal.

        Args:
            engine: Disposed test engine.
        """
        disposed.append(True)

    event.listen(demo_engine, "engine_disposed", on_dispose)
    monkeypatch.setattr(dependencies, "create_database_engine", make_engine)
    monkeypatch.setattr(dependencies, "create_state_store", make_store)
    monkeypatch.setattr(dependencies, "create_live_model", make_model)
    app = create_app(settings=settings)
    try:
        with TestClient(app) as client:
            assert calls == []
            assert client.get("/health").status_code == 200 and calls == []
            assert client.get("/ready").status_code == 200
            assert client.get("/ready").status_code == 200
            new_session(client)
            new_session(client)
            assert calls == ["engine", "store", "model"]
            assert not model.client._client._client.is_closed
        assert len(opened) == 1 and opened[0].pool.closed
        assert not state_store.pool.closed
        assert model.client._client._client.is_closed
        assert model.async_client._client._client.is_closed
        assert disposed == [True]
    finally:
        event.remove(demo_engine, "engine_disposed", on_dispose)


def test_readiness_database_failure_is_sanitized_without_affecting_liveness(
    monkeypatch: pytest.MonkeyPatch,
    offline_settings: AgentSettings,
) -> None:
    """Report dependency failures as unavailable while preserving independent health.

    Args:
        monkeypatch: Controlled database-failure boundary.
        offline_settings: Provider-free settings.
    """

    def unavailable(engine: Engine) -> None:
        """Represent a classified PostgreSQL connectivity failure.

        Args:
            engine: Unconnected lazy engine.

        Raises:
            AgentFailure: Always, without raw connection diagnostics.
        """
        raise AgentFailure(ErrorCode.DATABASE)

    monkeypatch.setattr(dependencies, "verify_demo", unavailable)
    with TestClient(create_app(settings=offline_settings)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "database"
        assert "PGHOST" not in response.text and "postgresql" not in response.text
        assert client.get("/health").status_code == 200


def test_http_history_survives_a_new_application_and_database_pool(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Resume the same HTTP capability through new application and service instances.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Original checkpoint pool.
        offline_settings: Explicit controlled settings.
    """
    initial_model = ScriptedModel(responses=[AIMessage(content="Contexto fictício.")])
    with TestClient(
        create_app(
            settings=offline_settings,
            engine=demo_engine,
            store=state_store,
            model=initial_model,
        )
    ) as first:
        identifier, headers = new_session(first)
        other, other_headers = new_session(first)
        assert (
            first.post(
                f"/api/v1/conversations/{identifier}/messages",
                headers=headers,
                json={"message": "Produto fictício 101."},
            ).status_code
            == 200
        )
    model = ScriptedModel(responses=[AIMessage(content="O mesmo produto.")])
    with ConversationStore(state_store.schema) as fresh_store:
        with TestClient(
            create_app(
                settings=offline_settings,
                engine=demo_engine,
                store=fresh_store,
                model=model,
            )
        ) as second:
            response = second.post(
                f"/api/v1/conversations/{identifier}/messages",
                headers=headers,
                json={"message": "E o mesmo produto?"},
            )
            assert response.status_code == 200
            assert any(
                item.content == "Produto fictício 101." for item in model.seen[0]
            )
            history = second.get(
                f"/api/v1/conversations/{identifier}", headers=headers
            ).json()
            assert len(history["messages"]) == 4
            assert (
                second.get(
                    f"/api/v1/conversations/{other}", headers=other_headers
                ).json()["messages"]
                == []
            )


@pytest.mark.parametrize("disconnect", [False, True])
def test_sse_flushes_status_and_does_not_block_other_http_requests(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    disconnect: bool,
) -> None:
    """Observe raw ASGI frames during blocked inference and a simulated disconnect.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Real isolated persistence.
        offline_settings: Explicit controlled settings.
        disconnect: Whether delivery fails after the committed response is produced.
    """
    entered, release = Event(), Event()
    model = BlockingModel(
        entered=entered,
        release=release,
        responses=[AIMessage(content="Resposta fictícia completa.")],
    )
    app = create_app(
        settings=offline_settings, engine=demo_engine, store=state_store, model=model
    )

    async def scenario() -> None:
        """Run concurrent HTTP checks without a buffering client hiding SSE timing."""
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                created = await client.post("/api/v1/conversations")
                data = created.json()
                identifier = data["conversation_id"]
                headers = {"Authorization": f"Bearer {data['access_token']}"}
                body = json.dumps({"message": "Olá, demonstração."}).encode()
                frames: asyncio.Queue[bytes] = asyncio.Queue()
                request_delivered = False
                never = asyncio.Event()

                async def receive() -> Message:
                    """Provide a bounded request body without spurious disconnects.

                    Returns:
                        Initial request data, followed by an unused disconnect wait.
                    """
                    nonlocal request_delivered
                    if not request_delivered:
                        request_delivered = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    await never.wait()
                    return {"type": "http.disconnect"}

                async def send(message: Message) -> None:
                    """Record each body frame immediately or simulate a closed socket.

                    Args:
                        message: Outgoing ASGI frame, never logged.

                    Raises:
                        OSError: For a simulated client disconnect after execution.
                    """
                    if message["type"] == "http.response.body" and message.get("body"):
                        if disconnect and b"event: message" in message["body"]:
                            raise OSError("Synthetic client disconnect.")
                        await frames.put(message["body"])

                scope = {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.4"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": f"/api/v1/conversations/{identifier}/stream",
                    "query_string": b"",
                    "root_path": "",
                    "server": ("test", 80),
                    "client": ("test", 123),
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"authorization", headers["Authorization"].encode()),
                    ],
                }
                running = asyncio.create_task(app(scope, receive, send))
                try:
                    first = await asyncio.wait_for(frames.get(), timeout=3)
                    assert parse_events(first.decode())[0] == (
                        "status",
                        {"status": "processing"},
                    )
                    assert await asyncio.to_thread(entered.wait, 3)
                    assert not running.done()
                    health = await asyncio.wait_for(client.get("/health"), timeout=2)
                    assert health.status_code == 200
                    independent = await asyncio.wait_for(
                        client.post("/api/v1/conversations"), timeout=2
                    )
                    assert independent.status_code == 201
                finally:
                    release.set()
                await asyncio.wait_for(running, timeout=10)
                remaining = []
                while not frames.empty():
                    remaining.extend(parse_events((await frames.get()).decode()))
                if not disconnect:
                    assert [name for name, _ in remaining] == ["message", "done"]
                assert len(model.seen) == 1
                history = await client.get(
                    f"/api/v1/conversations/{identifier}", headers=headers
                )
                # A completed transactional turn may commit even if delivery fails.
                assert len(history.json()["messages"]) == 2

    asyncio.run(scenario())


def test_chunked_request_body_is_bounded_before_service_execution(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Reject oversized bodies without Content-Length before resolving agent resources.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Initialized state store.
        offline_settings: Explicit controlled settings.
    """
    model = ScriptedModel(responses=[AIMessage(content="unused")])
    app = create_app(
        settings=offline_settings, engine=demo_engine, store=state_store, model=model
    )

    async def scenario() -> None:
        """Exercise chunked ASGI input without any outbound HTTP transport."""

        async def chunks() -> AsyncIterator[bytes]:
            """Yield two chunks whose combined length exceeds the request limit.

            Yields:
                Synthetic oversized input chunks.
            """
            yield b"x" * MAX_BODY_BYTES
            yield b"x"

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/api/v1/conversations", content=chunks())
                assert response.status_code == 413
                assert app.state.resources._service is None
                assert model.seen == []

    asyncio.run(scenario())


def test_openapi_describes_public_contract_without_native_history() -> None:
    """Expose typed versioned schemas and bearer security without provider metadata."""
    schema = create_app().openapi()
    properties = schema["components"]["schemas"]["MessageRequest"]["properties"]
    assert properties["message"]["maxLength"] == 2000
    assert (
        "access_token"
        in schema["components"]["schemas"]["ConversationCreated"]["properties"]
    )
    assert set(schema["components"]["schemas"]["HistoryMessage"]["properties"]) == {
        "role",
        "text",
    }
    stream = schema["paths"]["/api/v1/conversations/{conversation_id}/stream"]["post"]
    assert "text/event-stream" in stream["responses"]["200"]["content"]
    assert stream["security"] == [{"HTTPBearer": []}]
    assert "AIMessage" not in schema["components"]["schemas"]
