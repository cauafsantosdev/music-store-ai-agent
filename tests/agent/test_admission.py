"""Persistent public inference budgets using real PostgreSQL and controlled models."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID

import pytest
from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from psycopg import OperationalError
from pydantic import ValidationError
from sqlalchemy import Engine

from music_store_agent.agent import factory
from music_store_agent.agent.admission import ADMISSION_TABLES, AdmissionController
from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api.main import create_app


def enabled_settings(**changes: object) -> AgentSettings:
    """Create explicit offline admission configuration without external credentials.

    Args:
        changes: Explicit validated budget overrides for the selected regression.

    Returns:
        Enabled settings with bounded persistent budgets.
    """
    return AgentSettings(groq_api_key=None, ai_admission_enabled=True, **changes)


def usage(store: ConversationStore, identifier: UUID) -> tuple[int, int]:
    """Read persisted counters without logging capabilities or conversation contents.

    Args:
        store: Initialized isolated checkpoint and admission namespace.
        identifier: Authorized test conversation identifier.

    Returns:
        Current UTC daily model attempts and lifetime session admitted requests.
    """
    with store.transaction() as connection:
        daily = connection.execute(
            "SELECT model_calls FROM inference_daily_usage "
            "WHERE usage_day = (clock_timestamp() AT TIME ZONE 'UTC')::date"
        ).fetchone()
        session = connection.execute(
            "SELECT request_count FROM inference_session_usage "
            "WHERE conversation_id = %s",
            (identifier,),
        ).fetchone()
        return (
            daily["model_calls"] if daily else 0,
            session["request_count"] if session else 0,
        )


def test_admission_initialization_is_explicit_and_non_destructive(
    state_store: ConversationStore,
) -> None:
    """Upgrade usage tables explicitly while retaining existing conversation access.

    Args:
        state_store: Real original checkpoint schema without admission tables.
    """
    identifier, token = state_store.create()
    controller = AdmissionController(state_store, enabled_settings())
    with pytest.raises(AgentFailure) as failure:
        controller.check_ready()
    assert failure.value.code == ErrorCode.STATE_UNINITIALIZED
    state_store.initialize(admission=True)
    state_store.initialize(admission=True)
    controller.check_ready()
    with state_store.authorized(identifier, token):
        pass
    with state_store.transaction() as connection:
        names = {
            row["tablename"]
            for row in connection.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s",
                (state_store.schema,),
            )
        }
    assert ADMISSION_TABLES <= names
    assert usage(state_store, identifier) == (0, 0)


@pytest.mark.parametrize("owned", [False, True])
def test_failed_readiness_closes_only_owned_state_pool(
    demo_engine: Engine,
    state_store: ConversationStore,
    monkeypatch: pytest.MonkeyPatch,
    owned: bool,
) -> None:
    """Release failed factory-owned pools without closing externally managed pools.

    Args:
        demo_engine: Verified real sample database.
        state_store: Explicitly initialized state without admission counter tables.
        monkeypatch: Scoped factory hook retaining the real PostgreSQL pool.
        owned: Whether the factory creates and therefore owns the tested pool.
    """
    opened: list[ConversationStore] = []

    def open_owned(schema: str) -> ConversationStore:
        """Open a real initialized state pool and retain it for ownership assertions.

        Args:
            schema: Existing test namespace without admission initialization.

        Returns:
            A real pool that the factory must close if readiness fails.
        """
        store = ConversationStore(schema)
        store.check_ready()
        opened.append(store)
        return store

    monkeypatch.setattr(factory, "create_state_store", open_owned)
    model = ScriptedModel(responses=[AIMessage(content="Não deve executar.")])
    try:
        with pytest.raises(AgentFailure) as failure:
            create_store_agent(
                demo_engine,
                settings=enabled_settings(ai_state_schema=state_store.schema),
                model=model,
                state_store=None if owned else state_store,
            )
        assert failure.value.code == ErrorCode.STATE_UNINITIALIZED
        assert not model.seen
        if owned:
            assert len(opened) == 1
            assert opened[0].pool.closed
        else:
            assert not opened
            assert not state_store.pool.closed
            state_store.check_ready()
    finally:
        for store in opened:
            store.close()


def test_session_limit_prevents_inference_without_changing_history(
    demo_engine: Engine,
    state_store: ConversationStore,
) -> None:
    """Reject a third admitted request and retain two completed turns unchanged.

    Args:
        demo_engine: Approved sample records with real deterministic tools.
        state_store: Disposable real checkpoint namespace.
    """
    state_store.initialize(admission=True)
    model = ScriptedModel(
        responses=[
            AIMessage(content="Primeira resposta sintética.", id="admission-first"),
            AIMessage(content="Segunda resposta sintética.", id="admission-second"),
        ]
    )
    with create_store_agent(
        demo_engine,
        settings=enabled_settings(ai_session_request_limit=2),
        model=model,
        state_store=state_store,
    ) as agent:
        conversation = agent.create_conversation()
        assert conversation.send("Primeira pergunta.").code == "ok"
        assert conversation.send("Segunda pergunta.").code == "ok"
        assert conversation.send("Terceira pergunta.").code == ErrorCode.RATE_LIMIT
        assert len(model.seen) == 2
        assert (
            len(
                agent.get_conversation_history(
                    conversation.conversation_id, conversation.access_token
                )
            )
            == 4
        )
        assert usage(state_store, conversation.conversation_id) == (2, 2)


def test_failed_provider_attempt_is_not_refunded_after_restart(
    demo_engine: Engine,
    state_store: ConversationStore,
) -> None:
    """Persist attempted inference even though the entire graph turn rolls back.

    Args:
        demo_engine: Verified sample data.
        state_store: Real PostgreSQL state reused by a reconstructed service.
    """
    state_store.initialize(admission=True)
    settings = enabled_settings(ai_daily_model_call_limit=1)
    model = ScriptedModel(responses=[], failure_code=503)
    with create_store_agent(
        demo_engine, settings=settings, model=model, state_store=state_store
    ) as agent:
        conversation = agent.create_conversation()
        assert (
            conversation.send("Pergunta sintética.").code
            == ErrorCode.PROVIDER_UNAVAILABLE
        )
        assert len(model.seen) == 1
        assert (
            agent.get_conversation_history(
                conversation.conversation_id, conversation.access_token
            )
            == ()
        )
        assert usage(state_store, conversation.conversation_id) == (1, 1)
    with ConversationStore(state_store.schema) as fresh:
        replacement = ScriptedModel(responses=[AIMessage(content="Não deve executar.")])
        with create_store_agent(
            demo_engine, settings=settings, model=replacement, state_store=fresh
        ) as agent:
            resumed = agent.resume_conversation(
                conversation.conversation_id, conversation.access_token
            )
            assert (
                resumed.send("Nova tentativa explícita.").code == ErrorCode.RATE_LIMIT
            )
            assert not replacement.seen
            assert usage(fresh, conversation.conversation_id) == (1, 1)


def test_daily_budget_counts_each_model_call_within_a_tool_turn(
    demo_engine: Engine,
    state_store: ConversationStore,
) -> None:
    """Stop the second model attempt after a real tool, rolling back the whole turn.

    Args:
        demo_engine: Real approved sample catalog.
        state_store: Separate native checkpoint and accounting schema.
    """
    state_store.initialize(admission=True)
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_product_details",
                        "args": {"product_id": 101},
                        "id": "admission-tool",
                    }
                ],
            ),
            AIMessage(content="Não deve executar."),
        ]
    )
    with create_store_agent(
        demo_engine,
        settings=enabled_settings(ai_daily_model_call_limit=1),
        model=model,
        state_store=state_store,
    ) as agent:
        conversation = agent.create_conversation()
        assert conversation.send("Consulte o produto 101.").code == ErrorCode.RATE_LIMIT
        assert len(model.seen) == 1
        assert usage(state_store, conversation.conversation_id) == (1, 1)
        assert (
            agent.get_conversation_history(
                conversation.conversation_id, conversation.access_token
            )
            == ()
        )
        another = agent.create_conversation()
        assert another.send("Nova conversa.").code == ErrorCode.RATE_LIMIT
        assert len(model.seen) == 1
        assert usage(state_store, another.conversation_id) == (1, 0)


def test_invalid_capability_does_not_consume_usage(
    demo_engine: Engine,
    state_store: ConversationStore,
) -> None:
    """Authorize before any independently committed admission accounting occurs.

    Args:
        demo_engine: Approved sample catalog.
        state_store: Real isolated state namespace.
    """
    state_store.initialize(admission=True)
    model = ScriptedModel(responses=[AIMessage(content="Não deve executar.")])
    with create_store_agent(
        demo_engine, settings=enabled_settings(), model=model, state_store=state_store
    ) as agent:
        conversation = agent.create_conversation()
        reply = agent.send_message(conversation.conversation_id, "invalid", "Olá.")
        assert reply.code == ErrorCode.SESSION_UNVERIFIED
        assert not model.seen
        assert usage(state_store, conversation.conversation_id) == (0, 0)


def test_daily_budget_uses_database_utc_day(state_store: ConversationStore) -> None:
    """Open a new UTC day's allowance without resetting persisted historical counts.

    Args:
        state_store: Real PostgreSQL accounting tables.
    """
    state_store.initialize(admission=True)
    identifier, _ = state_store.create()
    with state_store.transaction() as connection:
        connection.execute(
            "INSERT INTO inference_daily_usage VALUES "
            "((clock_timestamp() AT TIME ZONE 'UTC')::date - 1, 100)"
        )
    controller = AdmissionController(
        state_store, enabled_settings(ai_daily_model_call_limit=1)
    )
    with controller.turn(identifier):
        controller.model_call()
    assert usage(state_store, identifier) == (1, 1)
    with state_store.transaction() as connection:
        assert (
            connection.execute(
                "SELECT sum(model_calls) AS total FROM inference_daily_usage"
            ).fetchone()["total"]
            == 101
        )


def test_atomic_daily_budget_across_concurrent_connections(
    state_store: ConversationStore,
) -> None:
    """Allow only one model attempt when two processes' slots race for one allowance.

    Args:
        state_store: Real shared checkpoint and accounting namespace.
    """
    state_store.initialize(admission=True)
    settings = enabled_settings(ai_daily_model_call_limit=1, ai_max_concurrent_turns=2)
    identifiers = [state_store.create()[0] for _ in range(2)]
    barrier = Barrier(2)

    def attempt(identifier: UUID) -> str:
        """Race the independently committed daily budget under separate slot locks.

        Args:
            identifier: Unique test conversation.

        Returns:
            Successful admission or its sanitized rejection code.
        """
        controller = AdmissionController(state_store, settings)
        with controller.turn(identifier):
            barrier.wait(timeout=5)
            try:
                controller.model_call()
                return "allowed"
            except AgentFailure as failure:
                return failure.code.value

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, identifiers))
    assert sorted(results) == ["allowed", "rate_limit"]
    assert usage(state_store, identifiers[0])[0] == 1


def test_concurrency_slot_is_cross_process_and_released(
    state_store: ConversationStore,
) -> None:
    """Reject another Python process during a held slot and admit it after release.

    Args:
        state_store: Real shared checkpoint namespace.
    """
    state_store.initialize(admission=True)
    first, _ = state_store.create()
    second, _ = state_store.create()
    controller = AdmissionController(state_store, enabled_settings())
    script = """
import json, sys
from music_store_agent.agent.admission import AdmissionController
from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.memory import ConversationStore
data=json.load(sys.stdin)
with ConversationStore(data["schema"]) as store:
    settings=AgentSettings(groq_api_key=None, ai_admission_enabled=True)
    controller=AdmissionController(store, settings)
    try:
        with controller.turn(data["identifier"]):
            print("allowed")
    except AgentFailure as failure:
        print(failure.code.value)
"""

    def child_attempt() -> str:
        """Run one independent PostgreSQL admission attempt without raw diagnostics.

        Returns:
            Only its fixed success or sanitized failure category.
        """
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps({"schema": state_store.schema, "identifier": str(second)}),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0
        return result.stdout.strip()

    with controller.turn(first):
        assert child_attempt() == "rate_limit"
    assert child_attempt() == "allowed"
    assert usage(state_store, first) == (0, 1)
    assert usage(state_store, second) == (0, 1)


def test_enabled_model_guard_requires_an_admitted_context(
    state_store: ConversationStore,
) -> None:
    """Fail closed if inference is attempted outside the admission context.

    Args:
        state_store: Explicit existing checkpoint namespace.
    """
    controller = AdmissionController(state_store, enabled_settings())
    with pytest.raises(AgentFailure) as failure:
        controller.model_call()
    assert failure.value.code == ErrorCode.CONFIGURATION


def test_accounting_database_failure_is_sanitized_and_closed(
    state_store: ConversationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject inference if the independent accounting connection cannot be opened.

    Args:
        state_store: Real existing state schema and authorized test conversation.
        monkeypatch: Scoped accounting connection failure without external traffic.
    """
    identifier, _ = state_store.create()
    controller = AdmissionController(state_store, enabled_settings())

    def unavailable(*args: object, **kwargs: object) -> None:
        """Simulate a connection failure containing unpublishable driver details.

        Args:
            args: Uninspected connection arguments.
            kwargs: Uninspected connection configuration.

        Raises:
            OperationalError: A controlled unpublishable database diagnostic.
        """
        raise OperationalError("private-connection-diagnostic")

    monkeypatch.setattr(
        "music_store_agent.agent.admission.Connection.connect", unavailable
    )
    with pytest.raises(AgentFailure) as failure:
        with controller.turn(identifier):
            pytest.fail("Unavailable accounting permitted inference.")
    assert failure.value.code == ErrorCode.DATABASE
    assert "private-connection-diagnostic" not in str(failure.value)


def test_api_readiness_and_sanitized_admission_feedback(
    demo_engine: Engine,
    state_store: ConversationStore,
) -> None:
    """Require initialized counters and preserve JSON/SSE feedback and history reads.

    Args:
        demo_engine: Real approved sample data.
        state_store: Existing isolated state namespace without implicit migrations.
    """
    model = ScriptedModel(responses=[AIMessage(content="Resposta sintética.")])
    with TestClient(
        create_app(
            settings=enabled_settings(ai_session_request_limit=1),
            model=model,
            engine=demo_engine,
            store=state_store,
        )
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        state_store.initialize(admission=True)
        assert client.get("/ready").status_code == 200
        created = client.post("/api/v1/conversations").json()
        target = "/api/v1/conversations/" + created["conversation_id"]
        headers = {"Authorization": "Bearer " + created["access_token"]}
        assert (
            client.post(
                target + "/messages", headers=headers, json={"message": "Olá."}
            ).status_code
            == 200
        )
        rejected = client.post(
            target + "/messages",
            headers=headers,
            json={"message": "Mais uma pergunta."},
        )
        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == ErrorCode.RATE_LIMIT
        streamed = client.post(
            target + "/stream",
            headers=headers,
            json={"message": "Nova tentativa explícita."},
        )
        assert streamed.status_code == 200
        assert "event: error" in streamed.text
        assert '"code": "rate_limit"' in streamed.text
        assert "event: done" in streamed.text
        assert len(model.seen) == 1
        history = client.get(target, headers=headers)
        assert history.status_code == 200
        assert len(history.json()["messages"]) == 2
        assert (
            created["access_token"] not in rejected.text + streamed.text + history.text
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("ai_session_request_limit", 0),
        ("ai_session_request_limit", 101),
        ("ai_daily_model_call_limit", 0),
        ("ai_daily_model_call_limit", 1001),
        ("ai_max_concurrent_turns", 0),
        ("ai_max_concurrent_turns", 5),
    ],
)
def test_unsafe_admission_settings_are_rejected(name: str, value: int) -> None:
    """Reject unbounded or invalid limits before any provider construction.

    Args:
        name: Budget configuration field.
        value: Invalid explicit bound.
    """
    with pytest.raises(ValidationError):
        enabled_settings(**{name: value})
