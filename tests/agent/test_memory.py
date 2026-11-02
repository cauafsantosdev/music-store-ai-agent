"""Real PostgreSQL checkpoint isolation, recovery, concurrency and capability tests."""

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import UUID, uuid4

import pytest
from conftest import ScriptedModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import sql
from pydantic import BaseModel, SecretStr
from sqlalchemy import Engine, inspect, text

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import (
    REQUIRED_TABLES,
    ConversationStore,
    create_state_store,
)


class BlockingModel(ScriptedModel):
    """Pause controlled generation while real checkpoint transactions remain open."""

    entered: Event
    release: Event

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        """Signal model entry and await a bounded release from the test.

        Args:
            messages: Native graph context.
            stop: Optional framework stop sequences.
            run_manager: Local framework callbacks.
            kwargs: Additional supported generation options.

        Returns:
            The scripted native model result after release.
        """
        self.entered.set()
        assert self.release.wait(8), "Controlled model release timed out."
        return super()._generate(messages, stop, run_manager, **kwargs)


class UnapprovedRecord(BaseModel):
    """Represent a custom type outside the official safe deserialization allowlist."""

    value: str


def tool_turn() -> list[AIMessage]:
    """Provide a synthetic product lookup with opaque provider metadata.

    Returns:
        One native tool request and its final response.
    """
    return [
        AIMessage(
            content="",
            additional_kwargs={"opaque_signature": "synthetic-signature"},
            tool_calls=[
                {
                    "name": "get_product_details",
                    "args": {"product_id": 101},
                    "id": "persistent-call",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Produto fictício consultado."),
    ]


def checkpoint_counts(store: ConversationStore, identifier: UUID) -> tuple[int, ...]:
    """Count all native checkpoint rows for rollback and deletion verification.

    Args:
        store: Isolated PostgreSQL state store.
        identifier: Target UUID.

    Returns:
        Counts for checkpoints, blobs and pending writes.
    """
    with store.transaction() as connection:
        return tuple(
            connection.execute(
                sql.SQL("SELECT count(*) AS count FROM {} WHERE thread_id = %s").format(
                    sql.Identifier(table)
                ),
                (str(identifier),),
            ).fetchone()["count"]
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
        )


def test_checkpoint_initialization_is_idempotent_and_physically_isolated(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Keep all checkpoint tables outside operational and public schemas.

    Args:
        demo_engine: Verified synthetic operational namespace.
        state_store: Explicitly initialized disposable state schema.
    """
    public = set(inspect(demo_engine).get_table_names(schema="public"))
    state_store.initialize()
    state_store.initialize()
    with state_store.pool.connection() as connection:
        assert connection.autocommit
        assert (
            connection.execute("SHOW search_path").fetchone()["search_path"]
            == state_store.schema
        )
        names = {
            row["tablename"]
            for row in connection.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s",
                (state_store.schema,),
            )
        }
    assert REQUIRED_TABLES == names
    assert set(inspect(demo_engine).get_table_names(schema="public")) == public
    verify_demo(demo_engine)


@pytest.mark.parametrize(
    "schema",
    [
        "public",
        "agent_demo",
        "agent_state;DROP SCHEMA public",
        "agent_state_" + "x" * 64,
    ],
)
def test_untrusted_state_namespaces_are_rejected(schema: str) -> None:
    """Reject arbitrary schemas before opening a connection or setting search_path.

    Args:
        schema: Unsafe deployment configuration.
    """
    with pytest.raises(AgentFailure) as raised:
        ConversationStore(schema)
    assert raised.value.code == ErrorCode.CONFIGURATION


def test_uninitialized_store_fails_without_implicit_ddl(state_namespace: str) -> None:
    """Require an explicit initialization operation before service startup.

    Args:
        state_namespace: Disposable namespace that does not yet contain tables.
    """
    with pytest.raises(AgentFailure) as raised:
        create_state_store(state_namespace)
    assert raised.value.code == ErrorCode.STATE_UNINITIALIZED


def test_setup_is_not_run_on_service_creation_or_messages(
    demo_engine: Engine, state_store: ConversationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep checkpoint migrations outside normal conversation operations.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Already initialized state store.
        monkeypatch: Scoped rejection of any implicit migration call.
    """

    def forbidden(self: PostgresSaver) -> None:
        """Reject implicit migrations after explicit initialization.

        Args:
            self: Official saver.
        """
        pytest.fail("Unexpected checkpoint setup during service execution.")

    monkeypatch.setattr(PostgresSaver, "setup", forbidden)
    with create_store_agent(
        demo_engine,
        state_store=state_store,
        model=ScriptedModel(responses=[AIMessage(content="Olá.")]),
    ) as agent:
        assert agent.create_conversation().send("Olá.").code == "ok"


def test_capabilities_are_random_and_only_hashes_are_stored(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Generate unique UUIDs and 256-bit capabilities without storing plaintext.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated state store.
    """
    model = ScriptedModel(responses=[AIMessage(content="Olá.")])
    agent = create_store_agent(demo_engine, state_store=state_store, model=model)
    first, second = agent.create_conversation(), agent.new_conversation()
    token = first.access_token.get_secret_value()
    assert isinstance(first.conversation_id, UUID)
    assert first.conversation_id.version == 4
    assert first.conversation_id != second.conversation_id
    assert len(token) >= 43
    assert first.access_token != second.access_token
    assert token not in repr(first) + repr(first.access_token)
    assert model.seen == []
    with state_store.transaction() as connection:
        row = connection.execute(
            "SELECT * FROM conversations WHERE conversation_id = %s",
            (first.conversation_id,),
        ).fetchone()
    assert set(row) == {
        "conversation_id",
        "access_token_hash",
        "created_at",
        "updated_at",
    }
    assert bytes(row["access_token_hash"]) == hashlib.sha256(token.encode()).digest()
    assert row["created_at"].tzinfo is not None
    assert checkpoint_counts(state_store, first.conversation_id) == (0, 0, 0)


@pytest.mark.parametrize(
    "method",
    [
        "resume_conversation",
        "get_conversation_history",
        "delete_conversation",
        "send_message",
    ],
)
@pytest.mark.parametrize(
    "wrong", ["unknown", "bad_token", "other_session", "malformed"]
)
def test_session_access_failures_are_uniform(
    demo_engine: Engine, state_store: ConversationStore, method: str, wrong: str
) -> None:
    """Reject unauthorized operations before reading or changing any checkpoint.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated state store.
        method: Protected public service operation.
        wrong: Unknown UUID, invalid capability, cross-session token, or bad ID.
    """
    model = ScriptedModel(responses=[AIMessage(content="Synthetic private context.")])
    agent = create_store_agent(demo_engine, state_store=state_store, model=model)
    first, second = agent.create_conversation(), agent.create_conversation()
    assert first.send("Synthetic conversation.").code == "ok"
    identifier = uuid4() if wrong == "unknown" else first.conversation_id
    identifier = "malformed" if wrong == "malformed" else identifier
    token = second.access_token if wrong == "other_session" else first.access_token
    token = "invalid-session-token" if wrong == "bad_token" else token
    if method == "send_message":
        result = agent.send_message(identifier, token, "Olá.")
        assert result.code == "session_unverified"
        output = result.text
    else:
        with pytest.raises(AgentFailure) as raised:
            getattr(agent, method)(identifier, token)
        assert raised.value.code == ErrorCode.SESSION_UNVERIFIED
        output = str(raised.value)
    assert output == "Não foi possível acessar essa conversa."
    assert "context" not in output
    assert len(model.seen) == 1
    assert len(first.messages) == 2
    assert second.messages == ()


def test_persistence_across_service_pool_and_connection_reconstruction(
    demo_engine: Engine, state_namespace: str
) -> None:
    """Restore tool IDs and opaque metadata through a fresh service and database pool.

    Args:
        demo_engine: Verified synthetic catalog.
        state_namespace: Disposable state namespace.
    """
    with ConversationStore(state_namespace) as original:
        original.initialize()
        with original.pool.connection() as connection:
            old_pid = connection.info.backend_pid
        agent = create_store_agent(
            demo_engine,
            state_store=original,
            model=ScriptedModel(responses=tool_turn()),
        )
        first, other = agent.new_conversation(), agent.new_conversation()
        assert first.send("Consulte o produto fictício 101.").code == "ok"
        identifier, token = first.conversation_id, first.access_token
        other_id, other_token = other.conversation_id, other.access_token
        agent.close()
    assert original.pool.closed
    model = ScriptedModel(
        responses=[AIMessage(content="O mesmo produto está no contexto.")]
    )
    with ConversationStore(state_namespace) as fresh:
        with fresh.pool.connection() as connection:
            assert connection.info.backend_pid != old_pid
        agent = create_store_agent(demo_engine, state_store=fresh, model=model)
        resumed = agent.resume_conversation(str(identifier).upper(), token)
        assert resumed.send("E o mesmo produto?").code == "ok"
        saved = model.seen[-1]
        signed = next(
            item for item in saved if isinstance(item, AIMessage) and item.tool_calls
        )
        assert signed.additional_kwargs == {"opaque_signature": "synthetic-signature"}
        assert signed.tool_calls[0]["id"] == "persistent-call"
        assert any(
            isinstance(item, ToolMessage) and item.tool_call_id == "persistent-call"
            for item in saved
        )
        assert len([item for item in saved if isinstance(item, HumanMessage)]) == 2
        assert agent.get_conversation_history(other_id, other_token) == ()
        display = agent.get_conversation_history(str(identifier).upper(), token)
        assert [item.role for item in display] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert "opaque_signature" not in repr(display)
        assert all(item.additional_kwargs == {} for item in resumed.messages)


def test_restart_in_a_fresh_python_process(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Resume native state in a new interpreter without sharing process memory.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated initialized state namespace.
    """
    agent = create_store_agent(
        demo_engine, state_store=state_store, model=ScriptedModel(responses=tool_turn())
    )
    session = agent.create_conversation()
    assert session.send("Consulte o produto fictício 101.").code == "ok"
    agent.close()
    payload = {
        "schema": state_store.schema,
        "demo": demo_engine.get_execution_options()["schema_translate_map"][None],
        "identifier": str(session.conversation_id),
        "token": session.access_token.get_secret_value(),
    }
    # Capabilities travel through stdin, never process arguments, environment or logs.
    script = """
import json, sys
sys.path.insert(0, "/app/tests/agent")
from conftest import ScriptedModel
from langchain_core.messages import AIMessage, ToolMessage
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.database.session import create_database_engine
data = json.load(sys.stdin)
engine = create_database_engine(schema=data["demo"])
model = ScriptedModel(responses=[AIMessage(content="Contexto fictício recuperado.")])
with ConversationStore(data["schema"]) as store:
    with create_store_agent(engine, state_store=store, model=model) as agent:
        conversation = agent.resume_conversation(data["identifier"], data["token"])
        result = conversation.send("E o mesmo produto?")
        assert result.code == "ok"
        assert len(conversation.messages) == 6
        assert any(
            isinstance(m, ToolMessage) and m.tool_call_id == "persistent-call"
            for m in model.seen[-1]
        )
        assert any(
            isinstance(m, AIMessage)
            and m.additional_kwargs.get("opaque_signature") == "synthetic-signature"
            for m in model.seen[-1]
        )
engine.dispose()
print("restart-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        "Fresh-process verification failed; output redacted."
    )
    assert completed.stdout.strip() == "restart-ok"


def test_deletion_removes_every_checkpoint_and_preserves_other_sessions(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Remove authorized metadata, checkpoints, blobs and pending writes atomically.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated state store.
    """
    agent = create_store_agent(
        demo_engine,
        state_store=state_store,
        model=ScriptedModel(responses=[AIMessage(content="Olá.")]),
    )
    first, other = agent.create_conversation(), agent.create_conversation()
    assert first.send("Olá.").code == other.send("Outra conversa.").code == "ok"
    assert checkpoint_counts(state_store, first.conversation_id)[0] > 0
    agent.delete_conversation(str(first.conversation_id).upper(), first.access_token)
    assert checkpoint_counts(state_store, first.conversation_id) == (0, 0, 0)
    assert first.send("Olá.").code == "session_unverified"
    with pytest.raises(AgentFailure):
        agent.resume_conversation(first.conversation_id, first.access_token)
    assert len(other.messages) == 2
    assert checkpoint_counts(state_store, other.conversation_id)[0] > 0


def test_failed_turn_rolls_back_intermediate_checkpoints_and_can_retry(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Rollback real tool checkpoints and pending writes when the final response fails.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated state store.
    """
    model = ScriptedModel(
        responses=[AIMessage(content="Olá."), tool_turn()[0], AIMessage(content="")]
    )
    agent = create_store_agent(demo_engine, state_store=state_store, model=model)
    session = agent.create_conversation()
    assert session.send("Olá.").code == "ok"
    before = checkpoint_counts(state_store, session.conversation_id)
    assert session.send("Consulte o produto fictício 101.").code == "empty_response"
    assert checkpoint_counts(state_store, session.conversation_id) == before
    assert len(session.messages) == 2
    retry_model = ScriptedModel(responses=[AIMessage(content="Consulta recuperada.")])
    fresh = create_store_agent(demo_engine, state_store=state_store, model=retry_model)
    recovered = fresh.resume_conversation(session.conversation_id, session.access_token)
    assert recovered.send("Continue.").code == "ok"
    assert not any(isinstance(item, ToolMessage) for item in retry_model.seen[-1])
    assert len(recovered.messages) == 4


def test_separate_instances_serialize_same_thread_without_blocking_other_threads(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Verify real row-lock contention across independent pools and agent instances.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Initialized isolated state store for the first instance.
    """
    entered, release = Event(), Event()
    model = BlockingModel(
        entered=entered,
        release=release,
        responses=[AIMessage(content="Primeira resposta.")],
    )
    first = create_store_agent(demo_engine, state_store=state_store, model=model)
    session = first.create_conversation()
    with ConversationStore(state_store.schema) as second_store:
        second_model = ScriptedModel(responses=[AIMessage(content="Outra resposta.")])
        second = create_store_agent(
            demo_engine, state_store=second_store, model=second_model
        )
        other = second.create_conversation()
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending_first = executor.submit(session.send, "Primeira pergunta fictícia.")
            assert entered.wait(5)
            pending_second = executor.submit(
                second.send_message,
                session.conversation_id,
                session.access_token,
                "Segunda pergunta fictícia.",
            )
            try:
                deadline = time.monotonic() + 3
                waiting = False
                while time.monotonic() < deadline:
                    with demo_engine.connect() as connection:
                        waiting = connection.scalar(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                                "WHERE wait_event_type = 'Lock' "
                                "AND query LIKE :pattern)"
                            ),
                            {"pattern": "SELECT access_token_hash%"},
                        )
                    if waiting:
                        break
                    Event().wait(0.01)
                assert waiting, (
                    "The second instance never reached the PostgreSQL row lock."
                )
                with pytest.raises(AgentFailure) as invalid:
                    second.resume_conversation(session.conversation_id, "invalid-token")
                assert invalid.value.code == ErrorCode.SESSION_UNVERIFIED
                assert other.send("Conversa independente.").code == "ok"
                assert not pending_second.done()
            finally:
                release.set()
            assert pending_first.result(timeout=10).code == "ok"
            assert pending_second.result(timeout=10).code == "ok"
        assert len(session.messages) == 4
        assert any(
            item.content == "Primeira pergunta fictícia."
            for item in second_model.seen[-1]
        )
        assert len(other.messages) == 2


def test_database_connection_loss_rolls_back_and_returns_safe_error(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Terminate only the test pool's backend during a turn and verify atomic recovery.

    Args:
        demo_engine: Disposable operational engine used for the test-only termination.
        state_store: Isolated state store whose backend belongs to this test.
    """
    entered, release = Event(), Event()
    model = BlockingModel(
        entered=entered,
        release=release,
        responses=[AIMessage(content="Não deve ser confirmada.")],
    )
    agent = create_store_agent(demo_engine, state_store=state_store, model=model)
    session = agent.create_conversation()
    with state_store.pool.connection() as connection:
        backend = connection.info.backend_pid
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(session.send, "Pergunta fictícia.")
        assert entered.wait(5)
        try:
            with demo_engine.connect() as connection:
                assert connection.scalar(
                    text("SELECT pg_terminate_backend(:pid)"), {"pid": backend}
                )
        finally:
            release.set()
        result = pending.result(timeout=10)
    assert result.code == "database"
    assert "SELECT" not in result.text
    with ConversationStore(state_store.schema) as recovered:
        fresh = create_store_agent(
            demo_engine,
            state_store=recovered,
            model=ScriptedModel(responses=[AIMessage(content="Olá.")]),
        )
        assert (
            fresh.get_conversation_history(
                session.conversation_id, session.access_token
            )
            == ()
        )
        assert checkpoint_counts(recovered, session.conversation_id) == (0, 0, 0)


def test_strict_serialization_and_checkpoint_secret_exclusion(
    demo_engine: Engine, state_store: ConversationStore
) -> None:
    """Reject pickle/custom types and keep credentials outside serialized graph state.

    Args:
        demo_engine: Verified synthetic catalog.
        state_store: Isolated state store.
    """
    key = "synthetic-api-key-canary"
    agent = create_store_agent(
        demo_engine,
        state_store=state_store,
        settings=AgentSettings(groq_api_key=SecretStr(key)),
        model=ScriptedModel(responses=[AIMessage(content="Olá.")]),
    )
    session = agent.create_conversation()
    assert session.send("Olá.").code == "ok"
    with state_store.transaction() as connection:
        serde = state_store.saver(connection).serde
        assert not serde.pickle_fallback
        assert serde._allowed_msgpack_modules is None
        with pytest.raises(NotImplementedError):
            serde.loads_typed(("pickle", b"untrusted"))
        encoded = serde.dumps_typed(UnapprovedRecord(value="synthetic"))
        decoded = serde.loads_typed(encoded)
        assert isinstance(decoded, dict)
        assert decoded["value"] == "synthetic"
        blobs = connection.execute(
            "SELECT blob FROM checkpoint_blobs WHERE thread_id = %s",
            (str(session.conversation_id),),
        ).fetchall()
        for row in blobs:
            data = bytes(row["blob"] or b"")
            assert key.encode() not in data
            assert session.access_token.get_secret_value().encode() not in data


def test_owned_pool_closes_with_service_context(demo_engine: Engine) -> None:
    """Release owned pool workers and connections at explicit service shutdown.

    Args:
        demo_engine: Verified synthetic catalog.
    """
    with create_store_agent(
        demo_engine, model=ScriptedModel(responses=[AIMessage(content="Olá.")])
    ) as agent:
        session = agent.create_conversation()
        assert session.send("Olá.").code == "ok"
    assert agent._store.pool.closed
    assert session.send("Olá.").code == "database"
