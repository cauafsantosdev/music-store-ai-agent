"""Persistent inference admission accounting and cross-process PostgreSQL slots."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

from psycopg import Connection, Error
from psycopg.rows import DictRow, dict_row

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.memory import ConversationStore

ADMISSION_TABLES: frozenset[str] = frozenset(
    {"inference_daily_usage", "inference_session_usage"}
)


def initialize_admission(connection: Connection[DictRow]) -> None:
    """Create usage counters explicitly without changing checkpoints or source data.

    Args:
        connection: Autocommit state connection under the existing setup lock.
    """
    connection.execute(
        "CREATE TABLE IF NOT EXISTS inference_daily_usage ("
        "usage_day DATE PRIMARY KEY, "
        "model_calls INTEGER NOT NULL CHECK (model_calls >= 0))"
    )
    # No foreign key: a separately committed counter must not wait on the outer
    # conversation transaction's row lock or disappear when a conversation is deleted.
    connection.execute(
        "CREATE TABLE IF NOT EXISTS inference_session_usage ("
        "conversation_id UUID PRIMARY KEY, "
        "request_count INTEGER NOT NULL CHECK (request_count >= 0))"
    )


class AdmissionController:
    """Bound admitted turns and model attempts independently of checkpoint rollback.

    Counters are shared by deployments using the same state schema. A dedicated
    PostgreSQL connection holds a session advisory slot during one complete turn;
    database disconnect or process termination releases the slot automatically.
    """

    def __init__(self, store: ConversationStore, settings: AgentSettings) -> None:
        """Retain explicit settings without connecting, migrating or loading secrets.

        Args:
            store: Existing state store with its validated schema namespace.
            settings: Validated admission budgets; disabled for default local use.
        """
        self._store = store
        self._settings = settings
        self._connection: ContextVar[Connection[DictRow] | None] = ContextVar(
            "inference_admission_connection", default=None
        )

    def check_ready(self) -> None:
        """Require explicit usage-table initialization whenever admission is enabled.

        Raises:
            AgentFailure: If tables are missing or database access is unavailable.
        """
        if not self._settings.ai_admission_enabled:
            return
        with self._store.transaction() as connection:
            rows = connection.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s",
                (self._store.schema,),
            ).fetchall()
            if not ADMISSION_TABLES.issubset({row["tablename"] for row in rows}):
                raise AgentFailure(ErrorCode.STATE_UNINITIALIZED)

    @contextmanager
    def turn(self, identifier: UUID | str) -> Iterator[None]:
        """Reserve one lifetime session request and hold a global inference slot.

        Authorization must complete before entering. Admitted requests are charged
        even if provider or tool execution subsequently fails. Rejections do not
        submit a provider request. No automatic waiting or retry is performed.

        Args:
            identifier: Previously authorized conversation UUID, never logged.

        Yields:
            Permission to execute the turn under its separate accounting context.

        Raises:
            AgentFailure: For exhausted budgets, busy slots or database failures.
        """
        if not self._settings.ai_admission_enabled:
            yield
            return
        try:
            with Connection.connect(
                conninfo="",
                autocommit=True,
                row_factory=dict_row,
                connect_timeout=5,
                options=f"-csearch_path={self._store.schema} "
                "-clock_timeout=5000 -cstatement_timeout=10000",
            ) as connection:
                slot = self._acquire_slot(connection)
                try:
                    self._reserve_request(connection, identifier)
                    context = self._connection.set(connection)
                    try:
                        yield
                    finally:
                        self._connection.reset(context)
                finally:
                    connection.execute("SELECT pg_advisory_unlock(%s)", (slot,))
        except Error:
            raise AgentFailure(ErrorCode.DATABASE) from None

    def _acquire_slot(self, connection: Connection[DictRow]) -> int:
        """Acquire one non-blocking state-schema slot shared across Python processes.

        Args:
            connection: Dedicated autocommit connection retaining the slot.

        Returns:
            The acquired PostgreSQL advisory lock identifier.

        Raises:
            AgentFailure: If every configured slot is busy.
        """
        for number in range(self._settings.ai_max_concurrent_turns):
            row = connection.execute(
                "SELECT hashtextextended(%s, 0) AS slot",
                (f"music-store-inference:{self._store.schema}:{number}",),
            ).fetchone()
            slot = row["slot"]
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired", (slot,)
            ).fetchone()
            if acquired["acquired"]:
                return slot
        raise AgentFailure(ErrorCode.RATE_LIMIT)

    def _reserve_request(
        self, connection: Connection[DictRow], identifier: UUID | str
    ) -> None:
        """Charge a bounded session attempt only while daily inference remains open.

        Args:
            connection: Separate autocommit accounting connection.
            identifier: Authorized UUID with no associated contact fields.

        Raises:
            AgentFailure: If the session or daily allowance is exhausted.
        """
        daily = connection.execute(
            "SELECT model_calls FROM inference_daily_usage "
            "WHERE usage_day = (clock_timestamp() AT TIME ZONE 'UTC')::date"
        ).fetchone()
        if daily and daily["model_calls"] >= self._settings.ai_daily_model_call_limit:
            raise AgentFailure(ErrorCode.RATE_LIMIT)
        row = connection.execute(
            "INSERT INTO inference_session_usage (conversation_id, request_count) "
            "VALUES (%s, 1) ON CONFLICT (conversation_id) DO UPDATE "
            "SET request_count = inference_session_usage.request_count + 1 "
            "WHERE inference_session_usage.request_count < %s "
            "RETURNING request_count",
            (UUID(str(identifier)), self._settings.ai_session_request_limit),
        ).fetchone()
        if row is None:
            raise AgentFailure(ErrorCode.RATE_LIMIT)

    def model_call(self) -> None:
        """Commit one UTC daily model attempt before invoking the external handler.

        Counts are never refunded on timeout, cancellation or graph rollback.
        PostgreSQL atomically enforces the limit across concurrent processes.

        Raises:
            AgentFailure: If the budget is exhausted or accounting is unavailable.
        """
        if not self._settings.ai_admission_enabled:
            return
        connection = self._connection.get()
        if connection is None:
            raise AgentFailure(ErrorCode.CONFIGURATION)
        try:
            row = connection.execute(
                "INSERT INTO inference_daily_usage (usage_day, model_calls) "
                "VALUES ((clock_timestamp() AT TIME ZONE 'UTC')::date, 1) "
                "ON CONFLICT (usage_day) DO UPDATE "
                "SET model_calls = inference_daily_usage.model_calls + 1 "
                "WHERE inference_daily_usage.model_calls < %s RETURNING model_calls",
                (self._settings.ai_daily_model_call_limit,),
            ).fetchone()
            if row is None:
                raise AgentFailure(ErrorCode.RATE_LIMIT)
        except Error:
            raise AgentFailure(ErrorCode.DATABASE) from None
