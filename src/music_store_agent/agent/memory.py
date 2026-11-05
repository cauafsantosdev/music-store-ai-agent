"""Pooled PostgreSQL checkpoints and capability-protected conversation metadata."""

import hashlib
import hmac
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection, Error, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from music_store_agent.agent.datasets import CHALLENGE_FINGERPRINT, DatasetMode
from music_store_agent.agent.errors import AgentFailure, ErrorCode

REQUIRED_TABLES: frozenset[str] = frozenset(
    {
        "conversations",
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }
)


class ConversationStore:
    """Own a bounded pool restricted to one application-controlled state schema.

    Initialization is explicit. The pool never changes operational schemas and
    never falls back to public. Call close() or use the context manager on exit.
    """

    schema: str
    pool: ConnectionPool[Connection[DictRow]]
    dataset_mode: DatasetMode

    def __init__(
        self, schema: str = "agent_state", *, dataset_mode: DatasetMode = "sample"
    ) -> None:
        """Open the state pool without creating or migrating any tables.

        Args:
            schema: Trusted deployment setting, never a customer-provided value.
            dataset_mode: Capability digest domain; legacy conversations remain sample.

        Raises:
            AgentFailure: If the namespace is unsafe or PostgreSQL is unavailable.
        """
        if (
            dataset_mode not in ("sample", "challenge")
            or len(schema) > 63
            or not re.fullmatch(r"agent_state(?:_[a-z0-9_]+)?", schema)
        ):
            raise AgentFailure(ErrorCode.CONFIGURATION)
        self.schema = schema
        self.dataset_mode = dataset_mode
        self.pool = ConnectionPool(
            conninfo="",
            min_size=1,
            max_size=4,
            open=False,
            timeout=5,
            reconnect_timeout=5,
            num_workers=1,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "connect_timeout": 5,
                # No public fallback; pg_catalog remains implicitly available.
                "options": f"-csearch_path={schema} -clock_timeout=5000 "
                "-cstatement_timeout=10000",
            },
        )
        try:
            self.pool.open(wait=True, timeout=5)
        except (Error, PoolTimeout, PoolClosed):
            self.pool.close()
            raise AgentFailure(ErrorCode.DATABASE) from None

    def __enter__(self) -> "ConversationStore":
        """Return the explicitly managed state store.

        Returns:
            This state store.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close pooled resources after normal completion or an exception.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception instance.
            traceback: Optional traceback, deliberately not logged.
        """
        self.close()

    def close(self) -> None:
        """Release connections and worker threads; repeated calls are safe."""
        self.pool.close()

    def saver(self, connection: Connection[DictRow]) -> PostgresSaver:
        """Bind the official saver to the current operation's database transaction.

        Args:
            connection: Pooled connection with controlled search_path.

        Returns:
            A strict allowlisted saver with pickle fallback disabled.
        """
        return PostgresSaver(
            connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
                allowed_json_modules=None,
            ),
        )

    def initialize(self, *, admission: bool = False) -> None:
        """Create metadata and checkpoint tables explicitly and idempotently.

        setup() can create concurrent indexes and therefore runs in autocommit,
        under a schema-specific session advisory lock rather than a transaction.

        Args:
            admission: Also initialize persistent inference counters explicitly.

        Raises:
            AgentFailure: If initialization fails.
        """
        try:
            with self.pool.connection() as connection:
                connection.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                    (f"music-store-state-init:{self.schema}",),
                )
                try:
                    connection.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                            sql.Identifier(self.schema)
                        )
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS conversations ("
                        "conversation_id UUID PRIMARY KEY, "
                        "access_token_hash BYTEA NOT NULL "
                        "CHECK (octet_length(access_token_hash) = 32), "
                        "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                    )
                    self.saver(connection).setup()
                    if admission:
                        from music_store_agent.agent.admission import (
                            initialize_admission,
                        )

                        initialize_admission(connection)
                finally:
                    connection.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                        (f"music-store-state-init:{self.schema}",),
                    )
            self.check_ready()
        except (Error, PoolTimeout, PoolClosed):
            raise AgentFailure(ErrorCode.DATABASE) from None

    def check_ready(self) -> None:
        """Verify initialization without executing schema migrations.

        Raises:
            AgentFailure: If the schema is missing or PostgreSQL is unavailable.
        """
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s",
                (self.schema,),
            ).fetchall()
            if not REQUIRED_TABLES.issubset({row["tablename"] for row in rows}):
                raise AgentFailure(ErrorCode.STATE_UNINITIALIZED)

    @contextmanager
    def transaction(self) -> Iterator[Connection[DictRow]]:
        """Provide an atomic operation with bounded acquisition and SQL waits.

        Yields:
            An autocommit-configured connection inside an explicit transaction.

        Raises:
            AgentFailure: If acquisition, execution, or commit fails.
        """
        try:
            with self.pool.connection() as connection, connection.transaction():
                # A waiter must see the previous turn's commit after acquiring its lock.
                connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                yield connection
        except (Error, PoolTimeout, PoolClosed):
            raise AgentFailure(ErrorCode.DATABASE) from None

    def create(self) -> tuple[UUID, str]:
        """Persist a new UUID and only the SHA-256 digest of a random capability.

        Returns:
            The conversation UUID and a fresh 256-bit opaque access token.
        """
        identifier, token = uuid4(), secrets.token_urlsafe(32)
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO conversations (conversation_id, access_token_hash) "
                "VALUES (%s, %s)",
                (identifier, self._capability_digest(token)),
            )
        return identifier, token

    @contextmanager
    def authorized(
        self, identifier: UUID | str, token: str
    ) -> Iterator[Connection[DictRow]]:
        """Authorize and lock a conversation before accessing any checkpoint state.

        Args:
            identifier: Untrusted conversation identifier.
            token: Opaque capability; never included in SQL parameters or logs.

        Yields:
            The locked operation's transaction connection.

        Raises:
            AgentFailure: Uniformly for unknown identifiers and invalid capabilities.
        """
        try:
            parsed = UUID(str(identifier))
        except (ValueError, TypeError, AttributeError):
            raise AgentFailure(ErrorCode.SESSION_UNVERIFIED) from None
        if not isinstance(token, str) or len(token) > 256:
            raise AgentFailure(ErrorCode.SESSION_UNVERIFIED)
        digest = self._capability_digest(token)
        with self.transaction() as connection:
            # Reject invalid capabilities before lock contention can reveal whether
            # a conversation is busy. Recheck under lock to account for deletion.
            for suffix in ("", " FOR UPDATE"):
                row = connection.execute(
                    "SELECT access_token_hash FROM conversations "
                    "WHERE conversation_id = %s" + suffix,
                    (parsed,),
                ).fetchone()
                expected = bytes(row["access_token_hash"]) if row else bytes(32)
                verified = hmac.compare_digest(expected, digest)
                if row is None or not verified:
                    raise AgentFailure(ErrorCode.SESSION_UNVERIFIED)
            yield connection

    def _capability_digest(self, token: str) -> bytes:
        """Bind challenge capabilities to the approved dataset without altering tables.

        Sample hashes retain their original format. Challenge hashes use a fixed
        domain and baseline pin, preventing cross-dataset resume even if a deployment
        explicitly reuses a state schema. Raw tokens are never persisted.

        Args:
            token: Unlogged bearer capability.

        Returns:
            A 32-byte SHA-256 digest for constant-time authorization checks.
        """
        prefix = (
            f"music-store:challenge:{CHALLENGE_FINGERPRINT}\0"
            if self.dataset_mode == "challenge"
            else ""
        )
        return hashlib.sha256((prefix + token).encode()).digest()

    def delete(self, identifier: UUID | str, token: str) -> None:
        """Atomically remove authorized metadata and every native checkpoint row.

        Args:
            identifier: Conversation identifier.
            token: Corresponding access capability.
        """
        with self.authorized(identifier, token) as connection:
            self.saver(connection).delete_thread(str(UUID(str(identifier))))
            connection.execute(
                "DELETE FROM conversations WHERE conversation_id = %s",
                (UUID(str(identifier)),),
            )


def create_state_store(
    schema: str, *, dataset_mode: DatasetMode = "sample"
) -> ConversationStore:
    """Open and validate an initialized store without implicit DDL.

    Args:
        schema: Validated deployment-controlled namespace.
        dataset_mode: Explicit authorized operational dataset.

    Returns:
        An open store owned by the caller.

    Raises:
        AgentFailure: If initialization is missing or database access fails.
    """
    store = ConversationStore(schema, dataset_mode=dataset_mode)
    try:
        store.check_ready()
        return store
    except BaseException:
        store.close()
        raise
