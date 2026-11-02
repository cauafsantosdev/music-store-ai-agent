"""Authorized durable conversations using native transactional LangGraph state."""

import re
from collections.abc import Callable
from typing import Literal
from uuid import UUID

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context
from pydantic import BaseModel, SecretStr
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode, provider_failure
from music_store_agent.agent.memory import ConversationStore


class AgentReply(BaseModel, frozen=True):
    """Customer-facing text and safe execution status without raw internal traces."""

    text: str
    code: str = "ok"
    tool_calls: int = 0


class HistoryMessage(BaseModel, frozen=True):
    """Display text without tool payloads, system prompts, or hidden metadata."""

    role: Literal["user", "assistant"]
    text: str


def _validate_message(message: str) -> None:
    """Bound demonstration input and reject ordinary real email/phone values.

    This guard is not a general PII classifier. Private documents and real
    personal records must never be submitted through demonstration clients.

    Args:
        message: Untrusted user text.

    Raises:
        AgentFailure: If input is invalid, oversized, or contains real contacts.
    """
    if not isinstance(message, str) or not message.strip() or len(message) > 2000:
        raise AgentFailure(ErrorCode.INVALID_INPUT)
    for domain in re.findall(r"[^\s@]+@([\w.-]+)", message):
        if domain.casefold().rstrip(".") != "example.test":
            raise AgentFailure(ErrorCode.INVALID_INPUT)
    if re.search(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", message):
        raise AgentFailure(ErrorCode.INVALID_INPUT)


def _token_value(token: str | SecretStr) -> str:
    """Extract an unlogged capability at the authorization boundary.

    Args:
        token: Plain or redacted capability wrapper.

    Returns:
        The capability value.
    """
    return token.get_secret_value() if isinstance(token, SecretStr) else token


def _config(identifier: UUID | str) -> RunnableConfig:
    """Build native thread routing without credentials or sensitive metadata.

    Args:
        identifier: Authorized conversation UUID.

    Returns:
        Native checkpoint routing configuration.
    """
    return {"configurable": {"thread_id": str(UUID(str(identifier)))}}


def _messages(saver: PostgresSaver, identifier: UUID | str) -> list[BaseMessage]:
    """Read native messages only after capability authorization.

    Args:
        saver: Saver bound to the authorized transaction.
        identifier: Authorized UUID.

    Returns:
        Native messages retaining required provider metadata.
    """
    checkpoint = saver.get_tuple(_config(identifier))
    if checkpoint is None:
        return []
    return checkpoint.checkpoint["channel_values"].get("messages", [])


class Conversation:
    """Hold a redacted capability to a durable session without caching history."""

    conversation_id: UUID
    access_token: SecretStr
    _agent: "StoreAgent"

    def __init__(self, agent: "StoreAgent", identifier: UUID, token: str) -> None:
        """Retain only an authorized session handle.

        Args:
            agent: Owning service.
            identifier: Unique conversation UUID.
            token: Opaque capability, redacted in representations.
        """
        self._agent = agent
        self.conversation_id = identifier
        self.access_token = SecretStr(token)

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        """Return sanitized native copies for local compatibility inspection.

        Persisted state retains all metadata. Customer clients should use
        get_conversation_history(), which also omits all tool payloads.

        Returns:
            Authorized copies without hidden metadata or tool artifacts.
        """
        return self._agent.native_history(self.conversation_id, self.access_token)

    def send(self, message: str) -> AgentReply:
        """Send a message through the capability-authorized durable service.

        Args:
            message: Non-sensitive synthetic input.

        Returns:
            A completed response or sanitized failure.
        """
        return self._agent.send_message(
            self.conversation_id, self.access_token, message
        )


class StoreAgent:
    """Manage durable sessions with PostgreSQL locks and official checkpoints."""

    def __init__(
        self,
        engine: Engine,
        build_graph: Callable[[PostgresSaver], CompiledStateGraph],
        settings: AgentSettings,
        store: ConversationStore,
        *,
        owns_store: bool = False,
    ) -> None:
        """Retain explicit dependencies without loading conversation history.

        Args:
            engine: Verified synthetic operational engine.
            build_graph: Bind create_agent to the current transaction's saver.
            settings: Bounded runtime configuration.
            store: Initialized state store in a separate schema.
            owns_store: Whether close() must release the store.
        """
        self._engine, self._build_graph = engine, build_graph
        self._settings, self._store = settings, store
        self._owns_store, self._closed = owns_store, False

    def __enter__(self) -> "StoreAgent":
        """Return the explicitly managed service.

        Returns:
            This agent service.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close owned resources on normal exit or interruption.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception instance.
            traceback: Optional traceback, deliberately not logged.
        """
        self.close()

    def close(self) -> None:
        """Release an owned pool without closing externally managed stores."""
        if self._owns_store:
            self._store.close()
        self._closed = True

    def _ensure_open(self) -> None:
        """Reject use after service shutdown.

        Raises:
            AgentFailure: If the service is closed.
        """
        if self._closed:
            raise AgentFailure(ErrorCode.DATABASE)

    def create_conversation(self) -> Conversation:
        """Create a UUID session and fresh unlogged access capability.

        Returns:
            A handle containing the UUID and redacted token.
        """
        self._ensure_open()
        verify_demo(self._engine)
        identifier, token = self._store.create()
        return Conversation(self, identifier, token)

    def new_conversation(self) -> Conversation:
        """Preserve the original creation interface using durable storage.

        Returns:
            A new durable session handle.
        """
        return self.create_conversation()

    def resume_conversation(
        self, conversation_id: UUID | str, access_token: str | SecretStr
    ) -> Conversation:
        """Authorize a prior session before reading checkpoint state.

        Args:
            conversation_id: Untrusted UUID.
            access_token: Matching capability.

        Returns:
            An authorized handle without duplicating saved history.

        Raises:
            AgentFailure: Uniformly for unknown IDs and invalid capabilities.
        """
        self._ensure_open()
        token = _token_value(access_token)
        with self._store.authorized(conversation_id, token):
            return Conversation(self, UUID(str(conversation_id)), token)

    def send_message(
        self, conversation_id: UUID | str, access_token: str | SecretStr, message: str
    ) -> AgentReply:
        """Atomically commit a complete turn under a cross-process row lock.

        The saver shares the outer transaction connection. Intermediate checkpoints,
        pending writes and counters roll back on failure or process death. Recovery
        starts from the last completed turn rather than replaying incomplete tools.

        Args:
            conversation_id: Target UUID.
            access_token: Matching capability.
            message: Synthetic input; only this new message is passed to the graph.

        Returns:
            Success only after commit; failure leaves prior history intact.
        """
        try:
            self._ensure_open()
            _validate_message(message)
            with self._store.authorized(
                conversation_id, _token_value(access_token)
            ) as connection:
                verify_demo(self._engine)
                saver = self._store.saver(connection)
                previous = _messages(saver, conversation_id)
                if len(previous) + 2 > self._settings.ai_max_history_messages:
                    raise AgentFailure(ErrorCode.EXECUTION_LIMIT)
                config = _config(conversation_id)
                config.update(
                    recursion_limit=self._settings.ai_recursion_limit, callbacks=[]
                )
                with tracing_context(enabled=False):
                    result = self._build_graph(saver).invoke(
                        {"messages": [HumanMessage(content=message.strip())]},
                        config=config,
                    )
                messages = result["messages"]
                last = messages[-1]
                if (
                    not isinstance(last, AIMessage)
                    or last.tool_calls
                    or not last.text.strip()
                ):
                    raise AgentFailure(ErrorCode.EMPTY_RESPONSE)
                if len(messages) > self._settings.ai_max_history_messages:
                    raise AgentFailure(ErrorCode.EXECUTION_LIMIT)
                calls = sum(
                    isinstance(item, ToolMessage) for item in messages[len(previous) :]
                )
                reply = AgentReply(text=last.text, tool_calls=calls)
                connection.execute(
                    "UPDATE conversations SET updated_at = clock_timestamp() "
                    "WHERE conversation_id = %s",
                    (UUID(str(conversation_id)),),
                )
            return reply
        except (
            GraphRecursionError,
            ModelCallLimitExceededError,
            ToolCallLimitExceededError,
        ):
            error = AgentFailure(ErrorCode.EXECUTION_LIMIT)
        except AgentFailure as failure:
            error = failure
        except Exception as failure:
            error = provider_failure(failure)
        return AgentReply(text=str(error), code=error.code.value)

    def native_history(
        self, conversation_id: UUID | str, access_token: str | SecretStr
    ) -> tuple[BaseMessage, ...]:
        """Read compatibility copies without mutating persisted provider metadata.

        Args:
            conversation_id: Target UUID.
            access_token: Matching capability.

        Returns:
            Native display copies with hidden metadata removed.
        """
        self._ensure_open()
        with self._store.authorized(
            conversation_id, _token_value(access_token)
        ) as connection:
            copies: list[BaseMessage] = []
            for message in _messages(self._store.saver(connection), conversation_id):
                item = message.model_copy(deep=True)
                item.additional_kwargs, item.response_metadata = {}, {}
                if isinstance(item, AIMessage):
                    item.usage_metadata = None
                    item.tool_calls = [
                        {
                            key: call[key]
                            for key in ("name", "args", "id", "type")
                            if key in call
                        }
                        for call in item.tool_calls
                    ]
                if isinstance(item, ToolMessage):
                    item.artifact = None
                copies.append(item)
            return tuple(copies)

    def get_conversation_history(
        self, conversation_id: UUID | str, access_token: str | SecretStr
    ) -> tuple[HistoryMessage, ...]:
        """Return authorized display text without system, tool or provider details.

        Args:
            conversation_id: Target UUID.
            access_token: Matching capability.

        Returns:
            Human and final assistant text only.
        """
        return tuple(
            HistoryMessage(
                role="user" if isinstance(item, HumanMessage) else "assistant",
                text=item.text,
            )
            for item in self.native_history(conversation_id, access_token)
            if isinstance(item, HumanMessage)
            or (
                isinstance(item, AIMessage)
                and not item.tool_calls
                and item.text.strip()
            )
        )

    def delete_conversation(
        self, conversation_id: UUID | str, access_token: str | SecretStr
    ) -> None:
        """Delete only authorized metadata and associated native checkpoints.

        Args:
            conversation_id: Target UUID.
            access_token: Matching capability.
        """
        self._ensure_open()
        self._store.delete(conversation_id, _token_value(access_token))
