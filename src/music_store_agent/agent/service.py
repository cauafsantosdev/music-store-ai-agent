"""Process-local conversations preserving native tool-call message metadata."""

import re
from threading import Lock

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context
from pydantic import BaseModel
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import (
    PUBLIC_MESSAGES,
    AgentFailure,
    ErrorCode,
    provider_failure,
)


class AgentReply(BaseModel, frozen=True):
    """Customer-facing text and safe execution status without raw internal traces."""

    text: str
    code: str = "ok"
    tool_calls: int = 0


def _validate_message(message: str) -> None:
    """Bound demonstration text and reject ordinary real email/phone inputs.

    This is a development guard, not a general PII classifier. Callers must never
    paste private documents or real personal records into the demonstration CLI.

    Args:
        message: Untrusted user text.

    Raises:
        AgentFailure: If text is empty, oversized, non-text, or contains real contacts.
    """
    if not isinstance(message, str) or not message.strip() or len(message) > 2000:
        raise AgentFailure(ErrorCode.INVALID_INPUT)
    for domain in re.findall(r"[^\s@]+@([\w.-]+)", message):
        if domain.casefold().rstrip(".") != "example.test":
            raise AgentFailure(ErrorCode.INVALID_INPUT)
    if re.search(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", message):
        raise AgentFailure(ErrorCode.INVALID_INPUT)


class Conversation:
    """Own one native-message history with no database checkpoints or disk logs."""

    _engine: Engine
    _graph: CompiledStateGraph
    _settings: AgentSettings
    _messages: list[BaseMessage]
    _lock: Lock

    def __init__(
        self, engine: Engine, graph: CompiledStateGraph, settings: AgentSettings
    ) -> None:
        """Start an isolated empty conversation.

        Args:
            engine: Verified synthetic data engine.
            graph: Stateless create_agent graph shared without checkpointers.
            settings: Validated execution and history limits.
        """
        self._engine, self._graph, self._settings = engine, graph, settings
        self._messages = []
        self._lock = Lock()

    @property
    def messages(self) -> tuple[BaseMessage, ...]:
        """Return defensive native-message copies for local inspection.

        Returns:
            Messages preserving tool-call IDs, signatures, and provider metadata.
        """
        return tuple(message.model_copy(deep=True) for message in self._messages)

    def send(self, message: str) -> AgentReply:
        """Execute one bounded turn and commit history only after a complete response.

        Args:
            message: Non-sensitive synthetic-demo text in a single human turn.

        Returns:
            A response in the requested persona or a sanitized Portuguese failure.
            Failed turns leave prior history intact; tool-call sequences are never
            truncated or reconstructed from flattened text/JSON.
        """
        with self._lock:
            try:
                _validate_message(message)
                # Reserve the smallest complete human/assistant exchange up front.
                if len(self._messages) + 2 > self._settings.ai_max_history_messages:
                    raise AgentFailure(ErrorCode.EXECUTION_LIMIT)
                verify_demo(self._engine)
                previous = [item.model_copy(deep=True) for item in self._messages]
                proposed = previous + [HumanMessage(content=message.strip())]
                # Explicitly disable tracing even if a developer has tracing env vars.
                with tracing_context(enabled=False):
                    result = self._graph.invoke(
                        {"messages": proposed},
                        config={
                            "recursion_limit": self._settings.ai_recursion_limit,
                            "callbacks": [],
                        },
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
                self._messages = [item.model_copy(deep=True) for item in messages]
                calls = sum(
                    isinstance(item, ToolMessage) for item in messages[len(previous) :]
                )
                return AgentReply(text=last.text, tool_calls=calls)
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
            return AgentReply(text=PUBLIC_MESSAGES[error.code], code=error.code.value)


class StoreAgent:
    """Create independent in-process conversations around one stateless agent graph."""

    _engine: Engine
    _graph: CompiledStateGraph
    _settings: AgentSettings

    def __init__(
        self, engine: Engine, graph: CompiledStateGraph, settings: AgentSettings
    ) -> None:
        """Retain explicit runtime dependencies without creating conversation state.

        Args:
            engine: Verified synthetic schema engine.
            graph: Compiled LangGraph runtime returned by create_agent.
            settings: Validated execution limits.
        """
        self._engine, self._graph, self._settings = engine, graph, settings

    def new_conversation(self) -> Conversation:
        """Start a new conversation without messages from any prior conversation.

        Returns:
            An isolated process-local conversation.
        """
        return Conversation(self._engine, self._graph, self._settings)
