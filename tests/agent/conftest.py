"""Offline fixtures using real PostgreSQL tools and controlled model output."""

import os
from collections.abc import Sequence

import pytest
from httpx import AsyncHTTPTransport, HTTPTransport
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import initialize_demo


class SimulatedProviderError(RuntimeError):
    """Expose a provider-style status code without making any external request."""

    status_code: int

    def __init__(self, code: int) -> None:
        """Create a controlled failure containing a diagnostic that must be hidden.

        Args:
            code: Simulated provider HTTP status.
        """
        self.status_code = code
        super().__init__("private-diagnostic-marker")


class ScriptedModel(FakeMessagesListChatModel):
    """Use real LangChain messages and tool orchestration with controlled responses."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)
    failure_code: int | None = None

    def bind_tools(
        self,
        tools: Sequence[object],
        *,
        tool_choice: str | None = None,
        **kwargs: object,
    ) -> Runnable:
        """Accept framework tool binding without replacing actual tool execution.

        Args:
            tools: Registered structured tool definitions.
            tool_choice: Optional framework selection hint.
            kwargs: Framework-specific binding options.

        Returns:
            This controlled model, preserving the real agent/tool graph.
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        """Record native input messages and generate a scripted reply or failure.

        Args:
            messages: Native framework messages including provider metadata.
            stop: Optional framework stop sequences.
            run_manager: Local LangChain callbacks.
            kwargs: Additional framework model options.

        Returns:
            The next scripted native chat result.

        Raises:
            SimulatedProviderError: If a provider status failure was requested.
        """
        self.seen.append([message.model_copy(deep=True) for message in messages])
        if self.failure_code is not None:
            raise SimulatedProviderError(self.failure_code)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture(autouse=True)
def block_live_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block real HTTP transports unless the explicit live test was enabled.

    Args:
        monkeypatch: Scoped SDK transport replacements for offline tests only.
    """
    if os.getenv("RUN_LIVE_GROQ") == "1":
        return

    def blocked(*args: object, **kwargs: object) -> None:
        """Reject accidental external generation calls during offline tests.

        Args:
            args: Any SDK positional inputs, deliberately not inspected.
            kwargs: Any SDK keyword inputs, deliberately not inspected.

        Raises:
            AssertionError: Always, because offline tests must not call GroqCloud.
        """
        raise AssertionError("External model requests are disabled in offline tests.")

    async def blocked_async(*args: object, **kwargs: object) -> None:
        """Reject accidental asynchronous transport calls in offline tests.

        Args:
            args: Uninspected transport positional inputs.
            kwargs: Uninspected transport keyword inputs.

        Raises:
            AssertionError: Always, because offline tests must not call GroqCloud.
        """
        blocked(*args, **kwargs)

    monkeypatch.setattr(HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(AsyncHTTPTransport, "handle_async_request", blocked_async)


@pytest.fixture
def demo_engine(database_engine: Engine) -> Engine:
    """Prepare a pinned synthetic dataset using the existing disposable schema fixture.

    Args:
        database_engine: Empty per-test namespace cleaned by the parent fixture.

    Returns:
        An engine verified against the application's pinned synthetic fingerprint.
    """
    initialize_demo(database_engine)
    return database_engine


@pytest.fixture
def offline_settings() -> AgentSettings:
    """Provide explicit model settings without reading any configured live credential.

    Returns:
        Validated bounded settings with no API key.
    """
    return AgentSettings(groq_api_key=None)
