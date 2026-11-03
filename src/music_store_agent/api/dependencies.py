"""Lazy per-application resource ownership and bearer capability dependencies."""

import re
from threading import RLock
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings, load_settings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_live_model, create_store_agent
from music_store_agent.agent.memory import ConversationStore, create_state_store
from music_store_agent.agent.service import Conversation, StoreAgent
from music_store_agent.api.errors import APIError
from music_store_agent.database.session import create_database_engine

bearer = HTTPBearer(
    auto_error=False, description="Demonstration conversation capability."
)


class AppResources:
    """Own lazy reusable resources without opening dependencies during startup."""

    def __init__(
        self,
        settings: AgentSettings | None = None,
        model: BaseChatModel | None = None,
        engine: Engine | None = None,
        store: ConversationStore | None = None,
    ) -> None:
        """Retain explicit injections; only internally created resources are owned.

        Args:
            settings: Optional validated settings, otherwise loaded lazily.
            model: Controlled model for offline tests; omitted for the real provider.
            engine: Optional externally owned synthetic operational engine.
            store: Optional externally owned initialized checkpoint store.
        """
        self._settings, self._model = settings, model
        self._engine, self._store = engine, store
        self._owns_engine, self._owns_store = engine is None, store is None
        self._owns_model = model is None
        self._service: StoreAgent | None = None
        # This lock protects lazy construction only, never conversation execution.
        self._initialization_lock = RLock()

    def settings(self) -> AgentSettings:
        """Read validated settings lazily, independently of application liveness.

        Returns:
            Validated deployment configuration.
        """
        with self._initialization_lock:
            if self._settings is None:
                self._settings = load_settings()
            return self._settings

    def engine(self) -> Engine:
        """Create one lazy operational engine per application lifespan.

        Returns:
            The shared synthetic-data engine.
        """
        with self._initialization_lock:
            if self._engine is None:
                self._engine = create_database_engine(
                    schema=self.settings().ai_demo_schema
                )
            return self._engine

    def store(self) -> ConversationStore:
        """Open only an explicitly initialized checkpoint schema without migrations.

        Returns:
            The reusable state pool.
        """
        with self._initialization_lock:
            if self._store is None:
                self._store = create_state_store(self.settings().ai_state_schema)
            return self._store

    def ready(self) -> None:
        """Check database, demo provenance, state initialization and key presence.

        No model construction or generation is required. Injected offline models
        replace the credential requirement without weakening data verification.

        Raises:
            AgentFailure: If an operational dependency is unavailable or unverified.
        """
        verify_demo(self.engine())
        self.store().check_ready()
        key = self.settings().groq_api_key
        if self._model is None and (key is None or not key.get_secret_value().strip()):
            raise AgentFailure(ErrorCode.MISSING_API_KEY)

    def service(self) -> StoreAgent:
        """Build one reusable service and provider client without generation calls.

        Returns:
            The existing durable StoreAgent application service.
        """
        with self._initialization_lock:
            if self._service is None:
                verify_demo(self.engine())
                store = self.store()
                if self._model is None:
                    self._model = create_live_model(self.settings())
                self._service = create_store_agent(
                    self.engine(),
                    settings=self.settings(),
                    model=self._model,
                    state_store=store,
                )
            return self._service

    def _close_sync(self) -> None:
        """Close owned synchronous clients, pools and engines on shutdown."""
        try:
            if self._service is not None:
                self._service.close()
            if self._owns_model and isinstance(self._model, ChatGroq):
                self._model.client._client.close()
        finally:
            try:
                if self._owns_store and self._store is not None:
                    self._store.close()
            finally:
                if self._owns_engine and self._engine is not None:
                    self._engine.dispose()

    async def aclose(self) -> None:
        """Close owned asynchronous and synchronous resources without blocking ASGI."""
        try:
            if self._owns_model and isinstance(self._model, ChatGroq):
                await self._model.async_client._client.close()
        finally:
            await run_in_threadpool(self._close_sync)


def get_resources(request: Request) -> AppResources:
    """Resolve resources belonging to the active application lifespan.

    Args:
        request: Current HTTP request.

    Returns:
        The application-owned lazy resource container.
    """
    return request.app.state.resources


def get_agent(resources: Annotated[AppResources, Depends(get_resources)]) -> StoreAgent:
    """Resolve the shared service in FastAPI's synchronous dependency thread pool.

    Args:
        resources: Application-scoped resources.

    Returns:
        A reusable durable application service.
    """
    return resources.service()


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> SecretStr:
    """Accept capabilities only from a well-formed bearer authorization header.

    Args:
        credentials: Framework-parsed Authorization header.

    Returns:
        A redacted capability passed only to existing service authorization.

    Raises:
        APIError: If the header is absent or malformed.
    """
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise APIError("unauthorized")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", credentials.credentials):
        raise APIError("unauthorized")
    return SecretStr(credentials.credentials)


def authorized_conversation(
    conversation_id: UUID,
    token: Annotated[SecretStr, Depends(require_token)],
    agent: Annotated[StoreAgent, Depends(get_agent)],
) -> Conversation:
    """Authorize before starting SSE or invoking any model work.

    Args:
        conversation_id: Validated path UUID.
        token: Bearer capability.
        agent: Existing durable service.

    Returns:
        An authorized internal handle, never serialized as an HTTP response.
    """
    return agent.resume_conversation(conversation_id, token)


def reject_query_parameters(request: Request) -> None:
    """Reject unsupported query fields rather than accepting loggable capabilities.

    Args:
        request: Current HTTP request.

    Raises:
        APIError: If any unsupported query parameter is present.
    """
    if request.query_params:
        raise APIError("invalid_request")
