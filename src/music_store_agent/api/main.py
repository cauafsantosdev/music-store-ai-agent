"""Versioned FastAPI application with lazy resources and independent liveness."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine
from starlette.exceptions import HTTPException

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api.conversations import router
from music_store_agent.api.dependencies import AppResources, get_resources
from music_store_agent.api.errors import (
    APIError,
    handle_agent_error,
    handle_api_error,
    handle_http_error,
    handle_validation_error,
)
from music_store_agent.api.middleware import RequestGuard
from music_store_agent.api.schemas import ReadinessResponse


def create_app(
    *,
    settings: AgentSettings | None = None,
    model: BaseChatModel | None = None,
    engine: Engine | None = None,
    store: ConversationStore | None = None,
) -> FastAPI:
    """Create a lazy application with optional real-service test dependencies.

    Args:
        settings: Optional validated configuration, loaded lazily when omitted.
        model: Controlled offline model; omitted for the existing Groq integration.
        engine: Optional externally owned synthetic-data engine.
        store: Optional externally owned initialized checkpoint store.

    Returns:
        An application that imports and starts without database or provider access.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Own resources for one serving lifespan without startup dependency checks.

        Args:
            application: Current FastAPI application.

        Yields:
            Control to the server; owned resources close after active requests finish.
        """
        resources = AppResources(settings, model, engine, store)
        application.state.resources = resources
        try:
            yield
        finally:
            await resources.aclose()

    application = FastAPI(title="Music Store AI Agent", lifespan=lifespan)
    application.add_middleware(RequestGuard)
    application.add_exception_handler(APIError, handle_api_error)
    application.add_exception_handler(AgentFailure, handle_agent_error)
    application.add_exception_handler(RequestValidationError, handle_validation_error)
    application.add_exception_handler(ResponseValidationError, handle_validation_error)
    application.add_exception_handler(HTTPException, handle_http_error)
    application.include_router(router)

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Report application liveness independently of credentials and PostgreSQL.

        Returns:
            The unchanged application liveness response.
        """
        return {"status": "ok"}

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={
            503: {
                "model": ReadinessResponse,
                "description": "Operational dependencies unavailable.",
            }
        },
    )
    async def ready(
        resources: Annotated[AppResources, Depends(get_resources)],
    ) -> JSONResponse:
        """Check operational readiness without any external model generation.

        Args:
            resources: Lazy application-owned dependencies.

        Returns:
            Ready status or a sanitized dependency failure with HTTP 503.
        """
        try:
            await run_in_threadpool(resources.ready)
            result = ReadinessResponse(status="ready")
            status = 200
        except AgentFailure as error:
            result = ReadinessResponse(
                status="not_ready", error=APIError(error.code).envelope().error
            )
            status = 503
        return JSONResponse(result.model_dump(exclude_none=True), status_code=status)

    return application


app = create_app()
