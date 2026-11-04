"""Minimal browser-facing cookie operations; all conversation history stays private.

Release routing requires only /browser-session/* to reach the API under the same
HTTPS origin as Streamlit. Other /api/* paths remain internal to the Docker network.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.api.dependencies import AppResources, get_resources
from music_store_agent.api.errors import APIError
from music_store_agent.browser_session import (
    COOKIE_NAME,
    COOKIE_PATH,
    COOKIE_TTL,
    BrowserCredential,
    BrowserSessionSettings,
    seal,
    unseal,
)

router = APIRouter(prefix=COOKIE_PATH, include_in_schema=False)


class BrowserOperation(BaseModel):
    """Accept one bounded public revision, never an ID or authorization capability."""

    model_config = ConfigDict(extra="forbid", strict=True)
    revision: UUID = Field(strict=False)


def settings_for(request: Request) -> BrowserSessionSettings:
    """Enforce exact origin and a custom-header CSRF guard before any mutation.

    Args:
        request: Browser request; forwarded host/protocol headers are ignored.

    Returns:
        Explicit deployment settings.

    Raises:
        APIError: For invalid configuration, origin, query or CSRF headers.
    """
    try:
        settings = BrowserSessionSettings()
    except ValueError:
        raise APIError("configuration") from None
    if not settings.enabled:
        raise APIError("configuration")
    if (
        request.headers.get("origin") != settings.app_origin
        or request.headers.get("x-browser-session") != "1"
        or request.query_params
    ):
        raise APIError("unauthorized")
    return settings


def response_for(
    settings: BrowserSessionSettings, payload: dict[str, str]
) -> JSONResponse:
    """Expose only status/revision JSON with exact credentialed CORS and no caching.

    Args:
        settings: Validated trusted application origin.
        payload: Fixed status and optional public operation revision.

    Returns:
        A browser-readable response without plaintext authorization material.
    """
    return JSONResponse(
        payload,
        headers={
            "Access-Control-Allow-Origin": settings.app_origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
            "Cache-Control": "no-store",
        },
    )


@router.options("/{operation}")
def preflight(request: Request) -> JSONResponse:
    """Allow only the configured browser origin and bounded component request shape.

    Args:
        request: Browser CORS preflight.

    Returns:
        Exact credentialed CORS permissions without wildcard origins.

    Raises:
        APIError: For unsupported origin, methods or request headers.
    """
    try:
        settings = BrowserSessionSettings()
    except ValueError:
        raise APIError("configuration") from None
    headers = {
        value.strip().lower()
        for value in request.headers.get("access-control-request-headers", "").split(
            ","
        )
        if value.strip()
    }
    if (
        not settings.enabled
        or request.headers.get("origin") != settings.app_origin
        or request.headers.get("access-control-request-method") != "POST"
        or not headers <= {"content-type", "x-browser-session"}
    ):
        raise APIError("unauthorized")
    response = response_for(settings, {"status": "ready"})
    response.headers["Access-Control-Allow-Methods"] = "POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Browser-Session"
    return response


@router.post("/check")
def check(
    request: Request,
    settings: Annotated[BrowserSessionSettings, Depends(settings_for)],
    resources: Annotated[AppResources, Depends(get_resources)],
) -> JSONResponse:
    """Verify cookie possession against existing PostgreSQL digests without inference.

    Args:
        request: Browser cookie request; values are never logged.
        settings: Validated cookie and CSRF settings.
        resources: Existing application resource owner.

    Returns:
        Only ready/empty state and a public revision, never a conversation token.
    """
    credential = unseal(request.cookies.get(COOKIE_NAME), settings)
    if credential is None:
        return response_for(settings, {"status": "empty"})
    try:
        with resources.store().authorized(
            credential.conversation_id, credential.access_token.get_secret_value()
        ):
            verify_demo(resources.engine())
    except AgentFailure as error:
        if error.code != ErrorCode.SESSION_UNVERIFIED:
            raise
        return response_for(settings, {"status": "empty"})
    return response_for(
        settings, {"status": "ready", "revision": str(credential.revision)}
    )


@router.post("/start")
@router.post("/new")
def start(
    operation: BrowserOperation,
    request: Request,
    settings: Annotated[BrowserSessionSettings, Depends(settings_for)],
    resources: Annotated[AppResources, Depends(get_resources)],
) -> JSONResponse:
    """Issue an encrypted active capability through an actual browser HTTP response.

    Args:
        operation: Public revision for one explicit browser action.
        request: Browser request with an optional existing HttpOnly cookie.
        settings: Stable encryption and trusted origin configuration.
        resources: Existing PostgreSQL stores; no provider is constructed.

    Returns:
        Status/revision only, with an HttpOnly encrypted cookie when newly issued.
    """
    existing = unseal(request.cookies.get(COOKIE_NAME), settings)
    if existing is not None and (
        request.url.path.endswith("/start") or existing.revision == operation.revision
    ):
        try:
            with resources.store().authorized(
                existing.conversation_id, existing.access_token.get_secret_value()
            ):
                verify_demo(resources.engine())
            return response_for(
                settings, {"status": "ready", "revision": str(existing.revision)}
            )
        except AgentFailure as error:
            if error.code != ErrorCode.SESSION_UNVERIFIED:
                raise
    verify_demo(resources.engine())
    identifier, token = resources.store().create()
    credential = BrowserCredential(
        conversation_id=identifier, access_token=token, revision=operation.revision
    )
    response = response_for(
        settings, {"status": "ready", "revision": str(operation.revision)}
    )
    response.set_cookie(
        COOKIE_NAME,
        seal(credential, settings),
        max_age=COOKIE_TTL,
        httponly=True,
        secure=settings.secure,
        samesite="lax",
        path="/",
    )
    return response


@router.post("/clear")
def clear(
    settings: Annotated[BrowserSessionSettings, Depends(settings_for)],
) -> JSONResponse:
    """Clear only the browser reference; conversation deletion is separately confirmed.

    Args:
        settings: Validated origin and cookie configuration.

    Returns:
        A cookie-expiration response without deleting backend conversations.
    """
    response = response_for(settings, {"status": "empty"})
    response.delete_cookie(
        COOKIE_NAME, path="/", secure=settings.secure, httponly=True, samesite="lax"
    )
    return response
