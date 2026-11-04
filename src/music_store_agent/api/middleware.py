"""Bound requests and correlate safe metrics without buffering SSE."""

from time import perf_counter
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from music_store_agent.api.errors import APIError, error_response
from music_store_agent.observability import REQUEST_ID, emit

# Allow 2000 characters even when JSON encodes non-BMP values as surrogate pairs.
MAX_BODY_BYTES: int = 32_768


class RequestGuard:
    """Enforce a small body limit before parsing or invoking any service dependency."""

    def __init__(self, app: ASGIApp) -> None:
        """Retain the wrapped ASGI application.

        Args:
            app: Application beneath the HTTP guard.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Generate request correlation and measure complete ASGI delivery.

        Caller-supplied request IDs are deliberately ignored. Context propagates
        into supported worker offloading and is reset even after cancellation.

        Args:
            scope: ASGI connection scope; paths and queries are never logged.
            receive: Incoming ASGI messages.
            send: Outgoing frames, forwarded without retaining response bodies.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        identifier = str(uuid4())
        context = REQUEST_ID.set(identifier)
        started = perf_counter()
        status: int | None = None
        completed = False

        async def correlated_send(message: Message) -> None:
            """Add one correlation header and track actual response completion.

            Args:
                message: Current outgoing frame, never logged or buffered.
            """
            nonlocal status, completed
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                message = {
                    **message,
                    "headers": headers
                    + [(b"x-request-id", identifier.encode("ascii"))],
                }
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                completed = True

        try:
            await self._http(scope, receive, correlated_send)
        finally:
            try:
                route = getattr(scope.get("route"), "path", "unmatched")
                emit(
                    "http_request",
                    method=scope.get("method", "OTHER"),
                    route=route,
                    status_code=status,
                    duration_ms=(perf_counter() - started) * 1000,
                    delivery="completed" if completed else "interrupted",
                )
            finally:
                REQUEST_ID.reset(context)

    async def _http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check request size and append no-store headers without logging payloads.

        Args:
            scope: ASGI connection scope.
            receive: Incoming ASGI messages.
            send: Outgoing ASGI messages.
        """
        path = scope.get("path", "")
        protected = path.startswith(("/api/", "/browser-session/")) or path == "/ready"
        started = False

        async def safe_send(message: Message) -> None:
            """Append cache controls without retaining response content.

            Args:
                message: Current outgoing ASGI frame.
            """
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                if protected:
                    headers = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower() not in (b"cache-control", b"pragma", b"vary")
                    ]
                    message = {
                        **message,
                        "headers": headers
                        + [
                            (b"cache-control", b"no-store"),
                            (b"pragma", b"no-cache"),
                            (
                                b"vary",
                                b"Origin, Authorization"
                                if path.startswith("/browser-session/")
                                else b"Authorization",
                            ),
                        ],
                    }
            await send(message)

        try:
            incoming = receive
            if (
                path.startswith(("/api/", "/browser-session/"))
                and scope.get("method") == "POST"
            ):
                chunks: list[bytes] = []
                size = 0
                for key, value in scope.get("headers", []):
                    if key.lower() == b"content-length":
                        try:
                            size_hint = int(value)
                        except ValueError:
                            raise APIError("invalid_request") from None
                        if size_hint < 0:
                            raise APIError("invalid_request")
                        if size_hint > MAX_BODY_BYTES:
                            raise APIError("request_too_large")
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > MAX_BODY_BYTES:
                        raise APIError("request_too_large")
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
                body = b"".join(chunks)
                delivered = False

                async def buffered_receive() -> Message:
                    """Replay only the bounded request body, then delegate disconnects.

                    Returns:
                        A bounded body frame or the actual next incoming frame.
                    """
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await receive()

                incoming = buffered_receive
            await self.app(scope, incoming, safe_send)
        except APIError as error:
            if not started:
                await error_response(error)(scope, receive, safe_send)
        except Exception:
            # Unexpected programming failures are 500, never successful replies.
            # Do not rethrow raw exceptions into server request logs.
            if not started:
                await error_response(APIError("internal_error"))(
                    scope, receive, safe_send
                )
