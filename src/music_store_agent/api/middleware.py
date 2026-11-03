"""Bound request bodies and sanitize unexpected failures without buffering SSE."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from music_store_agent.api.errors import APIError, error_response

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
        """Check request size and append no-store headers without logging payloads.

        Args:
            scope: ASGI connection scope.
            receive: Incoming ASGI messages.
            send: Outgoing ASGI messages.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        protected = path.startswith("/api/") or path == "/ready"
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
                            (b"vary", b"Authorization"),
                        ],
                    }
            await send(message)

        try:
            incoming = receive
            if path.startswith("/api/") and scope.get("method") == "POST":
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
