"""Offline frontend HTTP contracts, timeout bounds and complete-event SSE parsing."""

import json
from collections.abc import Iterator
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from web.client import (
    BackendClient,
    Capability,
    ClientError,
    FrontendSettings,
    parse_sse,
)


def capability() -> Capability:
    """Create synthetic transport credentials without any backend side effect.

    Returns:
        A redacted capability used only by offline tests.
    """
    return Capability.model_validate_json(
        json.dumps(
            {
                "conversation_id": str(uuid4()),
                "access_token": "synthetic-capability-canary",
            }
        )
    )


def frame(name: str, data: dict[str, object]) -> str:
    """Build a synthetic SSE frame matching the public API contract.

    Args:
        name: Controlled event name.
        data: Public synthetic event payload.

    Returns:
        A correctly terminated frame.
    """
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def success_stream() -> str:
    """Provide a complete response rather than artificial token fragments.

    Returns:
        A processing event, a whole Markdown reply and a completion event.
    """
    return (
        frame("status", {"status": "processing"})
        + frame(
            "message",
            {
                "text": "**Olá!** Informação fictícia.",
                "code": "ok",
                "tool_calls": 1,
            },
        )
        + frame("done", {"status": "completed"})
    )


class ByteChunks(httpx.SyncByteStream):
    """Split UTF-8 across individual network bytes to exercise HTTPX line decoding."""

    def __init__(self, content: str) -> None:
        """Retain synthetic wire content.

        Args:
            content: Complete synthetic SSE data.
        """
        self.content = content.encode()

    def __iter__(self) -> Iterator[bytes]:
        """Yield byte chunks unrelated to SSE frame or Unicode boundaries.

        Yields:
            One network byte at a time.
        """
        for value in self.content:
            yield bytes([value])


def test_default_timeouts_and_internal_backend_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep status/connect deadlines short and multi-iteration response waits bounded.

    Args:
        monkeypatch: Scoped removal of optional deployment overrides.
    """
    for name in (
        "BACKEND_URL",
        "BACKEND_TIMEOUT_SECONDS",
        "BACKEND_READ_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = FrontendSettings()
    assert str(settings.backend_url) == "http://api:8000/"
    timeout = settings.timeout()
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (
        5,
        180,
        10,
        5,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend_read_timeout_seconds": 601},
        {"backend_read_timeout_seconds": 0},
        {"backend_timeout_seconds": 0},
        {"backend_timeout_seconds": 31},
        {"backend_url": "http://user:secret@api:8000"},
        {"backend_url": "http://api:8000?access_token=secret"},
    ],
)
def test_invalid_frontend_configuration_is_rejected(
    overrides: dict[str, object],
) -> None:
    """Reject infinite waits and loggable credentials in configured URLs.

    Args:
        overrides: Invalid explicit frontend settings.
    """
    with pytest.raises(ValidationError):
        FrontendSettings(**overrides)


def test_sse_comments_crlf_multiline_and_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse full event framing over byte boundaries and invoke submission once.

    Args:
        monkeypatch: Scoped fake HTTP transport.
    """
    requests: list[httpx.Request] = []
    cap = capability()

    def respond(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Return a byte-fragmented SSE response without network access.

        Args:
            self: Actual frontend HTTP transport.
            request: Synthetic authenticated request.

        Returns:
            Complete framed SSE data with a comment and CRLF line endings.
        """
        requests.append(request)
        assert request.url.host == "api"
        assert not request.url.query
        assert request.headers["authorization"] == "Bearer synthetic-capability-canary"
        assert json.loads(request.content) == {"message": "Uma pergunta fictícia."}
        content = (": keepalive\n\n" + success_stream()).replace("\n", "\r\n")
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=ByteChunks(content),
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    statuses: list[bool] = []
    with BackendClient(FrontendSettings()) as client:
        result = client.stream_reply(
            cap, "Uma pergunta fictícia.", lambda: statuses.append(True)
        )
    assert result.text == "**Olá!** Informação fictícia."
    assert result.tool_calls == 1
    assert statuses == [True]
    assert len(requests) == 1 and requests[0].url.path.endswith("/stream")
    assert list(
        parse_sse(["event: status", 'data: {"status":', 'data: "processing"}', ""])
    ) == [("status", '{"status":\n"processing"}')]


@pytest.mark.parametrize(
    "body",
    [
        success_stream().split("event: done")[0],
        frame("message", {"text": "invalid order", "code": "ok", "tool_calls": 0}),
        frame("status", {"status": "processing"})
        + "event: message\ndata: not-json\n\n",
        frame("status", {"status": "processing"})
        + frame("done", {"status": "completed"}),
        frame("status", {"status": "processing"})
        + frame("message", {"text": " ", "code": "ok", "tool_calls": 0})
        + frame("done", {"status": "completed"}),
        'event: status\ndata: {"status":"processing"}',
    ],
)
def test_incomplete_or_malformed_sse_never_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    body: str,
) -> None:
    """Treat unconfirmed delivery as ambiguous without a second POST or fallback.

    Args:
        monkeypatch: Scoped fake transport.
        body: Truncated or malformed SSE content.
    """
    calls: list[str] = []

    def respond(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Supply malformed public protocol content without outbound traffic.

        Args:
            self: Frontend transport.
            request: Submitted synthetic request.

        Returns:
            Controlled SSE response.
        """
        calls.append(request.url.path)
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with (
        BackendClient(FrontendSettings()) as client,
        pytest.raises(ClientError) as raised,
    ):
        client.stream_reply(capability(), "Olá.", lambda: None)
    assert raised.value.ambiguous
    assert len(calls) == 1 and calls[0].endswith("/stream")


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "session"),
        (404, "session"),
        (429, "rate_limit"),
        (503, "unavailable"),
        (504, "provider_timeout"),
        (422, "invalid_input"),
        (500, "unavailable"),
    ],
)
@pytest.mark.parametrize("sse_error", [False, True])
def test_http_and_sse_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    code: str,
    sse_error: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ignore diagnostic text and credentials in backend failure payloads.

    Args:
        monkeypatch: Scoped fake transport.
        status: Public HTTP status.
        code: Expected fixed frontend feedback category.
        sse_error: Whether failure arrives after SSE headers.
        caplog: Captured logs.
    """

    def respond(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Return sensitive synthetic diagnostics that must never be displayed.

        Args:
            self: Frontend transport.
            request: Synthetic request.

        Returns:
            A controlled HTTP error or completed SSE failure sequence.
        """
        error = {
            "error": {
                "code": "private-marker",
                "message": "synthetic-capability-canary private-marker",
            }
        }
        if sse_error:
            body = (
                frame("status", {"status": "processing"})
                + frame("error", {**error, "status_code": status})
                + frame("done", {"status": "failed"})
            )
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        return httpx.Response(status, request=request, json=error)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with (
        BackendClient(FrontendSettings()) as client,
        pytest.raises(ClientError) as raised,
    ):
        client.stream_reply(capability(), "Olá.", lambda: None)
    assert raised.value.code == code
    assert "private-marker" not in str(raised.value) + caplog.text
    assert "synthetic-capability-canary" not in str(raised.value) + caplog.text


@pytest.mark.parametrize(
    "failure_type",
    [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectError],
)
def test_transport_failures_do_not_retry_or_leak_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[httpx.HTTPError],
) -> None:
    """Keep ambiguous transactional outcomes explicit and outbound attempts bounded.

    Args:
        monkeypatch: Scoped failing transport.
        failure_type: Real HTTPX transport exception category.
    """
    calls: list[str] = []

    def fail(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Raise a controlled transport failure before any real network access.

        Args:
            self: Frontend transport.
            request: Submitted synthetic request.

        Raises:
            httpx.HTTPError: Always, with private synthetic diagnostic text.
        """
        calls.append(request.url.path)
        raise failure_type("private-transport-marker", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    with (
        BackendClient(FrontendSettings()) as client,
        pytest.raises(ClientError) as raised,
    ):
        client.stream_reply(capability(), "Olá.", lambda: None)
    assert raised.value.ambiguous
    assert "private-transport-marker" not in str(raised.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "reference,accepted",
    [
        ("2eac6ff4-e1aa-4372-aae0-705fc6512280", True),
        ("private-header-marker" * 100, False),
        ("2eac6ff4-e1aa-1372-aae0-705fc6512280", False),
    ],
)
def test_response_request_references_are_bounded_and_validated(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    accepted: bool,
) -> None:
    """Retain only UUID4 correlation metadata without exposing arbitrary error headers.

    Args:
        monkeypatch: Controlled HTTP transport without network access.
        reference: Valid correlation or untrusted response header value.
        accepted: Whether the header is safe for display as a support reference.
    """

    def respond(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Return a sanitized HTTP failure with controlled correlation metadata.

        Args:
            transport: Unused HTTP transport.
            request: Synthetic SSE request.

        Returns:
            A provider-unavailable response, without raw error data in feedback.
        """
        return httpx.Response(
            503,
            request=request,
            headers={"X-Request-ID": reference},
            json={"error": {"message": "private-error-marker"}},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with BackendClient(FrontendSettings()) as client:
        with pytest.raises(ClientError) as raised:
            client.stream_reply(capability(), "Consulta fictícia.", lambda: None)
        assert client.request_id == (reference if accepted else None)
    assert "private-" not in str(raised.value)


def test_transport_failure_cannot_reuse_the_creation_request_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid attributing a failed message to the preceding creation request.

    Args:
        monkeypatch: Controlled creation response followed by a transport failure.
    """
    reference = "2eac6ff4-e1aa-4372-aae0-705fc6512280"
    cap = capability()

    def respond(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Create one synthetic capability, then simulate a pre-header disconnect.

        Args:
            transport: Unused HTTP transport.
            request: Explicit synthetic creation or message request.

        Returns:
            The controlled creation response.

        Raises:
            httpx.ReadTimeout: For the subsequent message request only.
        """
        if request.url.path == "/api/v1/conversations":
            return httpx.Response(
                201,
                request=request,
                headers={"X-Request-ID": reference},
                json={
                    "conversation_id": str(cap.conversation_id),
                    "access_token": cap.access_token.get_secret_value(),
                },
            )
        raise httpx.ReadTimeout("private-disconnect-marker", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    with BackendClient(FrontendSettings()) as client:
        created = client.create_conversation()
        assert client.request_id == reference
        with pytest.raises(ClientError):
            client.stream_reply(created, "Consulta fictícia.", lambda: None)
        assert client.request_id is None
