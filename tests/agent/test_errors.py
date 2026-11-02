"""Sanitized classification for actual Groq SDK and HTTP transport failures."""

import httpx
import pytest
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from music_store_agent.agent.errors import PUBLIC_MESSAGES, ErrorCode, provider_failure


def status_error(error_type: type[APIStatusError], status: int) -> APIStatusError:
    """Construct a real Groq failure with synthetic sensitive diagnostic values.

    Args:
        error_type: Concrete SDK HTTP exception class.
        status: Simulated provider HTTP status.

    Returns:
        A provider exception whose raw diagnostics must never reach public output.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return error_type(
        "private-diagnostic-marker",
        response=httpx.Response(status, request=request),
        body={"error": {"message": "private-response-body-marker"}},
    )


@pytest.mark.parametrize(
    "error",
    [
        status_error(InternalServerError, 504),
        status_error(APIStatusError, 408),
        httpx.ReadTimeout("private-diagnostic-marker"),
        httpx.ConnectTimeout("private-diagnostic-marker"),
        TimeoutError("private-diagnostic-marker"),
        APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_timeouts_are_distinct_and_sanitized(
    error: Exception,
    wrapped: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Classify direct and wrapped timeouts without logging diagnostic values.

    Args:
        error: Real SDK or transport exception containing synthetic diagnostics.
        wrapped: Whether an integration exception wraps the original failure.
        caplog: Captured log records.
        capsys: Captured standard output and error streams.
    """
    if wrapped:
        outer = RuntimeError("private-wrapper-marker")
        outer.__cause__ = error
        error = outer
    failure = provider_failure(error)
    assert failure.code == ErrorCode.PROVIDER_TIMEOUT
    assert str(failure) == PUBLIC_MESSAGES[ErrorCode.PROVIDER_UNAVAILABLE]
    assert "private" not in str(failure)
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("wrapped", [False, True])
def test_credit_exhaustion_is_distinct_from_rate_and_transport_failures(
    wrapped: bool,
) -> None:
    """Classify exhausted-credit responses without exposing account diagnostics.

    Args:
        wrapped: Whether an integration exception wraps the real Groq failure.
    """
    error: Exception = status_error(APIStatusError, 402)
    if wrapped:
        outer = RuntimeError("private-wrapper-marker")
        outer.__cause__ = error
        error = outer
    failure = provider_failure(error)
    assert failure.code == ErrorCode.QUOTA_EXHAUSTED
    assert str(failure) == PUBLIC_MESSAGES[ErrorCode.PROVIDER_UNAVAILABLE]
    assert "private" not in str(failure)


@pytest.mark.parametrize(
    "error,expected",
    [
        (status_error(AuthenticationError, 401), ErrorCode.AUTHENTICATION),
        (status_error(PermissionDeniedError, 403), ErrorCode.AUTHENTICATION),
        (status_error(RateLimitError, 429), ErrorCode.RATE_LIMIT),
        (status_error(BadRequestError, 400), ErrorCode.CONFIGURATION),
        (status_error(NotFoundError, 404), ErrorCode.CONFIGURATION),
        (status_error(UnprocessableEntityError, 422), ErrorCode.CONFIGURATION),
        (status_error(InternalServerError, 503), ErrorCode.PROVIDER_UNAVAILABLE),
        (
            APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")),
            ErrorCode.PROVIDER_UNAVAILABLE,
        ),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_real_groq_exceptions_are_sanitized(
    error: Exception, expected: ErrorCode, wrapped: bool
) -> None:
    """Use actual SDK status_code and exception types rather than message matching.

    Args:
        error: Real Groq exception containing only synthetic test diagnostics.
        expected: Safe application failure category.
        wrapped: Whether a runtime exception wraps the provider failure.
    """
    if wrapped:
        outer = RuntimeError("private-wrapper-marker")
        outer.__cause__ = error
        error = outer
    failure = provider_failure(error)
    assert failure.code == expected
    assert "private" not in str(failure)
