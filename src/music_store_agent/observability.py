"""Allowlisted JSON events without prompts, capabilities or exception diagnostics."""

import json
import logging
import math
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from music_store_agent.agent.errors import ErrorCode

type Event = Literal["http_request", "agent_turn", "sse_delivery"]
type Delivery = Literal["completed", "interrupted"]

REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
FIELDS: frozenset[str] = frozenset(
    {
        "event",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "outcome",
        "tool_calls",
        "delivery",
    }
)
OUTCOMES: frozenset[str] = frozenset(
    {"ok", "internal_error", *(code.value for code in ErrorCode)}
)
METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)


class SafeJSONFormatter(logging.Formatter):
    """Serialize only explicit metrics, excluding message arguments and tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a structured event without inspecting arbitrary exception data.

        Args:
            record: Dedicated logger record containing only safe_fields metrics.

        Returns:
            One JSON line with allowlisted fields and UTC timestamp.
        """
        payload = {
            key: value
            for key, value in getattr(record, "safe_fields", {}).items()
            if key in FIELDS
        }
        return json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                **payload,
            },
            ensure_ascii=True,
        )


LOGGER = logging.getLogger("music_store_agent.observability")
LOGGER.setLevel(logging.INFO)
# Own formatting avoids accidental propagation through verbose server formatters.
LOGGER.propagate = False
if not any(
    isinstance(handler.formatter, SafeJSONFormatter) for handler in LOGGER.handlers
):
    handler = logging.StreamHandler()
    handler.setFormatter(SafeJSONFormatter())
    LOGGER.addHandler(handler)


def emit(
    event: Event,
    *,
    duration_ms: float,
    method: str | None = None,
    route: str | None = None,
    status_code: int | None = None,
    outcome: str | None = None,
    tool_calls: int | None = None,
    delivery: Delivery | None = None,
) -> None:
    """Record only measured scalars and stable classifications at trusted boundaries.

    Args:
        event: Fixed request, committed/failed agent turn or SSE delivery event.
        duration_ms: Monotonic elapsed duration, including any lock/commit wait.
        method: HTTP method; unknown methods are replaced with OTHER.
        route: Registered route template or a fixed unmatched placeholder.
        status_code: Actual HTTP response status, omitted before headers exist.
        outcome: Existing sanitized outcome; unknown values become internal_error.
        tool_calls: Completed committed tool calls when available.
        delivery: Transport completion, independent of agent transaction success.
    """
    fields: dict[str, str | int | float] = {
        "event": event,
        "duration_ms": round(max(0, duration_ms), 3)
        if math.isfinite(duration_ms)
        else 0,
    }
    if identifier := REQUEST_ID.get():
        # Only generated UUID4 context is accepted; no arbitrary caller headers.
        try:
            parsed = UUID(identifier)
            if str(parsed) == identifier and parsed.version == 4:
                fields["request_id"] = identifier
        except ValueError:
            pass
    if method is not None:
        fields["method"] = method if method in METHODS else "OTHER"
    if route is not None:
        fields["route"] = route
    if status_code is not None:
        fields["status_code"] = status_code
    if outcome is not None:
        fields["outcome"] = outcome if outcome in OUTCOMES else "internal_error"
    if tool_calls is not None:
        fields["tool_calls"] = tool_calls
    if delivery is not None:
        fields["delivery"] = delivery
    try:
        LOGGER.info(event, extra={"safe_fields": fields})
    except Exception:
        # Telemetry failures must never change an already committed agent outcome.
        pass
