"""Versioned public HTTP models without checkpoint or provider internals."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from music_store_agent.agent.service import HistoryMessage, _validate_message


class EmptyRequest(BaseModel):
    """Allow only an empty creation payload without client configuration overrides."""

    model_config = ConfigDict(extra="forbid", strict=True)


class MessageRequest(EmptyRequest):
    """Accept up to 2000 characters within the 32 KiB encoded HTTP body limit."""

    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Check shape and phone privacy before dataset-aware dependency validation.

        Args:
            value: Untrusted customer text.

        Returns:
            Unchanged text for normal service processing.
        """
        _validate_message(value, check_contacts=False)
        return value


class ConversationCreated(BaseModel):
    """Return the capability only at the explicit conversation creation boundary."""

    conversation_id: UUID
    access_token: str = Field(repr=False)


class ConversationHistory(BaseModel):
    """Expose only authorized user and assistant display text."""

    conversation_id: UUID
    messages: tuple[HistoryMessage, ...]


class ErrorDetail(BaseModel):
    """Stable error identifier and sanitized Brazilian Portuguese message."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Common HTTP and SSE error envelope without sensitive diagnostic values."""

    error: ErrorDetail


class ReadinessResponse(BaseModel):
    """Report operational readiness without contacting the external model."""

    status: Literal["ready", "not_ready"]
    error: ErrorDetail | None = None
