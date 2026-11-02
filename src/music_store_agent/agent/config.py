"""Lazy environment configuration for GroqCloud and bounded execution."""

from typing import Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from music_store_agent.agent.errors import AgentFailure, ErrorCode


class AgentSettings(BaseSettings):
    """Read explicit agent settings without loading credentials or models at import."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    ai_model: Literal["openai/gpt-oss-120b"] = "openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    ai_demo_schema: str = Field(
        default="agent_demo", pattern=r"^agent_demo(?:_[a-z0-9_]+)?$"
    )
    ai_max_model_calls: int = Field(default=4, ge=1, le=8)
    ai_max_tool_calls: int = Field(default=6, ge=1, le=12)
    ai_recursion_limit: int = Field(default=24, ge=4, le=48)
    ai_max_output_tokens: int = Field(default=2048, ge=256, le=4096)
    ai_provider_timeout_seconds: float = Field(default=30, ge=5, le=60)
    ai_max_history_messages: int = Field(default=60, ge=8, le=100)


def load_settings() -> AgentSettings:
    """Validate current environment values while sanitizing configuration failures.

    Returns:
        Validated settings; a missing key is allowed until live model construction.

    Raises:
        AgentFailure: If configuration values are invalid.
    """
    try:
        return AgentSettings()
    except (ValidationError, ValueError):
        raise AgentFailure(ErrorCode.CONFIGURATION) from None
