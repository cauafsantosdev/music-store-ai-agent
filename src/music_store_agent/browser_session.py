"""Server-only encrypted browser recovery payloads and explicit deployment settings."""

import base64
import re
from typing import Literal
from uuid import UUID

from cryptography.fernet import Fernet, InvalidToken
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

COOKIE_NAME: str = "music_store_session"
COOKIE_TTL: int = 7 * 24 * 60 * 60
COOKIE_PATH: str = "/browser-session"


class BrowserSessionSettings(BaseSettings):
    """Require a stable secret and explicit localhost exception for HTTP development."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)
    browser_session_key: SecretStr | None = None
    browser_app_origin: AnyHttpUrl = AnyHttpUrl("http://localhost:8501")
    browser_session_api_origin: AnyHttpUrl | None = None
    browser_session_allow_http_localhost: bool = False

    @field_validator("browser_session_key", mode="before")
    @classmethod
    def blank_key(cls, value: object) -> object:
        """Treat a blank tracked configuration entry as unconfigured.

        Args:
            value: Secret configuration, never logged or displayed.

        Returns:
            None for an empty entry, otherwise the unchanged value.
        """
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_deployment(self) -> "BrowserSessionSettings":
        """Reject weak secrets, unsafe origins and implicit production HTTP exceptions.

        Returns:
            Validated settings without modifying application model configuration.

        Raises:
            ValueError: For unsafe deployment configuration, without supplied values.
        """
        origins = (
            self.browser_app_origin,
            self.browser_session_api_origin or self.browser_app_origin,
        )
        for origin in origins:
            if (
                origin.username
                or origin.password
                or origin.query
                or origin.fragment
                or origin.path not in ("", "/")
            ):
                raise ValueError("Invalid browser origin configuration.")
        if origins[0].host != origins[1].host:
            raise ValueError("Browser cookie origins must use the same host.")
        if self.browser_session_key is None:
            if any(origin.scheme == "https" for origin in origins):
                raise ValueError("A stable browser encryption key is required.")
            return self
        try:
            key = self.browser_session_key.get_secret_value()
            decoded = base64.b64decode(key, altchars=b"-_", validate=True)
            if len(key) != 44 or len(decoded) != 32 or len(set(decoded)) < 16:
                raise ValueError
            Fernet(key.encode("ascii"))
        except (ValueError, UnicodeError):
            raise ValueError("Invalid browser encryption key.") from None
        for origin in origins:
            if origin.scheme != "https" and not (
                self.browser_session_allow_http_localhost
                and origin.scheme == "http"
                and origin.host in ("localhost", "127.0.0.1", "::1")
            ):
                raise ValueError("Secure browser origins are required.")
        return self

    @property
    def enabled(self) -> bool:
        """Indicate whether browser persistence is explicitly configured.

        Returns:
            Whether a validated secret is present.
        """
        return self.browser_session_key is not None

    @property
    def secure(self) -> bool:
        """Use trusted deployment configuration, never forwarded request headers.

        Returns:
            Whether the cookie must have Secure set.
        """
        return self.browser_app_origin.scheme == "https"

    @property
    def app_origin(self) -> str:
        """Return the exact allowed browser origin.

        Returns:
            Canonical origin without a trailing slash.
        """
        return str(self.browser_app_origin).rstrip("/")

    @property
    def endpoint(self) -> str:
        """Resolve only the small browser-facing API prefix.

        Returns:
            Browser-reachable API base, including the release reverse-proxy prefix.
        """
        return (
            str(self.browser_session_api_origin or self.browser_app_origin).rstrip("/")
            + COOKIE_PATH
        )

    def cipher(self) -> Fernet:
        """Construct maintained authenticated encryption from the stable secret.

        Returns:
            A Fernet cipher without key generation or persistent local state.

        Raises:
            ValueError: If persistence is not configured.
        """
        if self.browser_session_key is None:
            raise ValueError("Browser persistence is not configured.")
        return Fernet(self.browser_session_key.get_secret_value().encode("ascii"))


class BrowserCredential(BaseModel):
    """Versioned, bounded server-only recovery material; never sent as browser JSON."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    v: Literal[1] = 1
    conversation_id: UUID
    access_token: SecretStr
    revision: UUID

    @field_validator("access_token")
    @classmethod
    def valid_token(cls, value: SecretStr) -> SecretStr:
        """Reject malformed authorization material before PostgreSQL verification.

        Args:
            value: Decrypted capability, never included in diagnostics.

        Returns:
            The validated secret wrapper.

        Raises:
            ValueError: For malformed bounded capabilities.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value.get_secret_value()):
            raise ValueError("Invalid recovery credential.")
        return value


def seal(credential: BrowserCredential, settings: BrowserSessionSettings) -> str:
    """Encrypt a small JSON payload without arbitrary Python serialization.

    Args:
        credential: Server-side capability and public operation revision.
        settings: Explicit validated encryption configuration.

    Returns:
        Opaque authenticated ciphertext suitable only for an HttpOnly cookie.
    """
    payload = credential.model_dump(mode="json")
    payload["access_token"] = credential.access_token.get_secret_value()
    import json

    return settings.cipher().encrypt(json.dumps(payload).encode()).decode("ascii")


def unseal(
    value: str | None, settings: BrowserSessionSettings
) -> BrowserCredential | None:
    """Validate integrity, lifetime and strict schema without exposing rejected values.

    Args:
        value: Encrypted handshake cookie, never a browser widget value.
        settings: Explicit stable encryption configuration.

    Returns:
        Server-side recovery material or None for absent, invalid or expired cookies.
    """
    if not value or len(value) > 2048 or not settings.enabled:
        return None
    try:
        data = settings.cipher().decrypt(value.encode("ascii"), ttl=COOKIE_TTL)
        return BrowserCredential.model_validate_json(data)
    except (InvalidToken, ValueError, UnicodeError):
        return None
