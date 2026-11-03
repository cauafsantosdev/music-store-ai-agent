"""Small typed HTTP client for public conversation contracts and completed SSE."""

import json
from collections.abc import Callable, Iterable, Iterator
from typing import Literal, TypeVar
from uuid import UUID

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Read bounded server-to-server HTTP settings independently of Groq credentials."""

    model_config = SettingsConfigDict(extra="ignore")
    backend_url: AnyHttpUrl = AnyHttpUrl("http://api:8000")
    backend_timeout_seconds: float = Field(default=5, ge=1, le=30)
    backend_read_timeout_seconds: float = Field(default=180, ge=30, le=600)

    @field_validator("backend_url")
    @classmethod
    def validate_base_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Reject credentials and query data in configured URLs.

        Args:
            value: Deployment-controlled backend URL.

        Returns:
            A safe base address for capability-free request paths.

        Raises:
            ValueError: If URL credentials, query parameters or fragments are present.
        """
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("Backend URLs must not contain credentials or query data.")
        return value

    def timeout(self) -> httpx.Timeout:
        """Build finite connect/read/write/pool deadlines.

        Returns:
            HTTP deadlines supporting multiple bounded agent iterations.
        """
        return httpx.Timeout(
            connect=self.backend_timeout_seconds,
            read=self.backend_read_timeout_seconds,
            write=10,
            pool=5,
        )


class PublicModel(BaseModel):
    """Accept only typed public fields while ignoring unrelated response metadata."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)


class Capability(PublicModel):
    """Keep the capability redacted and local to one server-side session."""

    conversation_id: UUID
    access_token: SecretStr = Field(repr=False)


class DisplayMessage(PublicModel):
    """Store only the role and visible text from the public history endpoint."""

    role: Literal["user", "assistant"]
    text: str


class History(PublicModel):
    """Represent filtered conversation history without native checkpoint objects."""

    conversation_id: UUID
    messages: list[DisplayMessage]


class Reply(PublicModel):
    """Represent a complete agent reply without implying token-level streaming."""

    text: str = Field(min_length=1)
    code: Literal["ok", "not_found", "order_unverified"]
    tool_calls: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def require_visible_text(cls, value: str) -> str:
        """Reject blank completed responses without displaying diagnostic values.

        Args:
            value: Public assistant response text.

        Returns:
            Valid nonblank response text.

        Raises:
            ValueError: If the backend returned only whitespace.
        """
        if not value.strip():
            raise ValueError("A completed response must contain visible text.")
        return value


ERROR_MESSAGES: dict[str, str] = {
    "session": "Não foi possível acessar esta conversa. Inicie uma nova conversa.",
    "rate_limit": (
        "O limite de consultas foi atingido. Aguarde antes de tentar novamente."
    ),
    "unavailable": "O atendimento está temporariamente indisponível. Tente mais tarde.",
    "provider_timeout": (
        "A consulta demorou mais que o esperado. Atualize o histórico "
        "antes de continuar."
    ),
    "invalid_input": "Confira a mensagem e use apenas dados fictícios de demonstração.",
    "connection": (
        "Não foi possível conectar ao atendimento. Verifique a conexão mais tarde."
    ),
    "read_timeout": (
        "O tempo de espera terminou. Atualize o histórico para confirmar o resultado."
    ),
    "protocol": (
        "Não foi possível confirmar a resposta do atendimento. Atualize o histórico."
    ),
    "incomplete": (
        "A conexão terminou antes da confirmação. Atualize o "
        "histórico; a mensagem não foi reenviada."
    ),
    "interrupted": (
        "A consulta foi interrompida. Atualize o histórico antes de "
        "enviar outra mensagem."
    ),
    "configuration": (
        "A conexão com o atendimento ainda não está configurada corretamente."
    ),
}


class ClientError(Exception):
    """Carry fixed Portuguese feedback without raw HTTP exceptions or payloads."""

    def __init__(self, code: str, *, ambiguous: bool = False) -> None:
        """Retain only allowlisted diagnostic categories.

        Args:
            code: Fixed frontend error identifier.
            ambiguous: Whether a submitted operation may already have committed.
        """
        self.code = code if code in ERROR_MESSAGES else "unavailable"
        self.ambiguous = ambiguous
        super().__init__(ERROR_MESSAGES[self.code])


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Parse SSE lines independently of network chunk and UTF-8 boundaries.

    HTTPX iter_lines() performs incremental byte decoding. SSE comments, data lines,
    CRLF and optional metadata are handled here; retry directives never cause retries.

    Args:
        lines: Incrementally decoded lines from the HTTP response.

    Yields:
        Complete event names and joined JSON data strings.

    Raises:
        ClientError: If a frame is oversized or the stream ends mid-frame.
    """
    name, data, length = "message", [], 0
    for raw in lines:
        line = raw.removesuffix("\r").removeprefix("\ufeff")
        length += len(line)
        if length > 262_144:
            raise ClientError("protocol", ambiguous=True)
        if not line:
            if data:
                yield name, "\n".join(data)
            name, data, length = "message", [], 0
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
    if data:
        raise ClientError("incomplete", ambiguous=True)


T = TypeVar("T", bound=BaseModel)


class BackendClient:
    """Own one short-lived HTTP client for an explicit UI action, without retries."""

    def __init__(self, settings: FrontendSettings) -> None:
        """Create a bounded HTTP client without issuing any request.

        Args:
            settings: Validated backend connection configuration.
        """
        self.settings = settings
        self.http = httpx.Client(
            base_url=str(settings.backend_url).rstrip("/"),
            timeout=settings.timeout(),
            follow_redirects=False,
        )

    def __enter__(self) -> "BackendClient":
        """Return this action-scoped HTTP client.

        Returns:
            The open client.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close connections even when parsing or execution is interrupted.

        Args:
            exc_type: Optional exception type.
            exc_value: Optional exception, never logged.
            traceback: Optional traceback, never displayed.
        """
        self.http.close()

    @staticmethod
    def headers(capability: Capability) -> dict[str, str]:
        """Send a capability exclusively through the Authorization header.

        Args:
            capability: Session-local access material.

        Returns:
            Bearer headers without credentials in paths or query parameters.
        """
        return {"Authorization": f"Bearer {capability.access_token.get_secret_value()}"}

    @staticmethod
    def http_error(status: int) -> ClientError:
        """Map HTTP outcomes without displaying even malformed backend error bodies.

        Args:
            status: HTTP status code.

        Returns:
            Fixed Portuguese feedback rather than raw backend diagnostics.
        """
        code = {
            401: "session",
            404: "session",
            413: "invalid_input",
            422: "invalid_input",
            429: "rate_limit",
            504: "provider_timeout",
        }.get(status, "unavailable")
        return ClientError(code)

    def request(
        self,
        method: str,
        path: str,
        *,
        capability: Capability | None = None,
        payload: dict[str, object] | None = None,
    ) -> httpx.Response:
        """Perform one bounded request without retries or loggable credentials.

        Args:
            method: Fixed HTTP method.
            path: Capability-free public endpoint path.
            capability: Optional session-local authorization material.
            payload: Optional public JSON request body.

        Returns:
            A successful HTTP response.

        Raises:
            ClientError: For sanitized transport and HTTP failures.
        """
        try:
            response = self.http.request(
                method,
                path,
                headers=self.headers(capability) if capability else {},
                json=payload,
            )
        except httpx.ReadTimeout:
            raise ClientError("read_timeout", ambiguous=method != "GET") from None
        except httpx.HTTPError:
            raise ClientError("connection", ambiguous=method != "GET") from None
        if response.status_code >= 300:
            raise self.http_error(response.status_code)
        return response

    @staticmethod
    def decode(response: httpx.Response, model: type[T]) -> T:
        """Validate only public response fields without exposing validation inputs.

        Args:
            response: Response containing public JSON.
            model: Expected transport schema.

        Returns:
            A typed public response.

        Raises:
            ClientError: If response JSON or types are malformed.
        """
        try:
            return model.model_validate_json(response.content)
        except (ValidationError, ValueError):
            raise ClientError("protocol", ambiguous=True) from None

    def create_conversation(self) -> Capability:
        """Create a session only after explicit user interaction.

        Returns:
            A redacted session capability.
        """
        response = self.request("POST", "/api/v1/conversations")
        if response.status_code != 201:
            raise ClientError("protocol", ambiguous=True)
        capability = self.decode(response, Capability)
        if not capability.access_token.get_secret_value():
            raise ClientError("protocol", ambiguous=True)
        return capability

    def history(self, capability: Capability) -> list[DisplayMessage]:
        """Recover filtered history without constructing native LangGraph messages.

        Args:
            capability: Current session capability.

        Returns:
            Public user and assistant display records.
        """
        response = self.request(
            "GET",
            f"/api/v1/conversations/{capability.conversation_id}",
            capability=capability,
        )
        history = self.decode(response, History)
        if history.conversation_id != capability.conversation_id:
            raise ClientError("protocol")
        return history.messages

    def delete(self, capability: Capability) -> None:
        """Delete only the explicitly confirmed authorized conversation.

        Args:
            capability: Current session capability.

        Raises:
            ClientError: Unless deletion was explicitly confirmed by HTTP 204.
        """
        response = self.request(
            "DELETE",
            f"/api/v1/conversations/{capability.conversation_id}",
            capability=capability,
        )
        if response.status_code != 204:
            raise ClientError("protocol", ambiguous=True)

    def status(self) -> bool:
        """Check liveness then readiness only on an explicit status action.

        Returns:
            Whether operational readiness was reported, without a generation call.
        """
        try:
            health = self.http.get(
                "/health", timeout=self.settings.backend_timeout_seconds
            )
            if health.status_code != 200 or health.json() != {"status": "ok"}:
                return False
            ready = self.http.get(
                "/ready", timeout=self.settings.backend_timeout_seconds
            )
            return ready.status_code == 200 and ready.json().get("status") == "ready"
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    def stream_reply(
        self,
        capability: Capability,
        message: str,
        on_status: Callable[[], None],
    ) -> Reply:
        """Submit once and accept a reply only after a valid completed SSE sequence.

        No ordinary-message fallback is attempted after starting this request. A
        dropped or malformed stream can still correspond to a committed backend turn.

        Args:
            capability: Current access capability.
            message: The single new customer message, never previous history.
            on_status: Fixed UI progress callback, without backend-provided text.

        Returns:
            A single complete reply after the done event.

        Raises:
            ClientError: For HTTP errors, explicit error events or incomplete delivery.
        """
        try:
            with self.http.stream(
                "POST",
                f"/api/v1/conversations/{capability.conversation_id}/stream",
                headers=self.headers(capability),
                json={"message": message},
            ) as response:
                if response.status_code != 200:
                    raise self.http_error(response.status_code)
                if (
                    response.headers.get("content-type", "").split(";")[0]
                    != "text/event-stream"
                ):
                    raise ClientError("protocol", ambiguous=True)
                started, reply, failure = False, None, None
                for name, raw in parse_sse(response.iter_lines()):
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise ClientError("protocol", ambiguous=True)
                    if (
                        name == "status"
                        and not started
                        and data == {"status": "processing"}
                    ):
                        started = True
                        on_status()
                    elif (
                        name == "message"
                        and started
                        and reply is None
                        and failure is None
                    ):
                        reply = Reply.model_validate_json(raw)
                    elif (
                        name == "error"
                        and started
                        and reply is None
                        and failure is None
                    ):
                        status = data.get("status_code")
                        if not isinstance(status, int) or not isinstance(
                            data.get("error"), dict
                        ):
                            raise ClientError("protocol", ambiguous=True)
                        failure = self.http_error(status)
                    elif name == "done" and started:
                        if (
                            data == {"status": "completed"}
                            and reply is not None
                            and failure is None
                        ):
                            return reply
                        if (
                            data == {"status": "failed"}
                            and failure is not None
                            and reply is None
                        ):
                            raise failure
                        raise ClientError("protocol", ambiguous=True)
                    else:
                        raise ClientError("protocol", ambiguous=True)
        except ClientError:
            raise
        except httpx.ReadTimeout:
            raise ClientError("read_timeout", ambiguous=True) from None
        except httpx.HTTPError:
            raise ClientError("connection", ambiguous=True) from None
        except (ValueError, ValidationError):
            raise ClientError("protocol", ambiguous=True) from None
        raise ClientError("incomplete", ambiguous=True)
