"""Session-local display state and explicit actions without backend business logic."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from web.client import BackendClient, Capability, ClientError, DisplayMessage

type Action = Literal["send", "history", "delete", "status"]


@dataclass
class ChatState:
    """Keep capabilities only in one Streamlit server-side browser session.

    Normal reruns retain this object. Configured HttpOnly-cookie recovery restores
    capabilities after a new WebSocket; PostgreSQL remains the history authority.
    """

    capability: Capability | None = field(default=None, repr=False)
    messages: list[DisplayMessage] = field(default_factory=list, repr=False)
    submitted: str | None = field(default=None, repr=False)
    action: Action | None = None
    busy: bool = False
    needs_refresh: bool = False
    confirm_delete: bool = False
    error: ClientError | None = None
    backend_ready: bool | None = None
    notice: str | None = None
    browser_persistent: bool = False
    browser_ready: bool = False
    browser_operation: Literal["start", "new", "clear"] | None = None
    browser_revision: str | None = None

    def reset(self) -> None:
        """Forget local state without deleting or implicitly reusing any capability."""
        self.capability = None
        self.messages = []
        self.submitted = None
        self.action = None
        self.busy = self.needs_refresh = self.confirm_delete = False
        self.error = None
        self.notice = None

    def queue(self, message: str) -> None:
        """Queue one nonempty user action without making any HTTP request.

        Args:
            message: Chat input or a verified synthetic suggestion.
        """
        if not message.strip() or self.busy or self.action or self.needs_refresh:
            return
        self.submitted = message.strip()
        self.action = "send"
        self.error = None
        self.notice = None

    def visible_text(self, text: str) -> str:
        """Defensively redact the current capability from all displayable text.

        Args:
            text: Public message or optimistic input.

        Returns:
            Text safe to send to Streamlit rendering primitives.
        """
        if self.capability:
            return text.replace(
                self.capability.access_token.get_secret_value(), "[credencial ocultada]"
            )
        return text

    def interrupt(self) -> None:
        """Mark a consumed, interrupted action as ambiguous rather than resubmitting."""
        self.busy = False
        self.action = None
        self.needs_refresh = self.capability is not None
        self.error = ClientError("interrupted", ambiguous=True)

    def execute(self, client: BackendClient, on_status: Callable[[], None]) -> None:
        """Consume one action before I/O so reruns cannot replay a submission.

        Args:
            client: Short-lived HTTP client owned by the active UI action.
            on_status: Fixed progress callback for valid SSE status events.
        """
        action, self.action = self.action, None
        if action is None or self.busy:
            return
        self.busy = True
        finished = False
        try:
            self.error, self.notice = None, None
            if action == "send" and self.submitted:
                if self.capability is None:
                    self.capability = client.create_conversation()
                reply = client.stream_reply(self.capability, self.submitted, on_status)
                self.messages.extend(
                    [
                        DisplayMessage(
                            role="user", text=self.visible_text(self.submitted)
                        ),
                        DisplayMessage(
                            role="assistant", text=self.visible_text(reply.text)
                        ),
                    ]
                )
                self.submitted = None
                self.needs_refresh = False
            elif action == "history" and self.capability:
                history = client.history(self.capability)
                self.messages = [
                    DisplayMessage(role=item.role, text=self.visible_text(item.text))
                    for item in history
                ]
                self.submitted = None
                self.needs_refresh = False
                self.notice = "Histórico atualizado. Nenhuma mensagem foi reenviada."
            elif action == "delete" and self.capability and self.confirm_delete:
                client.delete(self.capability)
                self.reset()
                if self.browser_persistent:
                    self.browser_operation = "clear"
                    self.browser_revision = str(uuid4())
                self.notice = "Conversa excluída."
            elif action == "status":
                self.backend_ready = client.status()
            finished = True
        except ClientError as error:
            error.request_id = client.request_id
            self.error = error
            # Failed sends stay visible as unconfirmed input, never assistant replies.
            # Even a timeout may have committed; only an explicit GET reconciles it.
            if self.capability and (
                action == "send" or error.code == "session" or error.ambiguous
            ):
                self.needs_refresh = True
            finished = True
        finally:
            self.busy = False
            if not finished:
                self.interrupt()
