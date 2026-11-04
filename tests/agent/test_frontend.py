"""Streamlit AppTest workflows over the real HTTP API and isolated PostgreSQL."""

import json
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from sqlalchemy import Engine
from streamlit.testing.v1 import AppTest

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api.main import create_app
from web.app import SUGGESTIONS
from web.client import BackendClient, Capability, Reply
from web.state import ChatState


@dataclass
class UIBackend:
    """Route frontend HTTP requests through the real API without external inference."""

    client: TestClient = field(repr=False)
    model: ScriptedModel = field(repr=False)
    requests: list[httpx.Request] = field(default_factory=list, repr=False)
    failure: int | None = None
    incomplete: bool = False
    timeout: bool = False

    def respond(self, request: httpx.Request) -> httpx.Response:
        """Forward real contracts or inject a controlled transport/delivery failure.

        Args:
            request: Frontend request, never logged or exposed in UI.

        Returns:
            Public HTTP API response with optional truncated SSE delivery.

        Raises:
            httpx.ReadTimeout: For a controlled ambiguous delivery failure.
        """
        self.requests.append(request)
        if self.timeout and request.url.path.endswith("/stream"):
            raise httpx.ReadTimeout("private-transport-marker", request=request)
        if self.failure and (
            request.url.path.endswith("/stream") or request.method == "DELETE"
        ):
            return httpx.Response(
                self.failure,
                request=request,
                json={"error": {"code": "private-marker", "message": "private-marker"}},
            )
        response = self.client.request(
            request.method,
            request.url.path,
            headers=dict(request.headers),
            content=request.content,
        )
        content = response.content
        if self.incomplete and request.url.path.endswith("/stream"):
            # The actual service commits, but the frontend never receives done.
            content = content.split(b"event: done")[0]
        return httpx.Response(
            response.status_code,
            request=request,
            headers={
                "content-type": response.headers.get("content-type", ""),
                "x-request-id": response.headers.get("x-request-id", ""),
            },
            content=content,
        )


@pytest.fixture
def ui_backend(
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[UIBackend]:
    """Exercise public HTTP contracts while replacing only external inference.

    Args:
        demo_engine: Fingerprint-verified synthetic operational schema.
        state_store: Isolated checkpoint storage.
        offline_settings: Explicit provider-free settings.
        monkeypatch: Scoped frontend HTTP bridge to the real API.

    Yields:
        A bridge recording outbound frontend operations without logging secrets.
    """
    model = ScriptedModel(
        responses=[
            AIMessage(content=f"**Resposta fictícia {index}.**", id=f"ui-{index}")
            for index in range(20)
        ]
    )
    application = create_app(
        settings=offline_settings, model=model, engine=demo_engine, store=state_store
    )
    with TestClient(application) as client:
        bridge = UIBackend(client, model)

        def respond(
            self: httpx.HTTPTransport, request: httpx.Request
        ) -> httpx.Response:
            """Forward only frontend HTTP requests to the isolated FastAPI application.

            Args:
                self: Frontend transport, never used for a real network request.
                request: Public API request.

            Returns:
                The controlled public HTTP response.
            """
            assert request.url.host == "api"
            assert not request.url.query
            return bridge.respond(request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("BROWSER_SESSION_KEY", raising=False)
        yield bridge


def open_ui() -> AppTest:
    """Run the actual Streamlit entry point without browser automation.

    Returns:
        A tested widget tree; this does not claim pixel-level visual inspection.
    """
    script = Path(__file__).resolve().parents[2] / "web" / "app.py"
    app = AppTest.from_file(str(script)).run(timeout=15)
    assert not app.exception
    return app


def visible_copy(app: AppTest) -> str:
    """Read only visible textual UI elements for privacy and localization checks.

    Args:
        app: Tested Streamlit widget tree.

    Returns:
        The public rendered copy, without server-side session state.
    """
    return "\n".join(
        str(item.value)
        for kind in (
            app.title,
            app.caption,
            app.markdown,
            app.error,
            app.info,
            app.success,
        )
        for item in kind
    )


def test_dark_theme_and_lazy_portuguese_welcome(ui_backend: UIBackend) -> None:
    """Load a dark Portuguese welcome without creating sessions or model requests.

    Args:
        ui_backend: Real HTTP API bridge with controlled inference.
    """
    theme = tomllib.loads(Path(".streamlit/config.toml").read_text())
    assert theme["theme"]["base"] == "dark"
    assert theme["theme"]["backgroundColor"] == "#101827"
    assert theme["theme"]["primaryColor"] == "#EAB36B"
    assert theme["runner"]["fastReruns"] is False
    app = open_ui()
    assert "Empório da Música" in app.title[0].value
    assert app.chat_input[0].placeholder == "Digite sua mensagem..."
    assert "Assistente virtual" in visible_copy(app)
    assert "dados fictícios" in visible_copy(app)
    assert {"Nova conversa", "Excluir conversa", "Atualizar histórico"}.issubset(
        {button.label for button in app.button}
    )
    assert ui_backend.requests == [] and ui_backend.model.seen == []


def test_single_submission_follow_up_and_reruns_preserve_capability(
    ui_backend: UIBackend,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use one creation and exactly one SSE POST for each explicit message action.

    Args:
        ui_backend: Actual API bridge and call spy.
        caplog: Captured logs for capability exclusion checks.
    """
    app = open_ui()
    app.chat_input[0].set_value("Olá, demonstração.").run(timeout=15)
    assert not app.exception
    state: ChatState = app.session_state["music_store_chat"]
    identifier = state.capability.conversation_id
    token = state.capability.access_token.get_secret_value()
    assert len(app.chat_message) == 2
    assert "Olá, demonstração." in visible_copy(app)
    assert "Resposta fictícia 0" in visible_copy(app)
    assert [r.method for r in ui_backend.requests] == ["POST", "POST"]
    assert ui_backend.requests[-1].url.path.endswith("/stream")
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2
    app.chat_input[0].set_value("E as condições de pagamento?").run(timeout=15)
    assert state.capability.conversation_id == identifier
    assert len(ui_backend.requests) == 3 and len(ui_backend.model.seen) == 2
    assert len(app.chat_message) == 4
    assert token not in visible_copy(app) + caplog.text + repr(state)
    assert all(token not in str(request.url) for request in ui_backend.requests)


@pytest.mark.parametrize("index", range(4))
def test_suggestions_share_the_normal_exactly_once_workflow(
    ui_backend: UIBackend, index: int
) -> None:
    """Send verified example questions through the same session and SSE pathway.

    Args:
        ui_backend: Real HTTP API bridge.
        index: One of the four verified synthetic example questions.
    """
    app = open_ui()
    app.button(key=f"suggestion_{index}").click().run(timeout=15)
    assert not app.exception
    streams = [
        request
        for request in ui_backend.requests
        if request.url.path.endswith("/stream")
    ]
    assert len(streams) == 1
    assert json.loads(streams[0].content) == {"message": SUGGESTIONS[index][1]}
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2
    assert len(ui_backend.model.seen) == 1


def test_empty_input_and_new_conversation_have_no_implicit_side_effects(
    ui_backend: UIBackend,
) -> None:
    """Reject blanks locally and reset display state without deleting earlier sessions.

    Args:
        ui_backend: Real HTTP API bridge.
    """
    app = open_ui()
    app.chat_input[0].set_value("   ").run(timeout=15)
    assert ui_backend.requests == []
    app.chat_input[0].set_value("Uma pergunta fictícia.").run(timeout=15)
    old = app.session_state["music_store_chat"].capability
    app.button(key="new_conversation").click().run(timeout=15)
    state = app.session_state["music_store_chat"]
    assert state.capability is None and state.messages == []
    assert not app.chat_message
    assert not any(request.method == "DELETE" for request in ui_backend.requests)
    assert len(ui_backend.requests) == 2
    app.chat_input[0].set_value("Outra pergunta.").run(timeout=15)
    assert state.capability.conversation_id != old.conversation_id
    assert len(ui_backend.requests) == 4


def test_truncated_committed_turn_requires_explicit_history_refresh(
    ui_backend: UIBackend,
) -> None:
    """Reconcile ambiguous delivery with GET, never by repeating a message POST.

    Args:
        ui_backend: Real service committing before a deliberately truncated SSE frame.
    """
    app = open_ui()
    ui_backend.incomplete = True
    app.chat_input[0].set_value("Confirme esta consulta fictícia.").run(timeout=15)
    state = app.session_state["music_store_chat"]
    assert state.needs_refresh and state.messages == []
    assert len(app.chat_message) == 1
    assert "Resposta fictícia" not in visible_copy(app)
    assert app.chat_input[0].disabled
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2
    app.button(key="refresh_history").click().run(timeout=15)
    assert ui_backend.requests[-1].method == "GET"
    assert len(ui_backend.requests) == 3
    assert len(app.chat_message) == 2
    assert len(ui_backend.model.seen) == 1
    assert not state.needs_refresh and not app.chat_input[0].disabled
    assert "Resposta fictícia 0" in visible_copy(app)


@pytest.mark.parametrize("status", [401, 404, 429, 503, 504])
def test_http_failures_are_feedback_not_assistant_messages(
    ui_backend: UIBackend, status: int
) -> None:
    """Keep backend failures sanitized, visible and session-local without retries.

    Args:
        ui_backend: Controlled HTTP failure bridge.
        status: Actual public API error status.
    """
    app = open_ui()
    ui_backend.failure = status
    app.chat_input[0].set_value("Pergunta de demonstração.").run(timeout=15)
    state = app.session_state["music_store_chat"]
    assert state.capability is not None and state.needs_refresh
    assert len(app.chat_message) == 1 and len(app.error) == 1
    assert "private-marker" not in visible_copy(app)
    assert state.capability.access_token.get_secret_value() not in visible_copy(app)
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2
    assert ui_backend.model.seen == []


def test_timeout_and_interrupted_reruns_never_repeat_a_turn(
    ui_backend: UIBackend,
) -> None:
    """Consume submissions before I/O and require reconciliation after uncertainty.

    Args:
        ui_backend: Controlled ambiguous transport failure.
    """
    app = open_ui()
    ui_backend.timeout = True
    app.chat_input[0].set_value("Pergunta fictícia.").run(timeout=15)
    state = app.session_state["music_store_chat"]
    assert state.needs_refresh
    state.busy = True
    app.run(timeout=15)
    assert not state.busy
    assert state.error.code == "interrupted"
    assert len(ui_backend.requests) == 2
    assert "private-transport-marker" not in visible_copy(app)


@pytest.mark.parametrize("failure", [None, 503])
def test_delete_requires_confirmation_and_preserves_state_on_failure(
    ui_backend: UIBackend, failure: int | None
) -> None:
    """Only clear credentials after explicit confirmation and a successful DELETE.

    Args:
        ui_backend: Real session management with optional deletion failure.
        failure: Optional controlled backend status.
    """
    app = open_ui()
    app.chat_input[0].set_value("Olá.").run(timeout=15)
    state = app.session_state["music_store_chat"]
    original = state.capability
    app.button(key="delete_conversation").click().run(timeout=15)
    assert len(ui_backend.requests) == 2
    app.button(key="cancel_delete").click().run(timeout=15)
    assert len(ui_backend.requests) == 2
    app.button(key="delete_conversation").click().run(timeout=15)
    ui_backend.failure = failure
    app.button(key="confirm_delete").click().run(timeout=15)
    assert ui_backend.requests[-1].method == "DELETE"
    if failure:
        assert state.capability == original and len(state.messages) == 2
        assert state.confirm_delete and len(app.error) == 1
    else:
        assert state.capability is None and state.messages == []
        assert not state.confirm_delete and not app.chat_message


def test_browser_sessions_do_not_share_capabilities_or_messages(
    ui_backend: UIBackend,
) -> None:
    """Keep two independent Streamlit sessions isolated over the same backend.

    Args:
        ui_backend: Shared API with isolated PostgreSQL state.
    """
    first = open_ui()
    first.chat_input[0].set_value("Primeira sessão.").run(timeout=15)
    first_state = first.session_state["music_store_chat"]
    second = open_ui()
    second_state = second.session_state["music_store_chat"]
    assert second_state.capability is None and not second.chat_message
    second.chat_input[0].set_value("Segunda sessão.").run(timeout=15)
    assert first_state.capability != second_state.capability
    assert "Primeira sessão." not in visible_copy(second)
    assert "Segunda sessão." not in visible_copy(first)


def test_status_is_only_checked_explicitly_without_generation(
    ui_backend: UIBackend,
) -> None:
    """Distinguish readiness from generation and avoid traffic on ordinary reruns.

    Args:
        ui_backend: Real readiness and health endpoints.
    """
    app = open_ui()
    app.button(key="check_status").click().run(timeout=15)
    assert [request.url.path for request in ui_backend.requests] == [
        "/health",
        "/ready",
    ]
    assert app.session_state["music_store_chat"].backend_ready
    assert ui_backend.model.seen == []
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2


def test_history_renders_public_text_without_native_tool_or_system_records(
    ui_backend: UIBackend,
) -> None:
    """Render authorized history after real pricing without leaking native records.

    Args:
        ui_backend: Real API, tool execution and isolated checkpoint persistence.
    """
    ui_backend.model.responses = [
        AIMessage(
            content="",
            additional_kwargs={"opaque_signature": "hidden-signature-marker"},
            tool_calls=[
                {
                    "name": "get_product_pricing",
                    "args": {"product_id": 101, "payment_method": "pix"},
                    "id": "hidden-tool-call-marker",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="**Preço verificado: R$ 1.104,93.**",
            additional_kwargs={"reasoning_content": "hidden-reasoning-marker"},
        ),
    ]
    app = open_ui()
    app.chat_input[0].set_value("Consulte o preço do produto fictício 101.").run(
        timeout=15
    )
    native = ui_backend.model.seen[-1]
    assert any(isinstance(item, ToolMessage) for item in native)
    assert any(isinstance(item, SystemMessage) for item in native)
    app.button(key="refresh_history").click().run(timeout=15)
    assert not app.exception and len(app.chat_message) == 2
    text = visible_copy(app)
    assert "**Preço verificado: R$ 1.104,93.**" in text
    for hidden in (
        "hidden-signature-marker",
        "hidden-reasoning-marker",
        "hidden-tool-call-marker",
        '"discount_source"',
        '"tool_calls"',
    ):
        assert hidden not in text
    for item in native:
        if isinstance(item, SystemMessage):
            assert item.content not in text


def test_capability_echo_is_redacted_and_unavailable_status_stays_safe(
    ui_backend: UIBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep defensive token redaction effective even after an explicit status failure.

    Args:
        ui_backend: Real HTTP bridge for initial session creation.
        monkeypatch: Scoped unavailable transport for explicit status verification.
        caplog: Captured logs checked without printing credentials.
    """
    app = open_ui()
    app.chat_input[0].set_value("Olá.").run(timeout=15)
    state = app.session_state["music_store_chat"]
    token = state.capability.access_token.get_secret_value()
    ui_backend.model.responses[1] = AIMessage(
        content=f"Eco fictício {token}", id="synthetic-echo"
    )
    app.chat_input[0].set_value("Outra consulta fictícia.").run(timeout=15)
    app.button(key="refresh_history").click().run(timeout=15)
    assert "[credencial ocultada]" in visible_copy(app)
    assert token not in visible_copy(app) + caplog.text

    def unavailable(
        self: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Simulate an unreachable service without a real network attempt.

        Args:
            self: Frontend transport.
            request: Explicit status request.

        Raises:
            httpx.ConnectError: Always, with an undisplayed diagnostic.
        """
        raise httpx.ConnectError("private-connection-marker", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", unavailable)
    app.button(key="check_status").click().run(timeout=15)
    assert not app.exception and state.backend_ready is False
    assert state.capability is not None and len(app.chat_message) == 4
    assert "Serviço temporariamente indisponível" in visible_copy(app)
    assert "private-connection-marker" not in visible_copy(app) + caplog.text


def test_processing_and_failed_turn_reference_preserve_one_submission(
    ui_backend: UIBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe a real status callback and expose only a validated failure reference.

    Args:
        ui_backend: Real API and PostgreSQL with controlled provider timeout.
        monkeypatch: Scoped observation of the existing complete-message workflow.
    """
    app = open_ui()
    ui_backend.model.failure_code = 504
    statuses: list[bool] = []
    original = BackendClient.stream_reply

    def observed(
        self: BackendClient,
        capability: Capability,
        message: str,
        on_status: Callable[[], None],
    ) -> Reply:
        """Observe pending state while forwarding the actual one-request SSE parser.

        Args:
            self: Action-scoped HTTP client.
            capability: Session-local access material, never displayed.
            message: Single synthetic user submission.
            on_status: Actual Streamlit progress update.

        Returns:
            The canonical reply if the provider succeeds.
        """

        def pending() -> None:
            """Check duplicate prevention during an actual backend processing event."""
            state = app.session_state["music_store_chat"]
            assert state.busy and state.action is None
            on_status()
            statuses.append(True)

        return original(self, capability, message, pending)

    monkeypatch.setattr(BackendClient, "stream_reply", observed)
    app.chat_input[0].set_value("Consulta fictícia com timeout controlado.").run(
        timeout=15
    )
    assert not app.exception and statuses == [True]
    state = app.session_state["music_store_chat"]
    assert state.error.code == "provider_timeout" and state.error.request_id
    assert f"Referência do atendimento: {state.error.request_id}" in visible_copy(app)
    assert "Solicitação não confirmada" in visible_copy(app)
    assert "Resposta confirmada" not in visible_copy(app)
    assert "private-diagnostic-marker" not in visible_copy(app)
    assert len(app.chat_message) == 1 and app.chat_input[0].disabled
    app.run(timeout=15)
    assert len(ui_backend.requests) == 2 and len(ui_backend.model.seen) == 1
    app.button(key="refresh_history").click().run(timeout=15)
    assert not state.needs_refresh and not app.chat_message
    assert len(ui_backend.requests) == 3 and ui_backend.requests[-1].method == "GET"


def test_new_streamlit_state_restores_authorized_history_without_inference(
    ui_backend: UIBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct browser state from HttpOnly recovery and read canonical history once.

    Args:
        ui_backend: Existing real API/SQL bridge with controlled inference.
        monkeypatch: Public cookie-reader boundary and explicit test configuration.
    """
    from cryptography.fernet import Fernet

    from music_store_agent.browser_session import COOKIE_NAME

    monkeypatch.setenv("BROWSER_SESSION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("BROWSER_APP_ORIGIN", "http://localhost:8501")
    monkeypatch.setenv("BROWSER_SESSION_API_ORIGIN", "http://localhost:8000")
    monkeypatch.setenv("BROWSER_SESSION_ALLOW_HTTP_LOCALHOST", "true")
    response = ui_backend.client.post(
        "/browser-session/start",
        headers={
            "Origin": "http://localhost:8501",
            "X-Browser-Session": "1",
        },
        json={"revision": "30f92243-4ac8-4a5c-9698-354506075795"},
    )
    assert response.status_code == 200
    cookie = response.cookies[COOKIE_NAME]

    def read_cookie() -> str:
        """Represent the server-side handshake cookie without a writable cookie API.

        Returns:
            Opaque test ciphertext, never a widget or browser-side capability.
        """
        return cookie

    monkeypatch.setattr("web.browser_session.handshake_cookie", read_cookie)
    first = open_ui()
    first.chat_input[0].set_value("Consulta fictícia persistente.").run(timeout=15)
    capability = first.session_state["music_store_chat"].capability
    before = len(ui_backend.model.seen)
    reconstructed = open_ui()
    state = reconstructed.session_state["music_store_chat"]
    assert state.capability == capability and state.browser_ready
    assert len(reconstructed.chat_message) == 2
    assert "Consulta fictícia persistente" in visible_copy(reconstructed)
    assert len(ui_backend.model.seen) == before == 1
    assert ui_backend.requests[-1].method == "GET"
    assert cookie not in visible_copy(reconstructed)
    assert capability.access_token.get_secret_value() not in visible_copy(reconstructed)
    requests = len(ui_backend.requests)
    reconstructed.run(timeout=15)
    assert len(ui_backend.requests) == requests
