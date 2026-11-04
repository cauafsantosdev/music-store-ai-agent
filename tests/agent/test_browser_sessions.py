"""Encrypted cookie authorization against real isolated PostgreSQL checkpoints."""

import json
import time
from collections.abc import Iterator
from uuid import uuid4

import pytest
from conftest import ScriptedModel
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy import Engine

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.api.main import create_app
from music_store_agent.browser_session import (
    COOKIE_NAME,
    COOKIE_TTL,
    BrowserCredential,
    BrowserSessionSettings,
    seal,
    unseal,
)

HEADERS: dict[str, str] = {"Origin": "http://localhost:8501", "X-Browser-Session": "1"}


@pytest.fixture
def cookie_settings(monkeypatch: pytest.MonkeyPatch) -> BrowserSessionSettings:
    """Supply a generated test-only secret and explicit localhost configuration.

    Args:
        monkeypatch: Scoped non-production configuration.

    Returns:
        Stable settings reused across service reconstruction within one test.
    """
    monkeypatch.setenv("BROWSER_SESSION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("BROWSER_APP_ORIGIN", HEADERS["Origin"])
    monkeypatch.setenv("BROWSER_SESSION_API_ORIGIN", "http://localhost:8000")
    monkeypatch.setenv("BROWSER_SESSION_ALLOW_HTTP_LOCALHOST", "true")
    return BrowserSessionSettings()


@pytest.fixture
def cookie_client(
    cookie_settings: BrowserSessionSettings,
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> Iterator[tuple[TestClient, ScriptedModel]]:
    """Use real capabilities and checkpoints while controlling only inference.

    Args:
        cookie_settings: Explicit test cookie configuration.
        demo_engine: Verified isolated synthetic data.
        state_store: Disposable PostgresSaver state.
        offline_settings: Provider-free agent configuration.

    Yields:
        A browser-like HTTP cookie jar and the controlled model call spy.
    """
    model = ScriptedModel(
        responses=[AIMessage(content="Resposta fictícia.", id="cookie-answer")]
    )
    with TestClient(
        create_app(
            settings=offline_settings,
            model=model,
            engine=demo_engine,
            store=state_store,
        )
    ) as client:
        yield client, model


def start_cookie(client: TestClient, operation: str = "start") -> str:
    """Issue one browser credential without accessing plaintext JSON authorization.

    Args:
        client: A browser-like HTTP client with its own cookie jar.
        operation: Explicit initial or new-conversation action.

    Returns:
        An opaque encrypted test cookie, never printed.
    """
    response = client.post(
        f"/browser-session/{operation}",
        headers=HEADERS,
        json={"revision": str(uuid4())},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"status", "revision"}
    return client.cookies.get(COOKIE_NAME)


def test_cookie_issuance_flags_and_server_authorized_history(
    cookie_client: tuple[TestClient, ScriptedModel],
    cookie_settings: BrowserSessionSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue opaque recovery and preserve history without repeated inference.

    Args:
        cookie_client: Real API with controlled model inference.
        cookie_settings: Stable encryption settings.
        caplog: Captured logs checked without exposing secret values.
    """
    client, model = cookie_client
    cookie = start_cookie(client)
    credential = unseal(cookie, cookie_settings)
    assert credential is not None and model.seen == []
    attributes = next(item for item in client.cookies.jar if item.name == COOKIE_NAME)
    assert attributes.has_nonstandard_attr("HttpOnly") and attributes.path == "/"
    assert attributes._rest["SameSite"] == "lax" and not attributes.secure
    assert attributes.expires > time.time()
    assert credential.access_token.get_secret_value() not in cookie
    headers = {"Authorization": f"Bearer {credential.access_token.get_secret_value()}"}
    path = f"/api/v1/conversations/{credential.conversation_id}"
    assert (
        client.post(
            path + "/stream", headers=headers, json={"message": "Consulta fictícia."}
        ).status_code
        == 200
    )
    history = client.get(path, headers=headers).json()["messages"]
    assert [item["role"] for item in history] == ["user", "assistant"]
    before = len(model.seen)
    for _ in range(2):
        assert (
            client.post("/browser-session/check", headers=HEADERS).json()["status"]
            == "ready"
        )
        assert client.get(path, headers=headers).json()["messages"] == history
        assert start_cookie(client) == cookie
    assert len(model.seen) == before
    assert (
        cookie not in caplog.text
        and credential.access_token.get_secret_value() not in caplog.text
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "tampered",
        "expired",
        "wrong-key",
        "malformed",
        "invalid-capability",
        "unknown-conversation",
    ],
)
def test_invalid_cookie_cannot_authorize_and_allows_a_fresh_session(
    cookie_client: tuple[TestClient, ScriptedModel],
    cookie_settings: BrowserSessionSettings,
    kind: str,
) -> None:
    """Reject invalid ciphertext uniformly without revealing prior conversation details.

    Args:
        cookie_client: Real API and controlled model.
        cookie_settings: Stable settings for producing bounded negative cases.
        kind: Missing, changed, old or unauthenticated cookie input.
    """
    client, model = cookie_client
    cookie = start_cookie(client)
    old = unseal(cookie, cookie_settings)
    client.cookies.clear()
    if kind == "tampered":
        cookie = cookie[:-10] + "AAAAAAAAAA"
    elif kind == "expired":
        payload = old.model_dump(mode="json")
        payload["access_token"] = old.access_token.get_secret_value()
        cookie = (
            cookie_settings.cipher()
            .encrypt_at_time(
                json.dumps(payload).encode(), int(time.time()) - COOKIE_TTL - 1
            )
            .decode()
        )
    elif kind == "wrong-key":
        cookie = Fernet(Fernet.generate_key()).encrypt(b"{}").decode()
    elif kind == "malformed":
        cookie = "not-encrypted"
    elif kind in ("invalid-capability", "unknown-conversation"):
        altered = BrowserCredential(
            conversation_id=uuid4()
            if kind == "unknown-conversation"
            else old.conversation_id,
            access_token="x" * 43 if kind == "invalid-capability" else old.access_token,
            revision=uuid4(),
        )
        cookie = seal(altered, cookie_settings)
    if kind != "missing":
        client.cookies.set(COOKIE_NAME, cookie)
    assert client.post("/browser-session/check", headers=HEADERS).json() == {
        "status": "empty"
    }
    client.cookies.clear()
    fresh = unseal(start_cookie(client), cookie_settings)
    assert fresh.conversation_id != old.conversation_id and model.seen == []


def test_new_delete_and_failed_delete_preserve_authorization_boundaries(
    cookie_client: tuple[TestClient, ScriptedModel],
    cookie_settings: BrowserSessionSettings,
) -> None:
    """Switch active references without deleting old threads, then clear after HTTP 204.

    Args:
        cookie_client: Browser-like API client.
        cookie_settings: Server-only decryption configuration.
    """
    client, model = cookie_client
    first = unseal(start_cookie(client), cookie_settings)
    second_cookie = start_cookie(client, "new")
    second = unseal(second_cookie, cookie_settings)
    assert first.conversation_id != second.conversation_id
    old_headers = {"Authorization": f"Bearer {first.access_token.get_secret_value()}"}
    assert (
        client.get(
            f"/api/v1/conversations/{first.conversation_id}", headers=old_headers
        ).status_code
        == 200
    )
    path = f"/api/v1/conversations/{second.conversation_id}"
    assert (
        client.delete(path, headers={"Authorization": "Bearer incorrect"}).status_code
        == 404
    )
    assert client.cookies.get(COOKIE_NAME) == second_cookie
    assert (
        client.post("/browser-session/check", headers=HEADERS).json()["status"]
        == "ready"
    )
    assert (
        client.delete(
            path,
            headers={
                "Authorization": f"Bearer {second.access_token.get_secret_value()}"
            },
        ).status_code
        == 204
    )
    assert client.post("/browser-session/check", headers=HEADERS).json() == {
        "status": "empty"
    }
    assert client.post("/browser-session/clear", headers=HEADERS).status_code == 200
    assert client.cookies.get(COOKIE_NAME) is None and model.seen == []


def test_reconstruction_and_separate_browser_cookie_jars(
    cookie_client: tuple[TestClient, ScriptedModel],
    cookie_settings: BrowserSessionSettings,
    demo_engine: Engine,
    state_store: ConversationStore,
    offline_settings: AgentSettings,
) -> None:
    """Recover through a new pool while another browser starts independently.

    Args:
        cookie_client: Initial API application and cookie jar.
        cookie_settings: Unchanged stable secret.
        demo_engine: Verified operational schema.
        state_store: Existing durable checkpoint namespace.
        offline_settings: Provider-free settings.
    """
    first, model = cookie_client
    cookie = start_cookie(first)
    credential = unseal(cookie, cookie_settings)
    with ConversationStore(state_store.schema) as reconstructed:
        with TestClient(
            create_app(
                settings=offline_settings,
                model=model,
                engine=demo_engine,
                store=reconstructed,
            )
        ) as other:
            assert other.post("/browser-session/check", headers=HEADERS).json() == {
                "status": "empty"
            }
            unrelated = unseal(start_cookie(other), cookie_settings)
            assert unrelated.conversation_id != credential.conversation_id
            other.cookies.clear()
            other.cookies.set(COOKIE_NAME, cookie)
            assert other.post("/browser-session/check", headers=HEADERS).json()[
                "revision"
            ] == str(credential.revision)
    assert model.seen == []


def test_database_outage_does_not_erase_cookie(
    cookie_client: tuple[TestClient, ScriptedModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve valid recovery credentials on transient PostgreSQL failures.

    Args:
        cookie_client: Browser client with a valid active cookie.
        monkeypatch: Controlled persistence outage.
    """
    client, _ = cookie_client
    cookie = start_cookie(client)

    def unavailable(*args: object, **kwargs: object) -> None:
        """Fail before authorization without exposing connection diagnostics.

        Args:
            args: Uninspected persistence arguments.
            kwargs: Uninspected persistence options.

        Raises:
            AgentFailure: Always, as a classified transient database outage.
        """
        raise AgentFailure(ErrorCode.DATABASE)

    monkeypatch.setattr(ConversationStore, "authorized", unavailable)
    response = client.post("/browser-session/check", headers=HEADERS)
    assert response.status_code == 503 and "set-cookie" not in response.headers
    assert client.cookies.get(COOKIE_NAME) == cookie


def test_origin_csrf_redirect_and_production_cookie_rules(
    cookie_client: tuple[TestClient, ScriptedModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject cross-origin mutations and retain Secure despite proxy HTTP transport.

    Args:
        cookie_client: Controlled real API.
        monkeypatch: Explicit production-origin settings.
    """
    client, model = cookie_client
    for headers in (
        {},
        {"Origin": HEADERS["Origin"]},
        {**HEADERS, "Origin": "https://attacker.example"},
    ):
        assert (
            client.post(
                "/browser-session/start",
                headers=headers,
                json={"revision": str(uuid4())},
            ).status_code
            == 401
        )
    assert client.get("/browser-session/new").status_code == 405
    assert (
        client.post(
            "/browser-session/start?next=https://attacker.example",
            headers=HEADERS,
            json={"revision": str(uuid4())},
        ).status_code
        == 401
    )
    monkeypatch.setenv("BROWSER_APP_ORIGIN", "https://music.example.test")
    monkeypatch.setenv("BROWSER_SESSION_API_ORIGIN", "https://music.example.test")
    monkeypatch.setenv("BROWSER_SESSION_ALLOW_HTTP_LOCALHOST", "false")
    response = client.post(
        "/browser-session/start",
        headers={
            **HEADERS,
            "Origin": "https://music.example.test",
            "X-Forwarded-Proto": "http",
        },
        json={"revision": str(uuid4())},
    )
    assert response.status_code == 200
    header = response.headers["set-cookie"]
    assert "Secure" in header and "HttpOnly" in header and "SameSite=lax" in header
    assert "Domain=" not in header and f"Max-Age={COOKIE_TTL}" in header
    assert (
        response.headers["access-control-allow-origin"] == "https://music.example.test"
    )
    assert "location" not in response.headers and model.seen == []


def test_invalid_and_missing_production_secrets_fail_closed(
    cookie_settings: BrowserSessionSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing/weak production secrets and implicit insecure deployment settings.

    Args:
        cookie_settings: Baseline valid test key configuration.
        monkeypatch: Scoped production configuration variants.
    """
    monkeypatch.setenv("BROWSER_APP_ORIGIN", "https://music.example.test")
    monkeypatch.setenv("BROWSER_SESSION_API_ORIGIN", "https://music.example.test")
    for value in ("", "invalid", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="):
        monkeypatch.setenv("BROWSER_SESSION_KEY", value)
        with pytest.raises(ValidationError):
            BrowserSessionSettings()


def test_missing_key_never_issues_a_cookie(
    cookie_client: tuple[TestClient, ScriptedModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep cookie issuance closed when persistent encryption is unconfigured.

    Args:
        cookie_client: Controlled API without external inference.
        monkeypatch: Scoped removal of the encryption secret.
    """
    client, model = cookie_client
    monkeypatch.delenv("BROWSER_SESSION_KEY")
    response = client.post(
        "/browser-session/start", headers=HEADERS, json={"revision": str(uuid4())}
    )
    assert response.status_code == 503 and "set-cookie" not in response.headers
    assert model.seen == []
