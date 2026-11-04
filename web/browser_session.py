"""Supported HttpOnly-cookie bootstrap and server-only authorized history recovery."""

from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v2 as components
from pydantic import ValidationError

from music_store_agent.browser_session import (
    COOKIE_NAME,
    BrowserSessionSettings,
    unseal,
)
from web.client import BackendClient, Capability, ClientError, FrontendSettings
from web.state import ChatState

bridge = components.component(
    "browser_session_bridge", js=Path(__file__).with_suffix(".js").read_text()
)


def changed() -> None:
    """Permit supported component state updates without executing side effects."""


def retry_recovery() -> None:
    """Allow an explicit recovery attempt after a transient error."""
    st.session_state.pop("browser_recovery_error", None)


def handshake_cookie() -> str | None:
    """Read the supported server-side cookie context from the WebSocket handshake.

    Returns:
        Opaque HttpOnly ciphertext, never returned to a browser component.
    """
    return st.context.cookies.get(COOKIE_NAME)


def prepare_browser_session(state: ChatState, http: FrontendSettings) -> bool:
    """Restore before rendering history or accepting input, without model execution.

    The handshake cookie is read-only and can change only after a real browser
    reload. The component never receives credentials, cookies or conversation text.

    Args:
        state: Session-local display and authorization state.
        http: Existing internal HTTP configuration.

    Returns:
        Whether ordinary chat rendering may continue.
    """
    try:
        settings = BrowserSessionSettings()
    except ValidationError:
        st.error("A recuperação de conversas não está disponível nesta configuração.")
        return False
    if not settings.enabled:
        st.caption(
            "A recuperação após recarregar a página "
            "não está configurada neste ambiente."
        )
        return True
    state.browser_persistent = True
    if state.browser_ready and state.browser_operation is None:
        return True
    credential = unseal(handshake_cookie(), settings)
    if state.browser_operation == "clear" and credential is None:
        state.browser_operation = "start"
        state.browser_revision = str(uuid4())
        state.browser_ready = False
    if credential is not None and (
        state.browser_operation in (None, "start")
        or str(credential.revision) == state.browser_revision
    ):
        if "browser_recovery_error" in st.session_state:
            st.error(
                "Não foi possível restaurar a conversa agora. O acesso foi preservado."
            )
            st.button("Tentar restaurar", on_click=retry_recovery)
            return False
        try:
            progress = st.empty()
            progress.info("Restaurando conversa...")
            capability = Capability(
                conversation_id=credential.conversation_id,
                access_token=credential.access_token,
            )
            with BackendClient(http) as client:
                messages = client.history(capability)
            state.reset()
            state.capability = capability
            state.messages = messages
            state.submitted = None
            state.action = None
            state.needs_refresh = False
            state.browser_operation = None
            state.browser_ready = True
            progress.empty()
            return True
        except ClientError as error:
            progress.empty()
            if error.code != "session":
                st.session_state.browser_recovery_error = True
                st.error(
                    "Não foi possível restaurar a conversa agora. "
                    "O acesso foi preservado."
                )
                st.button("Tentar restaurar", on_click=retry_recovery)
                return False
            # Invalid/deleted capabilities cannot restore data; only explicit
            # browser issuance can replace this credential with a fresh session.
    if state.browser_operation is None:
        state.browser_operation = "start"
        state.browser_revision = str(uuid4())
    st.info("Restaurando conversa...")
    result = bridge(
        key=f"browser_{state.browser_revision}",
        data={
            "endpoint": settings.endpoint,
            "operation": state.browser_operation,
            "revision": state.browser_revision,
        },
        on_failed_change=changed,
    )
    if result.failed:
        st.warning(
            "Não foi possível confirmar a recuperação neste navegador. "
            "Verifique a conexão e a permissão para cookies."
        )
        # No mutation is retried automatically after ambiguous delivery.
        if st.button("Tentar restaurar"):
            state.browser_revision = str(uuid4())
            st.rerun()
    return False
