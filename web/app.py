"""Portuguese dark-themed chat using only the public FastAPI HTTP interface."""

import os
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from web.browser_session import prepare_browser_session
from web.client import BackendClient, FrontendSettings
from web.state import Action, ChatState

# These questions reference only pinned public CSV/PDF assets and the documented
# synthetic association added by the demo initializer, never private source files.
SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("Instrumentos disponíveis", "O Sample Acoustic está disponível?"),
    (
        "Promoções e PIX",
        "Qual é a promoção ativa do Sample Acoustic e como fica o preço no PIX?",
    ),
    (
        "Envio e pagamentos",
        "Quais são as formas de pagamento e as regras de envio do cenário fictício?",
    ),
    (
        "Pedido fictício",
        (
            "Qual é o status do pedido fictício 303, associado ao email "
            "customer-203@example.test?"
        ),
    ),
)
CHALLENGE_SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("Instrumentos disponíveis", "Quais violões estão disponíveis?"),
    ("Promoções e PIX", "Quais instrumentos têm promoção ativa?"),
    ("Envio e pagamentos", "Quais são as formas de pagamento e as regras de envio?"),
    (
        "Pedido fictício",
        "Quais dados são necessários para consultar um pedido fictício?",
    ),
)


def chat_state() -> ChatState:
    """Keep capabilities in this server-side Streamlit session, never a global cache.

    Returns:
        The current browser session's display state.
    """
    if "music_store_chat" not in st.session_state:
        st.session_state.music_store_chat = ChatState()
    return st.session_state.music_store_chat


def queue_message(message: str | None = None) -> None:
    """Queue either a suggestion or ordinary chat input through the same workflow.

    Args:
        message: Explicit suggestion text, otherwise the submitted widget value.
    """
    value = message if message is not None else st.session_state.get("chat_input", "")
    if isinstance(value, str):
        chat_state().queue(value)


def new_conversation() -> None:
    """Reset local state without deleting the previous backend conversation."""
    state = chat_state()
    if not state.busy:
        if state.browser_persistent:
            state.browser_operation = "new"
            state.browser_revision = str(uuid4())
        else:
            state.reset()


def queue_action(action: Action) -> None:
    """Queue an explicit non-message action without side effects in callbacks.

    Args:
        action: History refresh, confirmed deletion or status verification.
    """
    state = chat_state()
    if not state.busy and state.action is None:
        state.action = action


def confirm_deletion(value: bool) -> None:
    """Toggle local deletion confirmation without sending a request.

    Args:
        value: Whether the confirmation prompt should be visible.
    """
    chat_state().confirm_delete = value


def render_sidebar(state: ChatState) -> None:
    """Render explicit session controls and a compact privacy notice.

    Args:
        state: Session-local state, with credentials never passed to widgets.
    """
    waiting = state.busy or state.action is not None
    with st.sidebar:
        st.markdown("### Empório da Música")
        st.caption("Instrumentos e atendimento, no mesmo compasso.")
        st.button(
            "Nova conversa",
            icon=":material/add:",
            type="primary",
            width="stretch",
            on_click=new_conversation,
            disabled=waiting,
            key="new_conversation",
        )
        st.divider()
        st.markdown("#### Conversa atual")
        if state.capability:
            st.caption(f"Sessão · {str(state.capability.conversation_id)[:8]}")
            st.caption("O histórico permanece disponível nesta sessão do navegador.")
        else:
            st.caption("Uma conversa será iniciada ao enviar a primeira mensagem.")
        st.button(
            "Atualizar histórico",
            icon=":material/refresh:",
            width="stretch",
            on_click=queue_action,
            args=("history",),
            disabled=waiting or state.capability is None,
            key="refresh_history",
            help="Confirme o resultado de uma consulta sem reenviar a mensagem.",
        )
        if state.confirm_delete:
            st.warning(
                "Excluir esta conversa e seu histórico? Esta ação não pode ser "
                "desfeita."
            )
            st.button(
                "Confirmar exclusão",
                on_click=queue_action,
                args=("delete",),
                disabled=waiting,
                type="primary",
                width="stretch",
                key="confirm_delete",
            )
            st.button(
                "Cancelar",
                on_click=confirm_deletion,
                args=(False,),
                disabled=waiting,
                width="stretch",
                key="cancel_delete",
            )
        else:
            st.button(
                "Excluir conversa",
                icon=":material/delete:",
                width="stretch",
                on_click=confirm_deletion,
                args=(True,),
                disabled=waiting or state.capability is None,
                key="delete_conversation",
            )
        st.divider()
        st.button(
            "Verificar conexão",
            on_click=queue_action,
            args=("status",),
            disabled=waiting,
            width="stretch",
            key="check_status",
        )
        if state.backend_ready is True:
            st.caption(":material/check_circle: Serviço pronto para consultas.")
        elif state.backend_ready is False:
            st.caption(":material/info: Serviço temporariamente indisponível.")
        st.caption("A verificação de conexão não garante a conclusão de uma consulta.")
        st.divider()
        st.caption("Ambiente de demonstração com dados fictícios.")
        st.caption(
            "As mensagens são processadas pela API externa da Groq. Não "
            "envie dados pessoais ou sensíveis."
        )


def render_welcome(state: ChatState) -> None:
    """Show restrained examples grounded in verified synthetic fixtures.

    Args:
        state: Current submission state used to disable duplicate interactions.
    """
    st.subheader("Como posso ajudar?")
    st.write(
        "Consulte instrumentos, promoções, pagamentos, políticas da "
        "loja e pedidos fictícios."
    )
    st.caption("Escolha uma sugestão ou escreva sua pergunta abaixo.")
    columns = st.columns(2)
    challenge = os.getenv("AI_DATASET_MODE", "sample") == "challenge"
    suggestions = CHALLENGE_SUGGESTIONS if challenge else SUGGESTIONS
    for index, (label, question) in enumerate(suggestions):
        with columns[index % 2]:
            st.button(
                label,
                key=f"suggestion_{index}",
                on_click=queue_message,
                args=(question,),
                width="stretch",
                disabled=state.busy or state.action is not None or state.needs_refresh,
            )
    if not challenge:
        st.caption(
            "O exemplo de rastreio usa exclusivamente o pedido 303 e o "
            "email customer-203@example.test, ambos fictícios."
        )


def render_messages(state: ChatState) -> None:
    """Render public display text without native tool or provider metadata.

    Args:
        state: Session-local public display history and optional unconfirmed input.
    """
    for message in state.messages:
        user = message.role == "user"
        with st.chat_message(
            "user" if user else "assistant",
            avatar=":material/person:" if user else ":material/music_note:",
        ):
            st.caption("Você" if user else "Assistente virtual · Resposta confirmada")
            st.markdown(state.visible_text(message.text), unsafe_allow_html=False)
    if state.submitted is not None:
        with st.chat_message("user", avatar=":material/person:"):
            st.caption("Você")
            st.markdown(state.visible_text(state.submitted), unsafe_allow_html=False)
            if state.error:
                st.caption("Solicitação não confirmada.")
                st.caption("Confira o histórico antes de reenviar esta mensagem.")


def execute_action(state: ChatState, settings: FrontendSettings) -> None:
    """Process one consumed user action with explicit progress and no auto-retries.

    Args:
        state: Mutable session-local submission state.
        settings: Bounded internal HTTP configuration.
    """
    label = {
        "send": "Consultando informações...",
        "history": "Atualizando histórico...",
        "delete": "Excluindo conversa...",
        "status": "Verificando conexão...",
    }[state.action]
    sending = state.action == "send"
    with st.status(label, expanded=sending) as progress:
        if sending:
            st.write(
                "A consulta pode levar alguns instantes. A resposta completa "
                "aparecerá após a confirmação do histórico."
            )

        def processing() -> None:
            """Display fixed progress text rather than untrusted SSE status strings."""
            progress.update(label="Consulta em andamento...", expanded=True)

        with BackendClient(settings) as client:
            state.execute(client, processing)
        progress.update(
            label="Solicitação não concluída." if state.error else "Concluído.",
            state="error" if state.error else "complete",
        )
    st.rerun()


def main() -> None:
    """Render a Portuguese chat without network or model requests on initial load."""
    st.set_page_config(
        page_title="Empório da Música · Assistente virtual",
        page_icon=":material/music_note:",
        layout="centered",
    )
    st.title("♫ Empório da Música")
    st.caption("Assistente virtual · Instrumentos, condições de pagamento e suporte.")
    st.badge("Demonstração", icon=":material/science:", color="gray")
    try:
        settings = FrontendSettings()
    except ValidationError:
        st.error("A conexão com o atendimento ainda não está configurada corretamente.")
        return
    state = chat_state()
    if not prepare_browser_session(state, settings):
        return
    if state.busy:
        # A consumed request interrupted by a rerun must never be replayed.
        state.interrupt()
    render_sidebar(state)
    if not state.messages and state.submitted is None:
        render_welcome(state)
    render_messages(state)
    if state.error:
        st.error(str(state.error))
        if state.error.request_id:
            st.caption(f"Referência do atendimento: {state.error.request_id}")
        if state.needs_refresh:
            st.info(
                "Atualize o histórico para confirmar o resultado ou inicie uma "
                "nova conversa. Nenhuma mensagem será reenviada automaticamente."
            )
    elif state.notice:
        st.success(state.notice)
    st.chat_input(
        "Digite sua mensagem...",
        key="chat_input",
        max_chars=2000,
        disabled=state.busy or state.action is not None or state.needs_refresh,
        on_submit=queue_message,
    )
    if state.action is not None:
        execute_action(state, settings)


if __name__ == "__main__":
    main()
