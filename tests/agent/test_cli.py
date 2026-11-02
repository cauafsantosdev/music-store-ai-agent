"""Minimal CLI startup, non-sensitive output, and interruption checks."""

import pytest
from conftest import ScriptedModel
from langchain_core.messages import AIMessage
from sqlalchemy import Engine

from music_store_agent.agent import cli
from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.agent.service import Conversation, StoreAgent


@pytest.fixture
def cli_demo(demo_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind CLI startup to the existing disposable synthetic database.

    Args:
        demo_engine: Real PostgreSQL namespace containing verified synthetic data.
        monkeypatch: Scoped CLI dependency and environment overrides.
    """
    monkeypatch.setenv("GROQ_API_KEY", "")

    def selected_engine(*, schema: str) -> Engine:
        """Use the isolated namespace instead of touching the persistent CLI demo.

        Args:
            schema: Configured CLI namespace.

        Returns:
            The disposable verified database engine.
        """
        return demo_engine

    monkeypatch.setattr(cli, "create_database_engine", selected_engine)


def test_cli_initializes_without_key_and_rejects_missing_live_key(
    cli_demo: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Initialize public data without credentials and sanitize live startup failures.

    Args:
        cli_demo: Isolated CLI database and blank provider key.
        capsys: Captured terminal output.
    """
    assert cli.main(["--init-demo"]) == 0
    assert "customer-203@example.test" in capsys.readouterr().out
    assert cli.main([]) == 1
    output = capsys.readouterr()
    assert "Configure GROQ_API_KEY" in output.out
    assert output.err == ""


def test_cli_initializes_state_without_provider_credentials(
    cli_demo: None, state_namespace: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Initialize persistent state explicitly and idempotently without a model.

    Args:
        cli_demo: Blank provider credential and isolated operational engine.
        state_namespace: Disposable state namespace selected through configuration.
        capsys: Captured terminal output.
    """
    assert cli.main(["--init-state"]) == 0
    assert cli.main(["--init-state"]) == 0
    with ConversationStore(state_namespace) as store:
        store.check_ready()
    output = capsys.readouterr()
    assert "Memória PostgreSQL inicializada." in output.out
    assert "Token local:" not in output.out
    assert output.err == ""


@pytest.mark.parametrize("show_token", [False, True])
def test_cli_creation_and_explicit_resume(
    cli_demo: None,
    demo_engine: Engine,
    state_store: ConversationStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    show_token: bool,
) -> None:
    """Resume native context through hidden token entry without automatic token output.

    Args:
        cli_demo: Isolated CLI operational engine.
        demo_engine: Verified synthetic catalog.
        state_store: Real PostgreSQL conversation state.
        monkeypatch: Controlled models and terminal input boundaries.
        capsys: Captured terminal output.
        show_token: Whether explicit development token display was requested.
    """
    models: list[ScriptedModel] = []
    sessions: list[Conversation] = []
    original_create = StoreAgent.create_conversation

    def offline_agent(engine: Engine, *, settings: AgentSettings) -> StoreAgent:
        """Construct a real persistent graph with a controlled chat model.

        Args:
            engine: Verified operational engine selected by the CLI.
            settings: Validated CLI configuration.

        Returns:
            A real service using only fake inference.
        """
        assert engine is demo_engine
        model = ScriptedModel(responses=[AIMessage(content="Atendimento fictício.")])
        models.append(model)
        return create_store_agent(
            engine, settings=settings, model=model, state_store=state_store
        )

    def capture_session(self: StoreAgent) -> Conversation:
        """Capture the returned capability without changing real session creation.

        Args:
            self: Actual persistent service.

        Returns:
            The unchanged durable session handle.
        """
        session = original_create(self)
        sessions.append(session)
        return session

    inputs = iter(["Olá.", "sair", "Continue.", "sair"])

    def read_input(prompt: str) -> str:
        """Provide bounded synthetic CLI input.

        Args:
            prompt: Local input prompt.

        Returns:
            The next synthetic line.
        """
        return next(inputs)

    def read_token(prompt: str) -> str:
        """Supply a capability at the hidden input boundary without printing it.

        Args:
            prompt: Hidden token prompt.

        Returns:
            The original session's capability.
        """
        return sessions[0].access_token.get_secret_value()

    monkeypatch.setattr(cli, "create_store_agent", offline_agent)
    monkeypatch.setattr(StoreAgent, "create_conversation", capture_session)
    monkeypatch.setattr("builtins.input", read_input)
    monkeypatch.setattr(cli, "getpass", read_token)
    assert cli.main(["--show-access-token"] if show_token else []) == 0
    output = capsys.readouterr()
    token = sessions[0].access_token.get_secret_value()
    assert (token in output.out) == show_token
    assert cli.main(["--resume", str(sessions[0].conversation_id)]) == 0
    output = capsys.readouterr()
    assert token not in output.out + output.err
    assert len(sessions) == 1
    assert any(message.content == "Olá." for message in models[-1].seen[0])


@pytest.mark.parametrize("during_send", [False, True])
def test_cli_interrupts_exit_cleanly(
    cli_demo: None,
    demo_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    during_send: bool,
) -> None:
    """Handle Ctrl+C both at the input prompt and during a model turn.

    Args:
        cli_demo: Isolated CLI database and blank provider key.
        demo_engine: Verified synthetic database for real agent construction.
        monkeypatch: Scoped model construction and terminal interrupt boundaries.
        capsys: Captured terminal output.
        during_send: Whether to interrupt while processing a message.
    """

    def offline_agent(engine: Engine, *, settings: AgentSettings) -> StoreAgent:
        """Construct the real graph with an offline model.

        Args:
            engine: Verified synthetic CLI engine.
            settings: Environment-derived CLI settings.

        Returns:
            A real store agent with provider-free generation.
        """
        assert engine is demo_engine
        return create_store_agent(
            engine,
            settings=settings,
            model=ScriptedModel(responses=[AIMessage(content="Olá.")]),
        )

    def interrupted_input(prompt: str) -> str:
        """Simulate terminal input or an interrupt at the prompt.

        Args:
            prompt: Normal CLI input prompt.

        Returns:
            Non-sensitive input when interrupting inside send instead.

        Raises:
            KeyboardInterrupt: If the interrupt occurs while awaiting input.
        """
        if during_send:
            return "Olá."
        raise KeyboardInterrupt

    def interrupted_send(self: Conversation, message: str) -> None:
        """Simulate cancellation during a provider turn.

        Args:
            self: Current CLI conversation.
            message: Non-sensitive demonstration text.

        Raises:
            KeyboardInterrupt: Always, without contacting a provider.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "create_store_agent", offline_agent)
    monkeypatch.setattr("builtins.input", interrupted_input)
    if during_send:
        monkeypatch.setattr(Conversation, "send", interrupted_send)
    assert cli.main([]) == 0
    output = capsys.readouterr()
    assert "Atendimento encerrado." in output.out
    assert output.err == ""
