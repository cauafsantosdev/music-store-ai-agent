"""Minimal Docker CLI for synthetic data and durable capability-protected chat."""

import argparse
from collections.abc import Sequence
from getpass import getpass

from music_store_agent.agent.config import load_settings
from music_store_agent.agent.demo import initialize_demo, select_dataset
from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.agent.service import StoreAgent
from music_store_agent.database.session import create_database_engine


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize data or manage a new or explicitly resumed durable conversation.

    Args:
        argv: Optional CLI arguments, otherwise taken from the process.

    Returns:
        Zero for normal exit; one for a sanitized startup failure.

    Raises:
        SystemExit: For argparse help or invalid command syntax.
    """
    parser = argparse.ArgumentParser(
        description="Demonstração sintética do assistente do Empório da Música.",
        epilog=(
            "Inicialize com --init-demo. "
            "Não informe dados pessoais ou documentos reais."
        ),
    )
    parser.add_argument(
        "--init-demo",
        action="store_true",
        help="Inicializar o esquema vazio e o PDF do conjunto sintético selecionado.",
    )
    parser.add_argument(
        "--init-state", action="store_true", help="Inicializar a memória PostgreSQL."
    )
    parser.add_argument(
        "--resume", metavar="UUID", help="Retomar uma conversa com token oculto."
    )
    parser.add_argument(
        "--show-access-token",
        action="store_true",
        help="Exibir explicitamente o token de uma nova conversa local.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.show_access_token:
        parser.error("--show-access-token requer uma nova conversa.")
    engine = None
    agent: StoreAgent | None = None
    try:
        settings = load_settings()
        if args.init_state:
            with ConversationStore(
                settings.ai_state_schema, dataset_mode=settings.ai_dataset_mode
            ) as store:
                store.initialize(admission=settings.ai_admission_enabled)
            print("Memória PostgreSQL inicializada.")
            return 0
        engine = select_dataset(
            create_database_engine(schema=settings.ai_demo_schema), settings
        )
        if args.init_demo:
            initialize_demo(engine)
            if settings.ai_dataset_mode == "challenge":
                print(
                    "Conjunto challenge confirmado: 65 produtos; PDF original indexado."
                )
                return 0
            print(
                "Demonstração sintética confirmada. "
                "Pedido de teste: 303; email: customer-203@example.test."
            )
            return 0
        agent = create_store_agent(engine, settings=settings)
        conversation = (
            agent.resume_conversation(args.resume, getpass("Token de acesso: "))
            if args.resume
            else agent.create_conversation()
        )
        print(f"Conversa: {conversation.conversation_id}")
        if args.show_access_token:
            print(f"Token local: {conversation.access_token.get_secret_value()}")
        print(
            "Assistente virtual — somente dados sintéticos. Digite sair para encerrar."
        )
        while True:
            try:
                message = input("Cliente: ")
            except (EOFError, KeyboardInterrupt):
                print("\nAtendimento encerrado.")
                return 0
            if message.strip().casefold() in {"sair", "exit", "quit"}:
                return 0
            print(f"Assistente: {conversation.send(message).text}")
    except (EOFError, KeyboardInterrupt):
        print("\nAtendimento encerrado.")
        return 0
    except AgentFailure as error:
        print(str(error))
        return 1
    finally:
        if agent is not None:
            agent.close()
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
