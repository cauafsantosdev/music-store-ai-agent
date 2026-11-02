"""Minimal Docker-oriented synthetic-demo initialization and process-local chat."""

import argparse
from collections.abc import Sequence

from music_store_agent.agent.config import load_settings
from music_store_agent.agent.demo import initialize_demo
from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.database.session import create_database_engine


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize a confirmed demo or interact with one new in-process conversation.

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
        help="Criar somente um esquema vazio usando amostras públicas verificadas.",
    )
    args = parser.parse_args(argv)
    engine = None
    try:
        settings = load_settings()
        engine = create_database_engine(schema=settings.ai_demo_schema)
        if args.init_demo:
            initialize_demo(engine)
            print(
                "Demonstração sintética confirmada. "
                "Pedido de teste: 303; email: customer-203@example.test."
            )
            return 0
        conversation = create_store_agent(engine, settings=settings).new_conversation()
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
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
