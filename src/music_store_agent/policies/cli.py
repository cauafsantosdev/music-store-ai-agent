"""Explicit local PDF indexing with concise, sanitized operational output."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from music_store_agent.database.session import create_database_engine
from music_store_agent.policies.errors import PolicyExtractionError, PolicyIndexError
from music_store_agent.policies.indexing import index_policy_pdf


def main(argv: Sequence[str] | None = None) -> int:
    """Parse an index command without automatically accessing private source files.

    Args:
        argv: CLI arguments; omitted to read the process argument list.

    Returns:
        Zero after an atomic index replacement, or one after a sanitized failure.

    Raises:
        SystemExit: For CLI help or invalid command syntax.
    """
    parser = argparse.ArgumentParser(
        description="Index a local policy PDF without logging its content."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "index", help="Atomically replace one logical policy index."
    )
    command.add_argument(
        "--source", type=Path, required=True, help="Explicit path to a text-based PDF."
    )
    command.add_argument(
        "--schema", default="public", help="Target PostgreSQL schema (default: public)."
    )
    command.add_argument(
        "--document-id",
        default="store-policy-manual",
        help="Stable logical document identifier.",
    )
    command.add_argument(
        "--label",
        default="Store Policy Manual",
        help="Safe public source label without a path.",
    )
    args = parser.parse_args(argv)
    engine = None
    try:
        engine = create_database_engine(schema=args.schema)
        result = index_policy_pdf(
            engine, args.source, document_id=args.document_id, label=args.label
        )
        print(
            f"Policy index committed: pages={result.page_count}, "
            f"sections={result.section_count}."
        )
    except (PolicyExtractionError, PolicyIndexError, ValueError) as error:
        print(f"Policy indexing failed: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
