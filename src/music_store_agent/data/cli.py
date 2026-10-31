"""Initialize and seed PostgreSQL through explicit, privacy-preserving commands."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.data.ingestion import IngestionError, seed_data
from music_store_agent.database.session import create_database_engine, initialize_schema


def main(argv: Sequence[str] | None = None) -> int:
    """Run schema initialization or transactional CSV ingestion.

    Args:
        argv: CLI arguments; omitted to read the process arguments.

    Returns:
        Zero on success, or one after a sanitized operational error.

    Raises:
        SystemExit: For argparse help or invalid CLI syntax.
    """
    parser = argparse.ArgumentParser(
        description="Initialize or seed PostgreSQL using the existing PG* environment."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="Create missing tables without deleting data."
    )
    seed = commands.add_parser("seed", help="Atomically upsert all six CSV datasets.")
    for command in (init, seed):
        command.add_argument(
            "--schema",
            default="public",
            help=(
                "PostgreSQL schema (default: public); "
                "use a separate schema for verification."
            ),
        )
    seed.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory recursively containing canonical or original prefixed CSVs.",
    )
    args = parser.parse_args(argv)
    engine = None
    try:
        engine = create_database_engine(schema=args.schema)
        if args.command == "init":
            initialize_schema(engine)
            print("Schema initialization completed; existing records preserved.")
        else:
            counts = seed_data(engine, args.source_dir)
            print(
                "Import committed: "
                + ", ".join(f"{name}={count}" for name, count in counts.items())
            )
    except IngestionError as error:
        print(f"Import failed; no changes committed: {error}", file=sys.stderr)
        return 1
    except (SQLAlchemyError, ValueError):
        # Raw driver errors can contain credentials or row values; do not print them.
        print(
            "Database operation failed; "
            "check PG* configuration and schema permissions.",
            file=sys.stderr,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
