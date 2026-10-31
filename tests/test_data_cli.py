"""Verify operational CLI behavior using disposable PostgreSQL schemas."""

from pathlib import Path

from pytest import CaptureFixture
from sqlalchemy import Engine

from music_store_agent.data.cli import main


def test_cli_initializes_and_seeds(
    database_engine: Engine, source_dir: Path, capsys: CaptureFixture[str]
) -> None:
    """Report repeatable initialization and import counts without printing row data.

    Args:
        database_engine: Engine identifying the isolated schema to initialize.
        source_dir: Synthetic CSV fixtures.
        capsys: Captured CLI output.
    """
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    assert main(["init", "--schema", schema]) == 0
    assert main(["init", "--schema", schema]) == 0
    for _ in range(2):
        assert main(["seed", "--schema", schema, "--source-dir", str(source_dir)]) == 0
    output = capsys.readouterr()
    assert "Schema initialization completed" in output.out
    assert (
        output.out.count(
            "categories=2, products=3, customers=2, "
            "orders=2, order_items=3, promotions=4"
        )
        == 2
    )
    assert not output.err


def test_cli_reports_sanitized_failure(
    database_engine: Engine, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """Exit unsuccessfully for absent datasets without a traceback or private values.

    Args:
        database_engine: Engine identifying an isolated schema.
        tmp_path: Empty source directory.
        capsys: Captured CLI output.
    """
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    assert main(["seed", "--schema", schema, "--source-dir", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "Import failed; no changes committed" in output.err
    assert "categories.csv: required dataset is missing" in output.err
