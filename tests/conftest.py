"""PostgreSQL fixtures isolated from development tables and private source data."""

from collections.abc import Iterator
from pathlib import Path
from shutil import copytree
from uuid import uuid4

import pytest
from sqlalchemy import Engine
from sqlalchemy.schema import DropSchema

from music_store_agent.database.session import create_database_engine, initialize_schema


@pytest.fixture(autouse=True)
def sample_test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep public tests independent of the selected local application dataset.

    Args:
        monkeypatch: Scoped defaults; challenge tests select their mode explicitly.
    """
    monkeypatch.setenv("AI_DATASET_MODE", "sample")
    monkeypatch.setenv("AI_DEMO_SCHEMA", "")
    monkeypatch.setenv("AI_STATE_SCHEMA", "")


@pytest.fixture
def database_engine() -> Iterator[Engine]:
    """Provide an empty unique namespace and remove only that namespace afterward.

    Yields:
        A lazy engine scoped to a generated test schema.
    """
    schema = f"test_music_store_{uuid4().hex}"
    engine = create_database_engine(schema=schema)
    try:
        yield engine
    finally:
        # The name is generated locally; development schemas are never dropped.
        try:
            with engine.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        finally:
            engine.dispose()


@pytest.fixture
def initialized_engine(database_engine: Engine) -> Engine:
    """Create all six application tables in an isolated PostgreSQL schema.

    Args:
        database_engine: Per-test engine targeting a disposable schema.

    Returns:
        The initialized test engine.
    """
    initialize_schema(database_engine)
    return database_engine


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Copy only invented public CSVs for independent per-test modifications.

    Args:
        tmp_path: Pytest-managed temporary directory inside the container.

    Returns:
        A mutable copy of the six public sample files.
    """
    samples = Path(__file__).resolve().parents[1] / "data" / "sample"
    return Path(copytree(samples, tmp_path / "source"))
