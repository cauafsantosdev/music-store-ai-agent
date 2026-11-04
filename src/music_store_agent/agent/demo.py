"""Pinned synthetic provenance and read-only snapshots for every model-facing tool."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

from sqlalchemy import Connection, Engine, func, insert, inspect, select
from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.datasets import (
    CHALLENGE_CHECKSUMS,
    CHALLENGE_FINGERPRINT,
    DatasetMode,
)
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.data.ingestion import seed_data
from music_store_agent.database.models import Base, Customer, Order
from music_store_agent.database.policies import PolicyDocument, PolicySection
from music_store_agent.database.session import initialize_schema
from music_store_agent.policies.indexing import index_policy_pdf

SAMPLE_DIRECTORY: Path = Path(__file__).resolve().parents[3] / "data" / "sample"
# Filled from the versioned public fixture bytes, never from a selected database.
SOURCE_CHECKSUMS: dict[str, str] = {
    "categories.csv": (
        "fa5c738a49929909f67e1990076de3a2087f9f7478d3862260d068818b28d645"
    ),
    "customers.csv": "e5aea6ec2c40340c37e3bbfb9c5eaa294d37140577058fc2d24e81e23c442bf9",
    "order_items.csv": (
        "e982c311eb2453b76823fbd41a810aa46e7b2f5122a530002d38e07b57fec42c"
    ),
    "orders.csv": "f57fa9691a74593ac78213f2d4b27bfea661032df5ee2686a9cfd64e0eae35a1",
    "products.csv": "1fb4c02c986435c5820d0a2f03117842fc70ec7936c4e8379a619c90a02afdcc",
    "promotions.csv": (
        "996dea7e3690d68e7bcd75646ec8ef6f6cee67c118f947f1a2dad94326e9dc63"
    ),
    "store_policies.pdf": (
        "88e539cfd5fa169b90d77d9c2a394e14f30e07e6ed5c8155ab6f9d97c1b2dac9"
    ),
}
DEMO_FINGERPRINT: str = (
    "054624a8352e5dc22511c22a822d08f22ef48749e59c80d3d801f5fca17cfdf0"
)


def dataset_mode(bind: Engine | Connection) -> DatasetMode:
    """Read explicit trusted engine selection without consulting mutable environment.

    Args:
        bind: Operational engine or its active transaction.

    Returns:
        Explicit challenge selection, otherwise the original sample default.

    Raises:
        AgentFailure: If an unsupported mode is attached to the engine.
    """
    mode = bind.get_execution_options().get("ai_dataset_mode", "sample")
    if mode not in ("sample", "challenge"):
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    return cast(DatasetMode, mode)


def select_dataset(engine: Engine, settings: AgentSettings) -> Engine:
    """Attach explicit mode and mounted source selection to a deployment engine.

    Args:
        engine: Engine already targeting the configured operational schema.
        settings: Validated deployment configuration.

    Returns:
        The sample engine unchanged or a challenge engine with trusted options.
    """
    if settings.ai_dataset_mode == "sample":
        return engine
    return engine.execution_options(
        ai_dataset_mode="challenge",
        ai_challenge_directory=settings.ai_challenge_directory,
    )


def challenge_sources(bind: Engine | Connection) -> Path:
    """Verify all seven approved source hashes before importing or authorizing tools.

    Args:
        bind: Explicit challenge engine with a read-only mounted source directory.

    Returns:
        The approved source root without exposing source contents.

    Raises:
        AgentFailure: If selection, file availability or any checksum is invalid.
    """
    if dataset_mode(bind) != "challenge":
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    root = Path(
        bind.get_execution_options().get(
            "ai_challenge_directory", "/run/music-store/challenge"
        )
    )
    try:
        for filename, expected in CHALLENGE_CHECKSUMS.items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
                raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    except OSError:
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED) from None
    return root


def _json_value(value: object) -> str:
    """Serialize exact source scalars for deterministic provenance hashes.

    Args:
        value: Source date, timestamp, or exact decimal.

    Returns:
        Its canonical string representation.

    Raises:
        TypeError: If an unexpected source scalar is encountered.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError("Unsupported provenance scalar.")


def _schema(bind: Engine | Connection) -> str:
    """Require an explicit non-public schema without relying on its name as proof.

    Args:
        bind: Existing engine or active snapshot connection.

    Returns:
        Explicit target schema name.

    Raises:
        AgentFailure: If a default, public, or system schema was selected.
    """
    schema = bind.get_execution_options().get("schema_translate_map", {}).get(None)
    if (
        not schema
        or schema == "public"
        or schema.startswith("pg_")
        or schema == "information_schema"
    ):
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    return schema


def _fingerprint(connection: Connection) -> str:
    """Hash all tool-visible records, including policy bodies and identity data.

    Args:
        connection: A transaction targeting an explicit schema.

    Returns:
        A canonical SHA-256 fingerprint, independent of index timestamps/serial IDs.

    Raises:
        AgentFailure: If required tables or policy index metadata are absent.
    """
    names = set(inspect(connection).get_table_names(schema=_schema(connection)))
    policy_tables = {"policy_documents", "policy_sections"}
    if not policy_tables.issubset(names):
        raise AgentFailure(ErrorCode.MISSING_POLICY)
    if names != set(Base.metadata.tables) | policy_tables:
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
    records: dict[str, list[dict[str, object]]] = {}
    for name, table in sorted(Base.metadata.tables.items()):
        rows = connection.execute(
            select(table).order_by(*table.primary_key.columns)
        ).mappings()
        records[name] = [dict(row) for row in rows]
    if not records["products"]:
        raise AgentFailure(ErrorCode.EMPTY_CATALOG)
    for model, excluded, ordering in (
        (PolicyDocument, {"indexed_at"}, (PolicyDocument.document_id,)),
        (
            PolicySection,
            {"section_id", "search_vector"},
            (PolicySection.document_id, PolicySection.document_order),
        ),
    ):
        columns = [
            column for column in model.__table__.columns if column.name not in excluded
        ]
        rows = connection.execute(select(*columns).order_by(*ordering)).mappings()
        records[model.__tablename__] = [dict(row) for row in rows]
    encoded = json.dumps(
        records,
        sort_keys=True,
        ensure_ascii=True,
        default=_json_value,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_demo(engine: Engine) -> None:
    """Fail closed unless every selected record equals the pinned synthetic baseline.

    Args:
        engine: Explicitly selected demo engine; no fallback schema is attempted.

    Raises:
        AgentFailure: If provenance is unverified or database access fails.
    """
    with verified_snapshot(engine):
        pass


@contextmanager
def verified_snapshot(engine: Engine) -> Iterator[Engine]:
    """Keep provenance checks and real tool queries in the same read-only snapshot.

    Args:
        engine: Existing database engine with explicit schema translation.

    Yields:
        A supported SQLAlchemy session bind preserving the validated snapshot.

    Raises:
        AgentFailure: If data differs from the approved synthetic fixtures.
    """
    _schema(engine)
    mode = dataset_mode(engine)
    if mode == "challenge":
        challenge_sources(engine)
    expected = CHALLENGE_FINGERPRINT if mode == "challenge" else DEMO_FINGERPRINT
    try:
        with engine.connect().execution_options(
            isolation_level="REPEATABLE READ", postgresql_readonly=True
        ) as connection:
            with connection.begin():
                if _fingerprint(connection) != expected:
                    raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
                # Session accepts Connection as a bind. Existing tool annotations
                # name Engine; the cast preserves their implementation unchanged.
                yield cast(Engine, connection)
    except SQLAlchemyError:
        raise AgentFailure(ErrorCode.DATABASE) from None


def initialize_demo(engine: Engine) -> None:
    """Populate an empty selected schema exclusively from pinned synthetic assets.

    Existing populated schemas must already match the selected baseline; records
    are never overwritten. Challenge mode imports the complete original dataset
    and PDF. Sample mode retains its additional invented order association.

    Args:
        engine: Explicit mode and non-public schema; sources remain unchanged.

    Raises:
        AgentFailure: If sources differ, the schema is populated with other data,
            or initialization fails. Incomplete schemas are never authorized for use.
    """
    schema = _schema(engine)
    try:
        if dataset_mode(engine) == "challenge":
            source = challenge_sources(engine)
            with engine.connect() as connection:
                populated = bool(inspect(connection).get_table_names(schema=schema))
            if not populated:
                initialize_schema(engine)
                seed_data(engine, source)
                index_policy_pdf(engine, source / "políticas_da_loja.pdf")
            verify_demo(engine)
            return
        for filename, expected in SOURCE_CHECKSUMS.items():
            if (
                hashlib.sha256((SAMPLE_DIRECTORY / filename).read_bytes()).hexdigest()
                != expected
            ):
                raise AgentFailure(ErrorCode.DEMO_UNVERIFIED)
        with engine.connect() as connection:
            populated = bool(inspect(connection).get_table_names(schema=schema))
        if populated:
            verify_demo(engine)
            return
        _populate_demo(engine)
        verify_demo(engine)
    except AgentFailure:
        raise
    except (OSError, SQLAlchemyError, ValueError):
        raise AgentFailure(ErrorCode.DEMO_UNVERIFIED) from None


def _populate_demo(engine: Engine) -> None:
    """Build a new synthetic schema using existing ingestion and indexing operations.

    Args:
        engine: Empty non-public namespace selected by the initializer.
    """
    initialize_schema(engine)
    seed_data(engine, SAMPLE_DIRECTORY)
    with engine.begin() as connection:
        customer = dict(
            connection.execute(
                select(Customer.__table__).order_by(Customer.customer_id).limit(1)
            )
            .mappings()
            .one()
        )
        order = dict(
            connection.execute(
                select(Order.__table__).order_by(Order.order_id).limit(1)
            )
            .mappings()
            .one()
        )
        customer_id = connection.scalar(select(func.max(Customer.customer_id))) + 1
        order_id = connection.scalar(select(func.max(Order.order_id))) + 1
        customer.update(
            customer_id=customer_id, email=f"customer-{customer_id}@example.test"
        )
        order.update(order_id=order_id, customer_id=customer_id)
        connection.execute(insert(Customer), customer)
        connection.execute(insert(Order), order)
    index_policy_pdf(engine, SAMPLE_DIRECTORY / "store_policies.pdf")
