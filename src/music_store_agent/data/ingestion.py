"""Import six CSV datasets atomically using PostgreSQL primary-key upserts."""

import csv
import json
import re
from collections.abc import Iterator
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Column, Date, Engine, Integer, Numeric, Table
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from music_store_agent.database.models import Base
from music_store_agent.database.session import create_session_factory

DATASET_NAMES: tuple[str, ...] = (
    "categories",
    "products",
    "customers",
    "orders",
    "order_items",
    "promotions",
)
SOURCE_PREFIX: str = "desafio_tecnico_ai_eng - "


class IngestionError(ValueError):
    """Report an import failure with dataset/row context and no source values."""


def _discover_sources(source_dir: Path) -> dict[str, Path]:
    """Find one canonical or prefixed file per dataset, including nested folders.

    Args:
        source_dir: Root containing the six source CSV files.

    Returns:
        Dataset names mapped to unambiguous source paths.

    Raises:
        IngestionError: If the root is invalid or a dataset is missing or ambiguous.
    """
    if not source_dir.is_dir():
        raise IngestionError("Source directory does not exist or is not a directory.")
    paths = list(source_dir.rglob("*.csv"))
    sources: dict[str, Path] = {}
    for name in DATASET_NAMES:
        filenames = {f"{name}.csv", f"{SOURCE_PREFIX}{name}.csv"}
        matches = [path for path in paths if path.is_file() and path.name in filenames]
        if not matches:
            raise IngestionError(f"{name}.csv: required dataset is missing.")
        if len(matches) > 1:
            raise IngestionError(f"{name}.csv: ambiguous duplicate source files.")
        sources[name] = matches[0]
    return sources


def _reject_json_constant(value: str) -> None:
    """Reject nonstandard JSON numeric constants without exposing source values.

    Args:
        value: Nonstandard constant supplied by the JSON decoder.

    Raises:
        ValueError: Always, because PostgreSQL JSONB requires valid JSON numbers.
    """
    raise ValueError("Nonstandard JSON constants are not accepted.")


def _parse_value(raw: str, column: Column[Any]) -> object:
    """Parse one field using its mapped SQL type without silently rounding money.

    Args:
        raw: Source field; nonempty text is preserved verbatim.
        column: Mapped column defining nullability and the target SQL type.

    Returns:
        A typed value suitable for a PostgreSQL insert.

    Raises:
        ValueError: If a required field is empty or the value cannot be represented.
        InvalidOperation: If a decimal cannot be parsed or quantized exactly.
    """
    value = raw.strip()
    if not value:
        if column.nullable:
            return None
        raise ValueError("Required value is empty.")
    if isinstance(column.type, Integer):
        if not re.fullmatch(r"[+-]?[0-9]+", value):
            raise ValueError("Expected an integer.")
        return int(value)
    if isinstance(column.type, Numeric):
        # Plain decimal notation avoids locale-dependent thousands separators.
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value):
            raise ValueError("Expected a finite decimal.")
        number = Decimal(value)
        scale = column.type.scale or 0
        if number != number.quantize(Decimal(1).scaleb(-scale)):
            raise ValueError("Decimal has more precision than the target column.")
        precision = column.type.precision
        if precision is not None and abs(number) >= Decimal(10) ** (precision - scale):
            raise ValueError("Decimal exceeds the target column range.")
        return number
    if isinstance(column.type, Boolean):
        flags = {"true": True, "false": False, "1": True, "0": False}
        if value.casefold() not in flags:
            raise ValueError("Expected true, false, 1, or 0.")
        return flags[value.casefold()]
    if isinstance(column.type, Date):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("Expected an ISO date in YYYY-MM-DD format.")
        return parsed
    if isinstance(column.type, JSONB):
        return json.loads(value, parse_constant=_reject_json_constant)
    return raw


def _read_rows(table: Table, path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    """Read structurally valid records with line numbers and sanitized read errors.

    Args:
        table: Table defining the required source columns.
        path: Selected CSV file.

    Yields:
        Physical line numbers and source field mappings.

    Raises:
        IngestionError: If headers, row structure, encoding, or file access is invalid.
    """
    reader = None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames or []
            missing = sorted(set(table.columns.keys()).difference(headers))
            if missing:
                raise IngestionError(
                    f"{table.name}.csv: missing required columns: {', '.join(missing)}."
                )
            if len(headers) != len(set(headers)):
                raise IngestionError(f"{table.name}.csv: duplicate column names.")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise IngestionError(
                        f"{table.name}.csv: row {reader.line_num}: "
                        "field count does not match header."
                    )
                yield reader.line_num, row
    except (UnicodeError, csv.Error):
        line = max(reader.reader.line_num, 1) if reader is not None else 1
        raise IngestionError(
            f"{table.name}.csv: row {line}: source is not valid UTF-8 CSV."
        ) from None
    except OSError:
        raise IngestionError(
            f"{table.name}.csv: source file could not be read."
        ) from None


def _write_dataset(session: Session, table: Table, path: Path) -> int:
    """Validate and upsert each row within the caller's single import transaction.

    Args:
        session: Active transaction spanning all six datasets.
        table: Table defining the source columns and primary key.
        path: Selected CSV file.

    Returns:
        Number of source records imported into this table.

    Raises:
        IngestionError: If source validation or a database write fails.
    """
    count = 0
    keys: set[tuple[object, ...]] = set()
    for line, row in _read_rows(table, path):
        context = f"{table.name}.csv: row {line}"
        values: dict[str, object] = {}
        for column in table.columns:
            try:
                values[column.name] = _parse_value(row[column.name], column)
            except (ValueError, InvalidOperation, OverflowError, RecursionError):
                # Decoder and driver exceptions may contain customer data.
                raise IngestionError(
                    f"{context}: invalid value for column {column.name}."
                ) from None
        key = tuple(values[column.name] for column in table.primary_key.columns)
        if key in keys:
            raise IngestionError(f"{context}: duplicate primary key in source.")
        keys.add(key)
        statement = insert(table).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=list(table.primary_key.columns),
            set_={
                column.name: statement.excluded[column.name]
                for column in table.columns
                if not column.primary_key
            },
        )
        try:
            session.execute(statement)
        except IntegrityError as error:
            code = getattr(error.orig, "sqlstate", None)
            reason = (
                "foreign-key reference does not exist"
                if code == "23503"
                else "database integrity constraint rejected the record"
            )
            raise IngestionError(f"{context}: {reason}.") from None
        except SQLAlchemyError:
            raise IngestionError(f"{context}: database write failed.") from None
        count += 1
    return count


def seed_data(engine: Engine, source_dir: Path) -> dict[str, int]:
    """Atomically upsert all six datasets while preserving unrelated records.

    Args:
        engine: Engine targeting the initialized PostgreSQL schema.
        source_dir: Directory containing canonical or original prefixed CSV files.

    Returns:
        Imported source row counts by dataset, emitted only after commit succeeds.

    Raises:
        IngestionError: If discovery, validation, a database write, or commit fails.
            All writes from the current import are rolled back.
    """
    try:
        sources = _discover_sources(source_dir)
        counts: dict[str, int] = {}
        # A late parse, FK, or commit failure rolls back earlier inserts and updates.
        with create_session_factory(engine).begin() as session:
            for name in DATASET_NAMES:
                counts[name] = _write_dataset(
                    session, Base.metadata.tables[name], sources[name]
                )
        return counts
    except IngestionError:
        raise
    except (OSError, UnicodeError, csv.Error):
        raise IngestionError(
            "Source files could not be read as valid UTF-8 CSV."
        ) from None
    except SQLAlchemyError:
        raise IngestionError("Database operation failed; import rolled back.") from None
