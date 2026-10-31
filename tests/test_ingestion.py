"""Exercise CSV ingestion and integrity against isolated PostgreSQL schemas."""

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from music_store_agent.data.ingestion import DATASET_NAMES, IngestionError, seed_data
from music_store_agent.database.models import (
    Base,
    Category,
    Customer,
    Order,
    OrderItem,
    Product,
    Promotion,
)
from music_store_agent.database.session import initialize_schema


def _rewrite_cell(source_dir: Path, dataset: str, column: str, value: str) -> None:
    """Replace the first record's selected field without changing other CSV fields.

    Args:
        source_dir: Mutable fixture directory.
        dataset: Dataset basename without extension.
        column: Header identifying the field to replace.
        value: Replacement source text.
    """
    path = source_dir / f"{dataset}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        rows = list(reader)
    assert headers is not None
    rows[0][column] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _row_counts(engine: Engine) -> dict[str, int]:
    """Read exact table sizes from the engine's schema, including after rollbacks.

    Args:
        engine: Isolated PostgreSQL engine.

    Returns:
        Row counts for all operational tables.
    """
    with Session(engine) as session:
        return {
            name: session.scalar(
                select(func.count()).select_from(Base.metadata.tables[name])
            )
            for name in DATASET_NAMES
        }


def test_schema_creation_is_repeatable(
    database_engine: Engine, source_dir: Path
) -> None:
    """Create six tables repeatedly and preserve previously imported records.

    Args:
        database_engine: Engine with no preexisting test schema or tables.
        source_dir: Synthetic CSV fixtures.
    """
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    assert not inspect(database_engine).has_schema(schema)
    initialize_schema(database_engine)
    assert set(inspect(database_engine).get_table_names(schema=schema)) == set(
        DATASET_NAMES
    )
    counts = seed_data(database_engine, source_dir)
    initialize_schema(database_engine)
    assert _row_counts(database_engine) == counts


def test_import_preserves_types_and_relationships(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Import every table and preserve decimals, JSON, dates, flags, and source totals.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Synthetic CSV fixtures.
    """
    assert seed_data(initialized_engine, source_dir) == {
        "categories": 2,
        "products": 3,
        "customers": 2,
        "orders": 2,
        "order_items": 3,
        "promotions": 4,
    }
    with Session(initialized_engine) as session:
        product = session.get(Product, 101)
        assert product is not None
        assert product.price_brl == Decimal("1234.56")
        assert isinstance(product.price_brl, Decimal)
        assert product.created_at == date(2025, 1, 10)
        assert product.specs == {
            "strings": 6,
            "materials": {"body": "sample wood"},
            "features": ["acoustic", "natural"],
        }
        assert product.category.name == "Strings"
        assert {row.product_id for row in product.category.products} == {101, 103}
        assert {row.promotion_id for row in product.promotions} == {401, 404}
        assert product.order_items[0].order_id == 301

        order = session.get(Order, 301)
        assert order is not None
        assert order.customer.customer_id == 201
        assert order.customer.orders[0].order_id == 301
        assert {item.product.product_id for item in order.items} == {101, 102}
        assert (order.tracking_code, order.estimated_delivery, order.notes) == (
            None,
            None,
            None,
        )
        # A different total detects accidental recomputation from product prices.
        assert order.total_brl == Decimal("1234.67")
        shipped = session.get(Order, 302)
        assert shipped is not None
        assert shipped.tracking_code == "SAMPLE-TRACK-302"
        assert shipped.estimated_delivery == date(2025, 4, 9)
        assert shipped.notes == "Invented delivery note"

        assert session.scalars(
            select(Product.status).order_by(Product.product_id)
        ).all() == ["active", "discontinued", "coming_soon"]
        assert session.scalars(
            select(Promotion.is_active).order_by(Promotion.promotion_id)
        ).all() == [True, False, True, False]
        assert session.scalar(
            select(Product.price_brl).where(Product.product_id == 102)
        ) == Decimal("0.10")
        assert session.scalar(
            select(Promotion.discount_percent).where(Promotion.promotion_id == 401)
        ) == Decimal("10.50")
        assert session.scalars(
            select(Customer.email).order_by(Customer.customer_id)
        ).all() == ["shared@example.test", "shared@example.test"]


def test_reimport_is_idempotent_and_updates(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Repeat identical inputs and update corrected source values without duplicates.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
    """
    expected = seed_data(initialized_engine, source_dir)
    assert seed_data(initialized_engine, source_dir) == expected
    assert _row_counts(initialized_engine) == expected
    _rewrite_cell(source_dir, "products", "price_brl", "12.34")
    _rewrite_cell(source_dir, "order_items", "quantity", "3")
    _rewrite_cell(source_dir, "promotions", "is_active", "false")
    seed_data(initialized_engine, source_dir)
    assert _row_counts(initialized_engine) == expected
    with Session(initialized_engine) as session:
        assert session.scalar(
            select(Product.price_brl).where(Product.product_id == 101)
        ) == Decimal("12.34")
        assert (
            session.scalar(
                select(OrderItem.quantity).where(
                    OrderItem.order_id == 301, OrderItem.product_id == 101
                )
            )
            == 3
        )
        assert (
            session.scalar(
                select(Promotion.is_active).where(Promotion.promotion_id == 401)
            )
            is False
        )


def test_unrelated_records_survive_import(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Preserve a category absent from the imported source files.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Synthetic CSV fixtures.
    """
    with Session(initialized_engine) as session, session.begin():
        session.add(
            Category(category_id=999, name="Unrelated", description="Keep this row")
        )
    seed_data(initialized_engine, source_dir)
    seed_data(initialized_engine, source_dir)
    with Session(initialized_engine) as session:
        unrelated = session.get(Category, 999)
        assert unrelated is not None
        assert unrelated.description == "Keep this row"


@pytest.mark.parametrize("filename_style", ["canonical", "prefixed", "nested_bom"])
def test_discovery_and_encoding(
    initialized_engine: Engine, source_dir: Path, filename_style: str
) -> None:
    """Support canonical names, original names, nested extraction folders, and BOMs.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
        filename_style: Filename and encoding variant under test.
    """
    if filename_style != "canonical":
        target = source_dir
        if filename_style == "nested_bom":
            target = source_dir / "extracted" / "datasets"
            target.mkdir(parents=True)
        for path in source_dir.glob("*.csv"):
            contents = path.read_text(encoding="utf-8")
            path.unlink()
            (target / f"desafio_tecnico_ai_eng - {path.name}").write_text(
                contents, encoding="utf-8-sig"
            )
    assert seed_data(initialized_engine, source_dir)["products"] == 3


@pytest.mark.parametrize(
    "defect",
    [
        "missing_file",
        "duplicate_file",
        "missing_column",
        "duplicate_column",
        "short_row",
        "extra_field",
        "duplicate_key",
        "invalid_utf8",
        "malformed_quotes",
    ],
)
def test_invalid_sources_fail_without_writes(
    initialized_engine: Engine, source_dir: Path, defect: str
) -> None:
    """Reject incomplete, ambiguous, or structurally malformed source datasets.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
        defect: Structural source failure to introduce.
    """
    path = source_dir / "promotions.csv"
    contents = path.read_text(encoding="utf-8")
    expected = "promotions.csv"
    match defect:
        case "missing_file":
            path.unlink()
            expected = "required dataset is missing"
        case "duplicate_file":
            (source_dir / "desafio_tecnico_ai_eng - promotions.csv").write_text(
                contents
            )
            expected = "ambiguous duplicate"
        case "missing_column":
            path.write_text(contents.replace("promotion_id", "unknown", 1))
            expected = "missing required columns: promotion_id"
        case "duplicate_column":
            path.write_text(contents.replace("is_active", "is_active,description", 1))
            expected = "duplicate column names"
        case "short_row":
            path.write_text(contents + "405,101\n")
            expected = "field count"
        case "extra_field":
            path.write_text(contents + "405,101,10.00,Invented,true,extra\n")
            expected = "field count"
        case "duplicate_key":
            path.write_text(contents + contents.splitlines()[1] + "\n")
            expected = "duplicate primary key"
        case "invalid_utf8":
            path.write_bytes(contents.encode() + b"\xff")
            expected = "valid UTF-8 CSV"
        case "malformed_quotes":
            path.write_text(contents + '405,101,10.00,"unfinished,true\n')
            expected = "promotions.csv: row .*valid UTF-8 CSV"
    with pytest.raises(IngestionError, match=expected):
        seed_data(initialized_engine, source_dir)
    assert all(count == 0 for count in _row_counts(initialized_engine).values())


@pytest.mark.parametrize(
    ("dataset", "column", "bad_value"),
    [
        ("products", "price_brl", "not-a-number"),
        ("products", "price_brl", "1.005"),
        ("products", "price_brl", "NaN"),
        ("products", "price_brl", "1000000000000.00"),
        ("products", "created_at", "2025-02-30"),
        ("products", "specs", "{invalid-json}"),
        ("products", "specs", '{"invalid": NaN}'),
        ("products", "stock_quantity", "1.5"),
        ("orders", "estimated_delivery", "not-a-date"),
        ("promotions", "is_active", "maybe"),
        ("promotions", "discount_percent", "Infinity"),
        ("customers", "name", ""),
    ],
)
def test_malformed_values_roll_back(
    initialized_engine: Engine,
    source_dir: Path,
    dataset: str,
    column: str,
    bad_value: str,
) -> None:
    """Reject invalid typed fields without leaving earlier datasets partially imported.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
        dataset: Dataset to corrupt.
        column: Typed field to corrupt.
        bad_value: Invalid source text.
    """
    _rewrite_cell(source_dir, dataset, column, bad_value)
    with pytest.raises(
        IngestionError,
        match=rf"{dataset}.csv: row 2: invalid value for column {column}",
    ):
        seed_data(initialized_engine, source_dir)
    assert all(count == 0 for count in _row_counts(initialized_engine).values())


@pytest.mark.parametrize(
    ("dataset", "column"),
    [
        ("products", "category_id"),
        ("orders", "customer_id"),
        ("order_items", "order_id"),
        ("order_items", "product_id"),
        ("promotions", "product_id"),
    ],
)
def test_invalid_foreign_keys_roll_back(
    initialized_engine: Engine, source_dir: Path, dataset: str, column: str
) -> None:
    """Enforce every source foreign key using actual PostgreSQL constraints.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
        dataset: Child dataset to corrupt.
        column: Foreign-key field to replace with an absent identifier.
    """
    _rewrite_cell(source_dir, dataset, column, "999999")
    with pytest.raises(IngestionError, match="foreign-key reference does not exist"):
        seed_data(initialized_engine, source_dir)
    assert all(count == 0 for count in _row_counts(initialized_engine).values())


@pytest.mark.parametrize(
    ("dataset", "column", "bad_value"),
    [
        ("products", "price_brl", "-0.01"),
        ("products", "stock_quantity", "-1"),
        ("orders", "total_brl", "-0.01"),
        ("order_items", "quantity", "0"),
        ("promotions", "discount_percent", "-0.01"),
        ("promotions", "discount_percent", "100.01"),
    ],
)
def test_integrity_constraints_roll_back(
    initialized_engine: Engine,
    source_dir: Path,
    dataset: str,
    column: str,
    bad_value: str,
) -> None:
    """Enforce nonnegative amounts, stock, positive quantities, and discount bounds.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
        dataset: Table whose check constraint is exercised.
        column: Constrained field.
        bad_value: Numerically valid but out-of-range value.
    """
    _rewrite_cell(source_dir, dataset, column, bad_value)
    with pytest.raises(IngestionError, match="database integrity constraint"):
        seed_data(initialized_engine, source_dir)
    assert all(count == 0 for count in _row_counts(initialized_engine).values())


def test_late_failure_rolls_back_updates_and_inserts(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Restore existing values and remove new rows after failure in the final dataset.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
    """
    expected = seed_data(initialized_engine, source_dir)
    _rewrite_cell(source_dir, "products", "price_brl", "77.77")
    with (source_dir / "categories.csv").open("a") as handle:
        handle.write("999,New category,Must roll back\n")
    _rewrite_cell(source_dir, "promotions", "product_id", "999999")
    with pytest.raises(IngestionError, match="foreign-key"):
        seed_data(initialized_engine, source_dir)
    assert _row_counts(initialized_engine) == expected
    with Session(initialized_engine) as session:
        assert session.get(Category, 999) is None
        assert session.scalar(
            select(Product.price_brl).where(Product.product_id == 101)
        ) == Decimal("1234.56")


def test_composite_primary_key_is_enforced(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Allow multiple products per order but reject an existing order/product pair.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Synthetic CSV fixtures.
    """
    seed_data(initialized_engine, source_dir)
    with Session(initialized_engine) as session:
        assert session.get(OrderItem, (301, 101)) is not None
        assert session.get(OrderItem, (301, 102)) is not None
    with (
        pytest.raises(IntegrityError),
        Session(initialized_engine) as session,
        session.begin(),
    ):
        session.add(OrderItem(order_id=301, product_id=101, quantity=2))
    assert _row_counts(initialized_engine)["order_items"] == 3


def test_errors_do_not_expose_source_values(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Keep malformed private field contents out of public exception messages.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        source_dir: Mutable synthetic CSV fixtures.
    """
    marker = "private-marker@example.test"
    _rewrite_cell(source_dir, "orders", "estimated_delivery", marker)
    with pytest.raises(IngestionError) as raised:
        seed_data(initialized_engine, source_dir)
    assert "orders.csv: row 2" in str(raised.value)
    assert "estimated_delivery" in str(raised.value)
    assert marker not in str(raised.value)


def test_missing_source_directory_fails(
    initialized_engine: Engine, tmp_path: Path
) -> None:
    """Fail clearly when the requested input location is absent.

    Args:
        initialized_engine: Empty isolated PostgreSQL schema.
        tmp_path: Temporary directory with no input datasets.
    """
    with pytest.raises(IngestionError, match="Source directory"):
        seed_data(initialized_engine, tmp_path / "absent")
