"""Catalog integration tests using synthetic records and isolated PostgreSQL schemas."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Connection, Engine, event, select
from sqlalchemy.engine import ExecutionContext
from sqlalchemy.orm import Session

from music_store_agent.data.ingestion import seed_data
from music_store_agent.database.models import Category, Product, Promotion
from music_store_agent.services.availability import AvailabilityReason
from music_store_agent.services.pricing import DiscountSource, PricingStatus
from music_store_agent.tools.products import (
    get_active_promotions,
    get_product_details,
    get_product_pricing,
    search_products,
)


def _product(
    product_id: int,
    name: str,
    price: str,
    *,
    category_id: int = 10,
    status: str = "active",
    stock: int = 3,
    description: str = "Invented instrument",
) -> Product:
    """Build a public synthetic product without relying on private catalog records.

    Args:
        product_id: Synthetic identifier.
        name: Synthetic display name.
        price: Exact list price text.
        category_id: Synthetic category identifier.
        status: Source-like inventory status.
        stock: Source-like stock quantity.
        description: Synthetic searchable description.

    Returns:
        An unpersisted ORM product.
    """
    return Product(
        product_id=product_id,
        name=name,
        price_brl=Decimal(price),
        category_id=category_id,
        status=status,
        stock_quantity=stock,
        description=description,
        specs={"invented": True, "features": ["sample"]},
        created_at=date(2025, 1, 1),
    )


@pytest.fixture
def catalog_engine(initialized_engine: Engine, source_dir: Path) -> Engine:
    """Extend imported samples with isolated records covering commercial edge cases.

    Args:
        initialized_engine: Existing fixture scoped to a disposable PostgreSQL schema.
        source_dir: Public synthetic CSVs from commit 02.

    Returns:
        A seeded test engine; the existing fixture owns schema cleanup.
    """
    seed_data(initialized_engine, source_dir)
    with Session(initialized_engine) as session, session.begin():
        session.add_all(
            [
                Category(
                    category_id=10,
                    name="Guitarras",
                    description="Invented electric instruments",
                ),
                Category(
                    category_id=11,
                    name="Sopros",
                    description="Invented wind instruments",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                _product(201, "Guitarra Elétrica Aurora", "100.00"),
                _product(202, "Guitarra sem estoque", "80.00", stock=0),
                _product(
                    203, "Guitarra antiga", "120.00", status="discontinued", stock=9
                ),
                _product(
                    204, "Guitarra futura", "90.00", status="coming_soon", stock=9
                ),
                _product(205, "Guitarra ambígua", "100.00"),
                _product(
                    206,
                    "Flauta 100%_literal / teste",
                    "0.30",
                    category_id=11,
                    description="A musician's literal example",
                ),
                _product(207, "Oferta zero", "20.00"),
                _product(208, "Oferta integral", "20.00"),
                _product(209, "Invented accessory", "10.00", category_id=2),
                _product(210, "Inactive-only instrument", "200.00"),
                _product(211, "Archived instrument", "60.00", status="archived"),
                _product(212, "Cent rounding instrument", "0.05"),
                _product(
                    213,
                    "Description match",
                    "75.00",
                    description="Distinct synthetic resonance",
                ),
            ]
        )
        session.flush()
        for promotion_id, product_id, percent, active in [
            (501, 201, "90", False),
            (502, 201, "30", False),
            (503, 205, "10", True),
            (504, 205, "20", True),
            (505, 207, "0", True),
            (506, 208, "100", True),
            (507, 209, "50", True),
            (508, 210, "90", False),
            (509, 210, "30", False),
            (510, 212, "10", True),
        ]:
            session.add(
                Promotion(
                    promotion_id=promotion_id,
                    product_id=product_id,
                    discount_percent=Decimal(percent),
                    description="Invented offer",
                    is_active=active,
                )
            )
    return initialized_engine


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("gUiTaRrA eLéTrIcA", [201]),
        ("ELÉTRICA AURORA", [201]),
        ("resonance", [213]),
        ("sopros", [206]),
        ("not in the catalog", []),
        ("' OR 1=1 --", []),
    ],
)
def test_literal_catalog_search(
    catalog_engine: Engine, query: str, expected: list[int]
) -> None:
    """Search names, descriptions, and Portuguese categories without raw SQL input.

    Args:
        catalog_engine: Isolated synthetic catalog.
        query: Literal case-insensitive search text.
        expected: Matching source product identifiers.
    """
    assert [
        row.product_id for row in search_products(catalog_engine, query=query)
    ] == expected


@pytest.mark.parametrize("query", ["%", "_", "/", "'", "%_"])
def test_search_wildcards_remain_literal(catalog_engine: Engine, query: str) -> None:
    """Escape SQL wildcards and the escape character without broadening results.

    Args:
        catalog_engine: Isolated synthetic catalog.
        query: Special character appearing in one synthetic product only.
    """
    assert [row.product_id for row in search_products(catalog_engine, query=query)] == [
        206
    ]


def test_category_ordering_and_default_visibility(catalog_engine: Engine) -> None:
    """Return deterministic available instrument records and apply category filters.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    first = search_products(catalog_engine)
    assert [row.product_id for row in first] == [
        101,
        201,
        205,
        206,
        207,
        208,
        210,
        212,
        213,
    ]
    assert first == search_products(catalog_engine)
    assert all(row.availability.available and row.in_store_scope for row in first)
    assert [row.product_id for row in search_products(catalog_engine, limit=2)] == [
        101,
        201,
    ]
    assert [
        row.product_id for row in search_products(catalog_engine, category="gUiTaRrAs")
    ] == [201, 205, 207, 208, 210, 212, 213]
    assert search_products(catalog_engine, category="Accessories") == ()
    assert search_products(catalog_engine, query="  ") == first


@pytest.mark.parametrize(
    ("product_id", "reason"),
    [
        (201, AvailabilityReason.AVAILABLE),
        (202, AvailabilityReason.OUT_OF_STOCK),
        (203, AvailabilityReason.DISCONTINUED),
        (204, AvailabilityReason.COMING_SOON),
        (211, AvailabilityReason.UNKNOWN_STATUS),
    ],
)
def test_explicit_details_explain_availability(
    catalog_engine: Engine, product_id: int, reason: AvailabilityReason
) -> None:
    """Retrieve unavailable records explicitly and honor status even with stock.

    Args:
        catalog_engine: Isolated synthetic catalog.
        product_id: Synthetic product identifier.
        reason: Expected structured availability reason.
    """
    result = get_product_details(catalog_engine, product_id)
    assert result is not None
    assert result.availability.reason == reason
    assert result.purchasable is (reason == AvailabilityReason.AVAILABLE)
    if product_id in (203, 204):
        assert result.stock_quantity > 0
    visible = {
        row.product_id for row in search_products(catalog_engine, available_only=False)
    }
    assert product_id in visible


def test_accessory_fixture_is_not_store_inventory(catalog_engine: Engine) -> None:
    """Keep source accessory records inspectable without offering them for sale.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    result = get_product_details(catalog_engine, 209)
    assert result is not None
    assert result.availability.available
    assert not result.in_store_scope
    assert not result.purchasable
    assert not result.verified_offer
    assert 209 not in {
        row.product_id for row in search_products(catalog_engine, available_only=False)
    }
    assert 209 not in {row.product_id for row in get_active_promotions(catalog_engine)}


def test_unknown_product_ids_return_none(catalog_engine: Engine) -> None:
    """Return explicit absence without fabricated records or recommendations.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    assert get_product_details(catalog_engine, 999999) is None
    assert get_product_pricing(catalog_engine, 999999) is None


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_result_limits_are_validated(catalog_engine: Engine, limit: int) -> None:
    """Reject invalid bounds for both customer search and promotion inspection.

    Args:
        catalog_engine: Isolated synthetic catalog.
        limit: Invalid requested bound.
    """
    with pytest.raises(ValueError, match="Limit"):
        search_products(catalog_engine, limit=limit)
    with pytest.raises(ValueError, match="Limit"):
        get_active_promotions(catalog_engine, limit=limit)


@pytest.mark.parametrize("price", ["-1", "NaN", "Infinity", "-Infinity", "bad", ""])
def test_invalid_budget_bounds_are_rejected(catalog_engine: Engine, price: str) -> None:
    """Reject malformed, negative, and nonfinite inclusive budget bounds.

    Args:
        catalog_engine: Isolated synthetic catalog.
        price: Invalid exact price input.
    """
    with pytest.raises(ValueError, match="Price bounds"):
        search_products(catalog_engine, max_price_brl=price)
    with pytest.raises(ValueError, match="Price bounds"):
        search_products(catalog_engine, min_price_brl=price)


def test_inverted_budget_bounds_are_rejected(catalog_engine: Engine) -> None:
    """Reject a minimum above the maximum instead of silently returning nothing.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    with pytest.raises(ValueError, match="Minimum price"):
        search_products(catalog_engine, min_price_brl="100", max_price_brl="99")


def test_budget_filters_use_the_selected_quote(catalog_engine: Engine) -> None:
    """Use promotional or explicitly selected PIX prices, with inclusive boundaries.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    query = "Elétrica Aurora"
    assert search_products(catalog_engine, query=query, max_price_brl="95") == ()
    assert (
        search_products(
            catalog_engine,
            query=query,
            max_price_brl="95",
            payment_method="credit_card",
        )
        == ()
    )
    pix = search_products(
        catalog_engine,
        query=query,
        min_price_brl="95",
        max_price_brl="95",
        payment_method="pix",
    )
    assert [row.product_id for row in pix] == [201]
    assert pix[0].quote.quoted_price == Decimal("95.00")
    assert pix[0].quote.advertised_price == Decimal("100.00")
    assert pix[0].quote.discount_source == DiscountSource.PIX
    promoted = search_products(
        catalog_engine, query="Sample Acoustic", max_price_brl="1104.93"
    )
    assert [row.product_id for row in promoted] == [101]
    assert promoted[0].quote.list_price == Decimal("1234.56")
    assert (
        search_products(
            catalog_engine, query="Sample Acoustic", max_price_brl="1104.92"
        )
        == ()
    )


def test_promotion_conflicts_are_visible_but_not_budget_offers(
    catalog_engine: Engine,
) -> None:
    """Expose conflicts without choosing a discount or admitting unverified budgets.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    result = get_product_pricing(catalog_engine, 205, payment_method="pix")
    assert result is not None
    assert result.quote.status == PricingStatus.CONFLICT
    assert result.quote.list_price == Decimal("100.00")
    assert result.quote.quoted_price is None
    assert not result.purchasable
    assert not result.verified_offer
    assert search_products(catalog_engine, query="ambígua", max_price_brl="1000") == ()
    assert (
        search_products(
            catalog_engine, query="ambígua", min_price_brl="0", payment_method="pix"
        )
        == ()
    )
    assert [
        row.product_id for row in search_products(catalog_engine, query="ambígua")
    ] == [205]


def test_active_promotion_inspection_distinguishes_offers(
    catalog_engine: Engine,
) -> None:
    """Distinguish active records from verified, purchasable price reductions.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    records = get_active_promotions(catalog_engine)
    assert [row.product_id for row in records] == [101, 205, 207, 208, 212]
    assert [
        row.product_id for row in get_active_promotions(catalog_engine, limit=2)
    ] == [101, 205]
    assert all(
        promotion.is_active
        for row in records
        for promotion in row.quote.active_promotions
    )
    by_id = {row.product_id: row for row in records}
    assert by_id[101].verified_offer
    assert not by_id[205].verified_offer
    assert not by_id[207].verified_offer
    assert by_id[207].quote.discount_percent == Decimal("0.00")
    assert by_id[208].quote.promotional_price == Decimal("0.00")
    assert by_id[208].verified_offer
    assert not by_id[212].verified_offer
    for product_id in (201, 210):
        price = get_product_pricing(catalog_engine, product_id)
        assert price is not None
        assert price.quote.active_promotions == ()
        assert price.quote.quoted_price == price.quote.list_price
    unavailable = get_product_pricing(catalog_engine, 103)
    assert unavailable is not None
    assert not unavailable.purchasable
    assert not unavailable.verified_offer


def test_pix_does_not_stack_in_database_results(catalog_engine: Engine) -> None:
    """Keep an active promotion's price and expose the selected payment context.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    result = get_product_pricing(catalog_engine, 101, payment_method="pix")
    assert result is not None
    assert result.quote.quoted_price == Decimal("1104.93")
    assert result.quote.promotional_price == Decimal("1104.93")
    assert result.quote.pix_price is None
    assert result.quote.discount_source == DiscountSource.PROMOTION
    assert result.quote.payment_method == "pix"


def test_results_serialize_exact_money_and_structured_specs(
    catalog_engine: Engine,
) -> None:
    """Produce JSON-safe results after session cleanup without leaking operational data.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    result = get_product_details(catalog_engine, 101)
    assert result is not None
    serialized = json.loads(result.model_dump_json())
    assert serialized["quote"]["list_price"] == "1234.56"
    assert serialized["quote"]["quoted_price"] == "1104.93"
    assert serialized["quote"]["discount_percent"] == "10.50"
    assert serialized["quote"]["discount_source"] == "promotion"
    assert serialized["availability"] == {"available": True, "reason": "available"}
    assert serialized["specs"]["materials"] == {"body": "sample wood"}
    assert serialized["specs"]["features"] == ["acoustic", "natural"]
    assert serialized["created_at"] == "2025-01-10"
    assert set(serialized) == {
        "product_id",
        "name",
        "description",
        "category",
        "stock_quantity",
        "status",
        "specs",
        "created_at",
        "availability",
        "in_store_scope",
        "purchasable",
        "verified_offer",
        "quote",
    }
    pricing = get_product_pricing(catalog_engine, 206, payment_method="pix")
    assert pricing is not None
    assert pricing.model_dump(mode="json")["quote"]["pix_price"] == "0.29"


def test_budget_limit_does_not_truncate_later_qualifying_rows(
    catalog_engine: Engine,
) -> None:
    """Read another bounded keyset batch when early candidates exceed the budget.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    with Session(catalog_engine) as session, session.begin():
        session.add_all(
            [
                _product(
                    1000 + offset,
                    "Batch sentinel",
                    "10000.00" if offset < 100 else "1.00",
                )
                for offset in range(101)
            ]
        )
    result = search_products(
        catalog_engine, query="Batch sentinel", max_price_brl="1.00", limit=1
    )
    assert [row.product_id for row in result] == [1100]


def test_catalog_reads_are_batched_and_do_not_write(catalog_engine: Engine) -> None:
    """Load multiple results without N+1 queries or writes to source records.

    Args:
        catalog_engine: Isolated synthetic catalog.
    """
    statements: list[str] = []

    def record_statement(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        """Observe executed SQL without replacing actual PostgreSQL behavior.

        Args:
            connection: Actual SQLAlchemy connection.
            cursor: Actual driver cursor.
            statement: Parameterized SQL text without source values.
            parameters: Bound query values, deliberately not recorded.
            context: SQLAlchemy execution metadata.
            executemany: Whether the execution contains multiple parameter sets.
        """
        statements.append(statement)

    event.listen(catalog_engine, "before_cursor_execute", record_statement)
    try:
        results = search_products(catalog_engine, payment_method="pix")
        assert len(results) > 3
        assert len(statements) == 2
        assert all(statement.lstrip().startswith("SELECT") for statement in statements)
    finally:
        event.remove(catalog_engine, "before_cursor_execute", record_statement)
    with Session(catalog_engine) as session:
        assert session.scalar(
            select(Product.price_brl).where(Product.product_id == 201)
        ) == Decimal("100.00")
