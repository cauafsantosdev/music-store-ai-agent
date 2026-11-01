"""Order tracking tests using only invented records in isolated PostgreSQL schemas."""

import json
import logging
from datetime import date
from decimal import Decimal
from pathlib import Path
from traceback import format_exception
from typing import cast

import pytest
from pytest import CaptureFixture, LogCaptureFixture
from sqlalchemy import Connection, Engine, MetaData, Table, event, select
from sqlalchemy.engine import ExecutionContext
from sqlalchemy.orm import Session
from sqlalchemy.schema import DropConstraint

from music_store_agent.data.ingestion import seed_data
from music_store_agent.database.models import Base, Customer, Order
from music_store_agent.services.order_verification import (
    EMAIL_WHITESPACE,
    OrderLookupError,
)
from music_store_agent.tools.orders import get_order_status
from music_store_agent.tools.results import OrderTrackingResult


def _order(
    order_id: int,
    customer_id: int,
    *,
    status: str = "processing",
    tracking_code: str | None = None,
    estimated_delivery: date | None = None,
) -> Order:
    """Build a synthetic order containing private fields that must remain internal.

    Args:
        order_id: Synthetic source identifier.
        customer_id: Synthetic associated identity.
        status: Arbitrary source status.
        tracking_code: Optional invented tracking identifier.
        estimated_delivery: Optional stored estimate, never an actual delivery event.

    Returns:
        An unpersisted order with known source facts and synthetic internal notes.
    """
    return Order(
        order_id=order_id,
        customer_id=customer_id,
        order_date=date(2020, 1, 2),
        status=status,
        total_brl=Decimal("123.45"),
        payment_method="pix",
        tracking_code=tracking_code,
        estimated_delivery=estimated_delivery,
        notes="Internal synthetic note that must never be returned",
    )


@pytest.fixture
def tracking_engine(initialized_engine: Engine, source_dir: Path) -> Engine:
    """Seed public fixtures and unique synthetic associations in a disposable schema.

    Args:
        initialized_engine: Existing per-test PostgreSQL schema fixture.
        source_dir: Invented CSV records, including the existing duplicate email.

    Returns:
        A seeded engine whose cleanup remains owned by the existing fixture.
    """
    seed_data(initialized_engine, source_dir)
    with Session(initialized_engine) as session, session.begin():
        for customer_id, email in [
            (501, "unique@example.test"),
            (502, "other@example.test"),
            (503, "first.last+music@example.test"),
            (504, "literal%_tag@example.test"),
            (505, "música@example.test"),
        ]:
            session.add(
                Customer(
                    customer_id=customer_id,
                    name="Invented Tracking Customer",
                    phone="+55-00-0000-0000",
                    email=email,
                    city="Invented Tracking City",
                )
            )
        session.flush()
        session.add_all(
            [
                _order(
                    601,
                    501,
                    status="shipped",
                    tracking_code="SAMPLE-TRACK-601",
                    estimated_delivery=date(2020, 1, 15),
                ),
                _order(602, 502),
                _order(603, 501, status="awaiting_manual_carrier_review"),
                _order(604, 503),
                _order(605, 504),
                _order(606, 501),
                _order(607, 505),
            ]
        )
    return initialized_engine


@pytest.mark.parametrize(
    ("order_id", "email"),
    [
        (601, "unique@example.test"),
        (601, "UNIQUE@EXAMPLE.TEST"),
        (601, " \tunique@example.test\n "),
        (601, "\u2003unique@example.test\u00a0"),
        (604, "FIRST.LAST+MUSIC@EXAMPLE.TEST"),
        (605, "literal%_tag@example.test"),
        (607, "MÚSICA@EXAMPLE.TEST"),
    ],
)
def test_unique_order_association_succeeds(
    tracking_engine: Engine, order_id: int, email: str
) -> None:
    """Match unique identities exactly while ignoring case and outer whitespace.

    Args:
        tracking_engine: Isolated synthetic tracking records.
        order_id: Order associated with the supplied synthetic email.
        email: Valid input variant, including literal SQL wildcard characters.
    """
    result = get_order_status(tracking_engine, order_id=order_id, customer_email=email)
    assert isinstance(result, OrderTrackingResult)
    assert result.order_id == order_id


def test_tracking_facts_remain_source_snapshots(tracking_engine: Engine) -> None:
    """Preserve dates, tracking, and shipped status despite a past delivery estimate.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    result = get_order_status(
        tracking_engine, order_id=601, customer_email="unique@example.test"
    )
    assert result is not None
    assert result.status == "shipped"
    assert result.order_date == date(2020, 1, 2)
    assert result.estimated_delivery == date(2020, 1, 15)
    assert result.tracking_code == "SAMPLE-TRACK-601"
    assert result.model_dump(mode="json") == {
        "order_id": 601,
        "status": "shipped",
        "order_date": "2020-01-02",
        "estimated_delivery": "2020-01-15",
        "tracking_code": "SAMPLE-TRACK-601",
    }


def test_missing_tracking_and_delivery_remain_null(tracking_engine: Engine) -> None:
    """Keep missing fields as None in Python and null in serialized JSON.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    result = get_order_status(
        tracking_engine, order_id=602, customer_email="other@example.test"
    )
    assert result is not None
    assert result.status == "processing"
    assert result.tracking_code is None
    assert result.estimated_delivery is None
    serialized = json.loads(result.model_dump_json())
    assert serialized["tracking_code"] is None
    assert serialized["estimated_delivery"] is None


def test_unknown_status_is_preserved(tracking_engine: Engine) -> None:
    """Return arbitrary source statuses without imposing a closed status vocabulary.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    result = get_order_status(
        tracking_engine, order_id=603, customer_email="unique@example.test"
    )
    assert result is not None
    assert result.status == "awaiting_manual_carrier_review"
    assert result.model_dump(mode="json")["status"] == "awaiting_manual_carrier_review"


@pytest.mark.parametrize(
    ("order_id", "email"),
    [
        (999999, "unique@example.test"),
        (2_147_483_647, "unique@example.test"),
        (601, "wrong@example.test"),
        (601, "other@example.test"),
        (602, "unique@example.test"),
        (301, "shared@example.test"),
        (302, "SHARED@EXAMPLE.TEST"),
        (601, "uni@example.test"),
        (601, "unique@example.test.extra"),
        (601, "uni%@example.test"),
        (601, "uniq_e@example.test"),
        (601, "%@example.test"),
        (601, "unique'OR'1'='1@example.test"),
        (604, "firstlast+music@example.test"),
        (604, "first.last@example.test"),
    ],
)
def test_association_failures_are_indistinguishable(
    tracking_engine: Engine, order_id: int, email: str
) -> None:
    """Return only None for unknown, mismatched, ambiguous, or literal probe lookups.

    Args:
        tracking_engine: Isolated synthetic tracking records.
        order_id: Requested source identifier, including absent orders.
        email: Structurally valid unmatched, duplicated, or hostile-looking address.
    """
    assert (
        get_order_status(tracking_engine, order_id=order_id, customer_email=email)
        is None
    )


@pytest.mark.parametrize(
    "duplicate_email",
    [
        "unique@example.test",
        "UNIQUE@EXAMPLE.TEST",
        " unique@example.test ",
        "\tunique@example.test\n",
        "\u2003unique@example.test\u00a0",
    ],
)
def test_normalized_duplicates_reject_every_order(
    tracking_engine: Engine, duplicate_email: str
) -> None:
    """Reject identity ambiguity even when another matching customer has no orders.

    Args:
        tracking_engine: Isolated synthetic tracking records.
        duplicate_email: Another identity with the same normalized address.
    """
    with Session(tracking_engine) as session, session.begin():
        session.add(
            Customer(
                customer_id=999,
                name="Invented Separate Identity",
                phone="000",
                email=duplicate_email,
                city="Invented City",
            )
        )
    for order_id in (601, 603, 606):
        assert (
            get_order_status(
                tracking_engine, order_id=order_id, customer_email="unique@example.test"
            )
            is None
        )


def test_missing_customer_association_fails_closed(tracking_engine: Engine) -> None:
    """Reject an orphan order if integrity has been broken outside normal ingestion.

    Args:
        tracking_engine: Isolated synthetic tracking records and disposable DDL scope.
    """
    schema = tracking_engine.get_execution_options()["schema_translate_map"][None]
    with tracking_engine.begin() as connection:
        table = Table("orders", MetaData(), schema=schema, autoload_with=connection)
        # Only the generated test schema is altered to simulate corrupt source data.
        for constraint in table.foreign_key_constraints:
            connection.execute(DropConstraint(constraint))
    with Session(tracking_engine) as session, session.begin():
        session.add(_order(699, 999999))
    assert (
        get_order_status(
            tracking_engine, order_id=699, customer_email="unique@example.test"
        )
        is None
    )


@pytest.mark.parametrize("padding", [" \t\n", "\u2003\u00a0", EMAIL_WHITESPACE])
def test_stored_email_padding_matches_input_normalization(
    tracking_engine: Engine, padding: str
) -> None:
    """Match unique stored identities using the same whitespace rules as submissions.

    Args:
        tracking_engine: Isolated synthetic tracking records.
        padding: ASCII or Unicode whitespace around the stored and submitted email.
    """
    with Session(tracking_engine) as session, session.begin():
        customer = session.get(Customer, 501)
        assert customer is not None
        customer.email = f"{padding}UNIQUE@EXAMPLE.TEST{padding}"
    result = get_order_status(
        tracking_engine,
        order_id=601,
        customer_email=f"{padding}unique@example.test{padding}",
    )
    assert result is not None
    assert result.order_id == 601


@pytest.mark.parametrize(
    "order_id", [0, -1, True, False, 1.5, "601", None, 2_147_483_648, 10**100]
)
def test_invalid_order_ids_fail_before_query(
    database_engine: Engine, order_id: object
) -> None:
    """Reject invalid types and ranges without connecting to uninitialized tables.

    Args:
        database_engine: Engine whose application tables have not been initialized.
        order_id: Untrusted input deliberately violating the typed API contract.
    """
    with pytest.raises(ValueError, match="Order ID"):
        get_order_status(
            database_engine,
            order_id=cast(int, order_id),
            customer_email="unique@example.test",
        )


@pytest.mark.parametrize(
    "email",
    [
        "",
        " \t\n",
        "missing-at.example.test",
        "@example.test",
        "unique@",
        "unique@example",
        "unique@example..test",
        "two@@example.test",
        "unique name@example.test",
        "unique@example.test\x00",
        "unique@example.test\nother@example.test",
        "unique@example.test' OR '1'='1",
        "a" * 255 + "@example.test",
        None,
        123,
        False,
    ],
)
def test_invalid_emails_fail_before_query(
    database_engine: Engine, email: object
) -> None:
    """Reject malformed or oversized input before database access.

    Args:
        database_engine: Engine whose application tables have not been initialized.
        email: Untrusted input deliberately violating the typed API contract.
    """
    with pytest.raises(ValueError, match="Customer email"):
        get_order_status(database_engine, order_id=601, customer_email=cast(str, email))


def test_maximum_email_length_is_accepted(tracking_engine: Engine) -> None:
    """Allow an exact 254-character synthetic address without provider-specific limits.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    email = "a" * 63 + "@" + "b" * 63 + "." + "c" * 63 + "." + "d" * 57 + ".test"
    assert len(email) == 254
    with Session(tracking_engine) as session, session.begin():
        customer = session.get(Customer, 501)
        assert customer is not None
        customer.email = email
    result = get_order_status(tracking_engine, order_id=601, customer_email=email)
    assert result is not None
    assert result.order_id == 601


def test_both_lookup_inputs_are_required(database_engine: Engine) -> None:
    """Reject ID-only and email-only calls before database access can occur.

    Args:
        database_engine: Engine whose application tables have not been initialized.
    """
    with pytest.raises(TypeError, match="customer_email"):
        get_order_status(database_engine, order_id=601)
    with pytest.raises(TypeError, match="order_id"):
        get_order_status(database_engine, customer_email="unique@example.test")


def test_result_is_minimal_detached_and_consistent(tracking_engine: Engine) -> None:
    """Serialize only tracking facts, consistently returning the single requested order.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    result = get_order_status(
        tracking_engine, order_id=601, customer_email="unique@example.test"
    )
    assert result is not None
    assert result == get_order_status(
        tracking_engine, order_id=601, customer_email="UNIQUE@EXAMPLE.TEST"
    )
    serialized = json.loads(result.model_dump_json())
    assert set(serialized) == {
        "order_id",
        "status",
        "order_date",
        "estimated_delivery",
        "tracking_code",
    }
    assert serialized["order_id"] == 601
    for private_value in (
        "unique@example.test",
        "Invented Tracking Customer",
        "+55-00-0000-0000",
        "Invented Tracking City",
        "Internal synthetic note",
    ):
        assert private_value not in result.model_dump_json()
    with pytest.raises(ValueError, match="frozen"):
        result.status = "invented_status"


def _source_snapshot(engine: Engine) -> dict[str, list[dict[str, object]]]:
    """Read all isolated source records to detect any mutation by lookup operations.

    Args:
        engine: Disposable test engine containing only synthetic data.

    Returns:
        Every operational table in deterministic primary-key order.
    """
    with Session(engine) as session:
        return {
            name: [
                dict(row)
                for row in session.execute(
                    select(table).order_by(*table.primary_key.columns)
                ).mappings()
            ]
            for name, table in Base.metadata.tables.items()
        }


def test_lookups_are_single_query_reads_and_preserve_data(
    tracking_engine: Engine,
) -> None:
    """Verify association checks use read-only bounded projections without mutations.

    Args:
        tracking_engine: Isolated synthetic tracking records.
    """
    before = _source_snapshot(tracking_engine)
    statements: list[str] = []

    def record_statement(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        """Observe actual database statements without recording sensitive bound values.

        Args:
            connection: Actual SQLAlchemy connection.
            cursor: Actual driver cursor.
            statement: Parameterized SQL text.
            parameters: Bound values, deliberately excluded from observations.
            context: SQLAlchemy execution metadata.
            executemany: Whether the execution batches parameter sets.
        """
        statements.append(statement)

    event.listen(tracking_engine, "before_cursor_execute", record_statement)
    try:
        assert (
            get_order_status(
                tracking_engine, order_id=601, customer_email="unique@example.test"
            )
            is not None
        )
        assert (
            get_order_status(
                tracking_engine, order_id=601, customer_email="other@example.test"
            )
            is None
        )
        assert (
            get_order_status(
                tracking_engine, order_id=301, customer_email="shared@example.test"
            )
            is None
        )
    finally:
        event.remove(tracking_engine, "before_cursor_execute", record_statement)
    assert len(statements) == 3
    assert all(statement.lstrip().startswith("SELECT") for statement in statements)
    assert _source_snapshot(tracking_engine) == before


def test_logging_omits_customer_values(
    tracking_engine: Engine, caplog: LogCaptureFixture, capsys: CaptureFixture[str]
) -> None:
    """Keep lookup inputs and source PII out of stdout, stderr, and SQL parameter logs.

    Args:
        tracking_engine: Existing engine configured to hide SQL parameters.
        caplog: Captured library logging output.
        capsys: Captured stdout and stderr.
    """
    with caplog.at_level(logging.INFO, logger="sqlalchemy.engine"):
        assert (
            get_order_status(
                tracking_engine, order_id=601, customer_email="unique@example.test"
            )
            is not None
        )
        assert (
            get_order_status(
                tracking_engine, order_id=601, customer_email="other@example.test"
            )
            is None
        )
    output = capsys.readouterr()
    observed = caplog.text + output.out + output.err
    for private_value in (
        "unique@example.test",
        "other@example.test",
        "Invented Tracking Customer",
        "+55-00-0000-0000",
        "Internal synthetic note",
    ):
        assert private_value not in observed


def test_database_failures_are_sanitized(database_engine: Engine) -> None:
    """Report an actual PostgreSQL schema error without exposing SQL or input values.

    Args:
        database_engine: Engine intentionally missing application tables.
    """
    email = "private-marker@example.test"
    with pytest.raises(
        OrderLookupError, match="Order lookup could not be completed"
    ) as raised:
        get_order_status(database_engine, order_id=601, customer_email=email)
    rendered = "".join(format_exception(raised.value))
    assert email not in rendered
    assert "SELECT" not in rendered
    assert raised.value.__suppress_context__
