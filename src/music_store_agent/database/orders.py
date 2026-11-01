"""Minimal order projections guarded by exact, unambiguous customer association."""

from datetime import date

from sqlalchemy import Row, func, select
from sqlalchemy.orm import Session, aliased

from music_store_agent.database.models import Customer, Order
from music_store_agent.services.order_verification import EMAIL_WHITESPACE


def find_order_tracking(
    session: Session,
    *,
    order_id: int,
    customer_email: str,
) -> Row[tuple[int, str, date, date | None, str | None]] | None:
    """Select tracking facts only when the related email identifies one customer.

    Args:
        session: Caller-owned read session using the existing private PG configuration.
        order_id: Validated exact source identifier.
        customer_email: Trimmed, structurally valid email compared literally.

    Returns:
        The five public tracking fields, or None for every association failure.
        Customer records, email values, order notes, and purchased items are not loaded.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If the database query fails; public callers
            must sanitize operational failures before exposing them.
    """
    other_customer = aliased(Customer)
    # Shared trimming and database case mapping prevent normalization differences
    # from allowing an ambiguous stored identity to bypass duplicate detection.
    matching_email = func.lower(customer_email)
    duplicate = (
        select(1)
        .select_from(other_customer)
        .where(
            other_customer.customer_id != Customer.customer_id,
            func.lower(func.btrim(other_customer.email, EMAIL_WHITESPACE))
            == matching_email,
        )
        .correlate(Customer)
        .exists()
    )
    # The inner join fails closed for missing associations. EXISTS checks another
    # identity without loading the customer table or revealing duplicate counts.
    statement = (
        select(
            Order.order_id,
            Order.status,
            Order.order_date,
            Order.estimated_delivery,
            Order.tracking_code,
        )
        .join(Customer, Order.customer_id == Customer.customer_id)
        .where(
            Order.order_id == order_id,
            func.lower(func.btrim(Customer.email, EMAIL_WHITESPACE)) == matching_email,
            ~duplicate,
        )
    )
    return session.execute(statement).one_or_none()
