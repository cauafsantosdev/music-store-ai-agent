"""Internal demonstration order tracking with limited association verification."""

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.database.orders import find_order_tracking
from music_store_agent.database.session import create_session_factory
from music_store_agent.services.order_verification import (
    OrderLookupError,
    normalize_customer_email,
    validate_order_id,
)
from music_store_agent.tools.results import OrderTrackingResult


def get_order_status(
    engine: Engine,
    *,
    order_id: int,
    customer_email: str,
) -> OrderTrackingResult | None:
    """Return stored tracking facts after limited customer association verification.

    Order number and email knowledge do not authenticate identity. Public access
    to real customer data must remain disabled until authentication and authorization
    are implemented. Timing-side-channel resistance is not provided.

    Args:
        engine: Existing engine with SQL parameter logging disabled via hide_parameters.
        order_id: Positive source order identifier; booleans are rejected.
        customer_email: Exact associated email, ignoring case and outer whitespace.

    Returns:
        Minimal detached facts on success. Unknown orders, wrong emails, missing
        associations, and duplicate-email identities all return None without metadata.
        Stored status and dates are snapshots, not live carrier tracking information.

    Raises:
        ValueError: If either input is invalid; messages omit supplied values.
        OrderLookupError: If database access fails; diagnostic SQL and parameters
            are excluded from the public exception message and traceback chain.
    """
    validate_order_id(order_id)
    email = normalize_customer_email(customer_email)
    try:
        with create_session_factory(engine)() as session:
            record = find_order_tracking(
                session, order_id=order_id, customer_email=email
            )
    except SQLAlchemyError:
        # Driver exceptions can include row values even when parameters are hidden.
        raise OrderLookupError("Order lookup could not be completed.") from None
    if record is None:
        return None
    source_id, status, order_date, estimated_delivery, tracking_code = record
    return OrderTrackingResult(
        order_id=source_id,
        status=status,
        order_date=order_date,
        estimated_delivery=estimated_delivery,
        tracking_code=tracking_code,
    )
