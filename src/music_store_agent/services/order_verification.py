"""Input validation for demonstration-only order/customer association checks."""

import re

MAX_EMAIL_LENGTH: int = 254
MAX_ORDER_ID: int = 2_147_483_647
# Share Python 3.12's whitespace set with SQL btrim so duplicate detection uses
# exactly the same edge normalization as submitted addresses, including Unicode.
EMAIL_WHITESPACE: str = (
    "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


class OrderLookupError(RuntimeError):
    """Report an operational lookup failure without SQL parameters or customer data."""


def validate_order_id(order_id: int) -> None:
    """Require a positive source identifier within the PostgreSQL INTEGER range.

    Args:
        order_id: Untrusted source order identifier; booleans are not identifiers.

    Raises:
        ValueError: If the identifier is not a positive, representable integer.
            The supplied value is never included in the error message.
    """
    if type(order_id) is not int or not 1 <= order_id <= MAX_ORDER_ID:
        raise ValueError("Order ID must be a positive PostgreSQL integer.")


def normalize_customer_email(customer_email: str) -> str:
    """Trim and structurally validate an email without provider-specific rewriting.

    Args:
        customer_email: Untrusted email with at most 254 characters, including padding.

    Returns:
        The trimmed email. The query applies PostgreSQL lowercasing to both sides
        of the comparison, keeping Unicode case behavior consistent. Dots and
        plus tags remain unchanged; this is structural validation, not verification.

    Raises:
        ValueError: If the input is not text, is oversized, or has an invalid shape.
            Errors never include the supplied email.
    """
    if not isinstance(customer_email, str) or len(customer_email) > MAX_EMAIL_LENGTH:
        raise ValueError("Customer email must be text of at most 254 characters.")
    email = customer_email.strip(EMAIL_WHITESPACE)
    if (
        not email.isprintable()
        or re.fullmatch(r"[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)+", email) is None
    ):
        raise ValueError("Customer email must have a valid address format.")
    return email
