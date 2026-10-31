"""Exact, non-cumulative promotion and payment-dependent pricing rules."""

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum

from pydantic import BaseModel, Field

CENT: Decimal = Decimal("0.01")
PIX_DISCOUNT_PERCENT: Decimal = Decimal("5")


class PaymentMethod(StrEnum):
    """Supported quotation contexts; only PIX changes non-promotional prices."""

    PIX = "pix"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BOLETO = "boleto"


class DiscountSource(StrEnum):
    """Identify the single discount used by a verified quotation."""

    NONE = "none"
    PROMOTION = "promotion"
    PIX = "pix"


class PricingStatus(StrEnum):
    """Distinguish verified prices from conflicting active promotion records."""

    VERIFIED = "verified"
    CONFLICT = "conflict"


class PromotionDiscount(BaseModel, frozen=True):
    """A source promotion used as input to framework-independent pricing rules."""

    promotion_id: int
    discount_percent: Decimal = Field(strict=True)
    description: str
    is_active: bool


class PriceQuote(BaseModel, frozen=True):
    """Expose price bases, payment context, and the resulting single discount.

    Decimal fields remain Decimal in Python and serialize as exact JSON strings.
    Conflicts preserve the list price and active records but suppress all computed
    prices and the selected discount. A verified price is not proof of availability.
    """

    list_price: Decimal
    advertised_price: Decimal | None
    promotional_price: Decimal | None
    pix_price: Decimal | None
    quoted_price: Decimal | None
    discount_percent: Decimal | None
    discount_source: DiscountSource | None
    payment_method: PaymentMethod | None
    status: PricingStatus
    price_reduced: bool
    active_promotions: tuple[PromotionDiscount, ...]


def calculate_pricing(
    list_price: Decimal,
    promotions: Sequence[PromotionDiscount] = (),
    *,
    payment_method: PaymentMethod | str | None = None,
) -> PriceQuote:
    """Compute one verified price or report ambiguous active promotions.

    Args:
        list_price: Nonnegative, finite source price with exact cent precision.
        promotions: Source records; inactive records never influence prices.
        payment_method: Optional explicit payment context; omitted means no PIX.

    Returns:
        List and applicable prices with exact discount and conflict metadata.
        Multiple active records suppress computed prices, including PIX. A single
        0% active record blocks stacking but does not claim a price reduction.

    Raises:
        ValueError: If the price, an active discount, or payment method is invalid.
    """
    if (
        not isinstance(list_price, Decimal)
        or not list_price.is_finite()
        or list_price < 0
    ):
        raise ValueError("List price must be a nonnegative Decimal with exact cents.")
    try:
        exact_cents = list_price == list_price.quantize(CENT)
    except InvalidOperation:
        exact_cents = False
    if not exact_cents:
        raise ValueError("List price must be a nonnegative Decimal with exact cents.")
    method = PaymentMethod(payment_method) if payment_method is not None else None
    active = tuple(
        sorted(
            (promotion for promotion in promotions if promotion.is_active),
            key=lambda promotion: promotion.promotion_id,
        )
    )
    for promotion in active:
        percent = promotion.discount_percent
        if not percent.is_finite() or not Decimal("0") <= percent <= Decimal("100"):
            raise ValueError(
                "Active discounts must be finite percentages from 0 to 100."
            )
    if len(active) > 1:
        # Source data provides no rule for choosing between simultaneous records.
        return PriceQuote(
            list_price=list_price,
            advertised_price=None,
            promotional_price=None,
            pix_price=None,
            quoted_price=None,
            discount_percent=None,
            discount_source=None,
            payment_method=method,
            status=PricingStatus.CONFLICT,
            price_reduced=False,
            active_promotions=active,
        )
    promotional_price = None
    pix_price = None
    percent = Decimal("0")
    source = DiscountSource.NONE
    advertised_price = list_price
    quoted_price = list_price
    if active:
        percent = active[0].discount_percent
        promotional_price = (
            list_price * (Decimal("1") - percent / Decimal("100"))
        ).quantize(CENT, rounding=ROUND_HALF_UP)
        advertised_price = quoted_price = promotional_price
        source = DiscountSource.PROMOTION
    elif method == PaymentMethod.PIX:
        # PIX always starts from the stored list price, never a discounted result.
        percent = PIX_DISCOUNT_PERCENT
        pix_price = (list_price * Decimal("0.95")).quantize(
            CENT, rounding=ROUND_HALF_UP
        )
        quoted_price = pix_price
        source = DiscountSource.PIX
    return PriceQuote(
        list_price=list_price,
        advertised_price=advertised_price,
        promotional_price=promotional_price,
        pix_price=pix_price,
        quoted_price=quoted_price,
        discount_percent=percent,
        discount_source=source,
        payment_method=method,
        status=PricingStatus.VERIFIED,
        price_reduced=quoted_price < list_price,
        active_promotions=active,
    )
