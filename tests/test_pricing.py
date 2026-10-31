"""Unit tests for availability, scope, and exact non-cumulative pricing rules."""

from decimal import Decimal

import pytest

from music_store_agent.services.availability import (
    AvailabilityReason,
    assess_availability,
    is_in_store_scope,
)
from music_store_agent.services.pricing import (
    DiscountSource,
    PaymentMethod,
    PricingStatus,
    PromotionDiscount,
    calculate_pricing,
)


def _promotion(
    percent: str, *, promotion_id: int = 1, active: bool = True
) -> PromotionDiscount:
    """Construct an exact synthetic promotion without database dependencies.

    Args:
        percent: Exact decimal percentage text.
        promotion_id: Synthetic source identifier.
        active: Source activation flag.

    Returns:
        A typed promotion record for pricing rules.
    """
    return PromotionDiscount(
        promotion_id=promotion_id,
        discount_percent=Decimal(percent),
        description="Invented discount",
        is_active=active,
    )


@pytest.mark.parametrize(
    ("status", "stock", "reason"),
    [
        ("active", 2, AvailabilityReason.AVAILABLE),
        ("active", 0, AvailabilityReason.OUT_OF_STOCK),
        ("discontinued", 9, AvailabilityReason.DISCONTINUED),
        ("discontinued", 0, AvailabilityReason.DISCONTINUED),
        ("coming_soon", 9, AvailabilityReason.COMING_SOON),
        ("coming_soon", 0, AvailabilityReason.COMING_SOON),
        ("archived", 9, AvailabilityReason.UNKNOWN_STATUS),
    ],
)
def test_availability(status: str, stock: int, reason: AvailabilityReason) -> None:
    """Give status precedence over stock and fail closed for unknown statuses.

    Args:
        status: Synthetic source status.
        stock: Synthetic inventory quantity.
        reason: Expected machine-readable decision.
    """
    result = assess_availability(status, stock)
    assert result.reason == reason
    assert result.available is (reason == AvailabilityReason.AVAILABLE)


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [
        ("Guitarras", "Electric instruments", True),
        ("Cordas Orquestrais", "Orchestral string instruments", True),
        ("Strings", "Invented string instruments", True),
        ("Accessories", "Invented merchandise", False),
        ("ACESSÓRIOS", "Invented merchandise", False),
        ("Amplificadores", "Invented equipment", False),
        ("Cables and cases", "Invented merchandise", False),
        ("Pedais", "Invented pedals", False),
    ],
)
def test_catalog_scope(name: str, description: str, expected: bool) -> None:
    """Exclude explicit accessory families without rejecting string instruments.

    Args:
        name: Invented category name.
        description: Invented category description.
        expected: Expected scope decision.
    """
    assert is_in_store_scope(name, description) is expected


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [
        ("Guitarras", "Electric instruments compatible with pedals", True),
        ("Violões", "Acoustic instruments transported in cases", True),
        ("Acessórios", "Related products for guitar players", False),
        ("Amplificadores", "Equipment for electric instruments", False),
        ("Cordas Orquestrais", "Orchestral instruments transported in cases", True),
    ],
)
def test_scope_uses_category_labels(
    name: str, description: str, expected: bool
) -> None:
    """Preserve instrument categories when descriptions mention compatible equipment.

    Args:
        name: Explicit instrument, accessory, or equipment category label.
        description: Category description that may mention compatible equipment.
        expected: Expected scope decision based on the category label.
    """
    assert is_in_store_scope(name, description) is expected


def test_default_price_does_not_assume_pix() -> None:
    """Retain the stored list price when no payment context is supplied."""
    result = calculate_pricing(Decimal("1234.56"))
    assert (
        result.list_price
        == result.advertised_price
        == result.quoted_price
        == Decimal("1234.56")
    )
    assert result.payment_method is None
    assert result.pix_price is None
    assert result.promotional_price is None
    assert result.discount_source == DiscountSource.NONE
    assert not result.price_reduced


@pytest.mark.parametrize(
    ("price", "percent", "expected"),
    [
        ("1234.56", "10.50", "1104.93"),
        ("0.05", "10", "0.05"),
        ("20.00", "0", "20.00"),
        ("20.00", "100", "0.00"),
        ("999999999999.99", "99.99", "100000000.00"),
    ],
)
def test_promotional_decimal_precision(price: str, percent: str, expected: str) -> None:
    """Use source prices and ROUND_HALF_UP, including zero and full discounts.

    Args:
        price: Exact original list price.
        percent: Exact active discount percentage.
        expected: Expected final two-decimal price.
    """
    result = calculate_pricing(Decimal(price), [_promotion(percent)])
    assert result.list_price == Decimal(price)
    assert (
        result.promotional_price
        == result.advertised_price
        == result.quoted_price
        == Decimal(expected)
    )
    assert result.discount_percent == Decimal(percent)
    assert result.discount_source == DiscountSource.PROMOTION
    assert result.active_promotions[0].description == "Invented discount"
    assert result.price_reduced is (Decimal(expected) < Decimal(price))


def test_pix_is_conditional_and_rounded_half_up() -> None:
    """Round a half-cent upward and keep the advertised price independent of PIX."""
    result = calculate_pricing(Decimal("0.30"), payment_method="pix")
    assert result.list_price == result.advertised_price == Decimal("0.30")
    assert result.pix_price == result.quoted_price == Decimal("0.29")
    assert result.discount_percent == Decimal("5")
    assert result.discount_source == DiscountSource.PIX
    assert result.payment_method == PaymentMethod.PIX


@pytest.mark.parametrize("percent", ["10", "0", "100"])
def test_pix_never_stacks_with_active_promotions(percent: str) -> None:
    """Retain the active promotion even when it produces no reduction.

    Args:
        percent: Active promotional percentage, including both range boundaries.
    """
    regular = calculate_pricing(Decimal("100.00"), [_promotion(percent)])
    pix = calculate_pricing(
        Decimal("100.00"), [_promotion(percent)], payment_method="pix"
    )
    assert pix.quoted_price == regular.quoted_price
    assert pix.pix_price is None
    assert pix.discount_source == DiscountSource.PROMOTION
    assert pix.payment_method == PaymentMethod.PIX
    if percent == "0":
        assert not pix.price_reduced


def test_historical_inactive_promotions_never_influence_quotes() -> None:
    """Ignore multiple inactive records and allow PIX from the list price."""
    promotions = [
        _promotion("90", active=False),
        _promotion("50", promotion_id=2, active=False),
    ]
    result = calculate_pricing(Decimal("100.00"), promotions, payment_method="pix")
    assert result.quoted_price == Decimal("95.00")
    assert result.discount_source == DiscountSource.PIX
    assert result.active_promotions == ()


@pytest.mark.parametrize("second_percent", ["10", "20"])
def test_multiple_active_promotions_are_ambiguous(second_percent: str) -> None:
    """Reject arbitrary selection even when active records have equal percentages.

    Args:
        second_percent: Same or different discount in the second active record.
    """
    result = calculate_pricing(
        Decimal("100.00"),
        [
            _promotion(second_percent, promotion_id=2),
            _promotion("10"),
            _promotion("50", promotion_id=3, active=False),
        ],
        payment_method="pix",
    )
    assert result.status == PricingStatus.CONFLICT
    assert result.list_price == Decimal("100.00")
    assert result.advertised_price is None
    assert result.quoted_price is None
    assert result.promotional_price is None
    assert result.pix_price is None
    assert result.discount_source is None
    assert result.discount_percent is None
    assert not result.price_reduced
    assert [row.promotion_id for row in result.active_promotions] == [1, 2]


@pytest.mark.parametrize("price", ["-0.01", "NaN", "Infinity", "1.001", "1E+100"])
def test_invalid_list_prices_are_rejected(price: str) -> None:
    """Reject negative, nonfinite, and fractional-cent monetary inputs.

    Args:
        price: Invalid domain input represented exactly as a Decimal.
    """
    with pytest.raises(ValueError, match="List price"):
        calculate_pricing(Decimal(price))


@pytest.mark.parametrize("percent", ["-0.01", "100.01"])
def test_invalid_active_discounts_are_rejected(percent: str) -> None:
    """Prevent negative prices and invalid percentage ranges outside the database.

    Args:
        percent: Out-of-range active percentage.
    """
    with pytest.raises(ValueError, match="Active discounts"):
        calculate_pricing(Decimal("100.00"), [_promotion(percent)])


def test_unknown_payment_method_is_rejected() -> None:
    """Reject unsupported payment contexts rather than silently applying a rule."""
    with pytest.raises(ValueError):
        calculate_pricing(Decimal("100.00"), payment_method="unknown")


def test_cash_payment_method_is_rejected() -> None:
    """Reject cash because the accepted payment methods exclude cash payments."""
    with pytest.raises(ValueError):
        calculate_pricing(Decimal("100.00"), payment_method="cash")
