"""Public catalog operations owning read sessions and applying deterministic rules."""

from decimal import Decimal, InvalidOperation

from sqlalchemy import Engine

from music_store_agent.database.catalog import find_product, iter_products
from music_store_agent.database.models import Product
from music_store_agent.database.session import create_session_factory
from music_store_agent.services.availability import (
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
from music_store_agent.tools.results import (
    CategoryDetails,
    ProductDetails,
    ProductPricing,
)

MAX_RESULTS: int = 100


def _validate_limit(limit: int) -> None:
    """Reject unbounded or noninteger customer result limits.

    Args:
        limit: Requested result count.

    Raises:
        ValueError: If the limit is not an integer from 1 through 100.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"Limit must be an integer from 1 through {MAX_RESULTS}.")


def _price_bound(value: Decimal | str | None) -> Decimal | None:
    """Validate exact nonnegative budget bounds without converting through floats.

    Args:
        value: Decimal, decimal string, or omitted bound.

    Returns:
        A finite nonnegative Decimal, or None. Bounds are not rounded.

    Raises:
        ValueError: If the input is malformed, negative, nonfinite, or inexactly typed.
    """
    if value is None:
        return None
    if not isinstance(value, (Decimal, str)):
        raise ValueError("Price bounds must be Decimals or decimal strings.")
    try:
        price = Decimal(value)
    except InvalidOperation:
        raise ValueError("Price bounds must be valid decimal strings.") from None
    if not price.is_finite() or price < 0:
        raise ValueError("Price bounds must be finite and nonnegative.")
    return price


def _details(product: Product, payment_method: PaymentMethod | None) -> ProductDetails:
    """Convert a loaded ORM record to a detached, fully evaluated public result.

    Args:
        product: Product with category and promotions eagerly loaded.
        payment_method: Explicit quotation context or no payment assumption.

    Returns:
        Source information and commercial outcomes, without operational customer data.
    """
    availability = assess_availability(product.status, product.stock_quantity)
    scope = is_in_store_scope(product.category.name, product.category.description)
    quote = calculate_pricing(
        product.price_brl,
        tuple(
            PromotionDiscount(
                promotion_id=promotion.promotion_id,
                discount_percent=promotion.discount_percent,
                description=promotion.description,
                is_active=promotion.is_active,
            )
            for promotion in product.promotions
        ),
        payment_method=payment_method,
    )
    purchasable = (
        availability.available and scope and quote.status == PricingStatus.VERIFIED
    )
    return ProductDetails(
        product_id=product.product_id,
        name=product.name,
        description=product.description,
        category=CategoryDetails(
            category_id=product.category.category_id,
            name=product.category.name,
            description=product.category.description,
        ),
        stock_quantity=product.stock_quantity,
        status=product.status,
        specs=product.specs,
        created_at=product.created_at,
        availability=availability,
        in_store_scope=scope,
        purchasable=purchasable,
        verified_offer=(
            purchasable
            and quote.discount_source == DiscountSource.PROMOTION
            and quote.price_reduced
        ),
        quote=quote,
    )


def search_products(
    engine: Engine,
    *,
    query: str | None = None,
    category: str | None = None,
    max_price_brl: Decimal | str | None = None,
    min_price_brl: Decimal | str | None = None,
    available_only: bool = True,
    payment_method: PaymentMethod | str | None = None,
    limit: int = 20,
) -> tuple[ProductDetails, ...]:
    """Search literal case-insensitive substrings and filter exact applicable prices.

    Args:
        engine: Existing engine targeting the catalog schema.
        query: Optional product-name, description, or category-name substring.
        category: Optional category-name substring; Portuguese accents are preserved.
        max_price_brl: Inclusive upper bound on the quote in the selected context.
        min_price_brl: Inclusive lower bound on the quote in the selected context.
        available_only: Exclude unavailable statuses and stock by default.
        payment_method: Optional explicit context; omitted means advertised pricing.
        limit: Result count from 1 through 100, ordered by ascending product ID.

    Returns:
        Catalog records, excluding accessory categories. Conflicts remain visible
        without budget bounds but are excluded whenever either bound is provided.
        Empty or whitespace-only search strings mean no text filter.

    Raises:
        ValueError: If a limit, payment method, or budget bound is invalid.
        sqlalchemy.exc.SQLAlchemyError: If a database query fails.
    """
    _validate_limit(limit)
    minimum, maximum = _price_bound(min_price_brl), _price_bound(max_price_brl)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Minimum price must not exceed maximum price.")
    method = PaymentMethod(payment_method) if payment_method is not None else None
    results: list[ProductDetails] = []
    with create_session_factory(engine)() as session:
        for product in iter_products(
            session,
            query=query.strip() if query else None,
            category=category.strip() if category else None,
            available_only=available_only,
        ):
            result = _details(product, method)
            if not result.in_store_scope:
                continue
            if minimum is not None or maximum is not None:
                price = result.quote.quoted_price
                # Filtering precedes the public limit; keyset batches avoid truncating
                # qualifying later products or resolving ambiguous discounts in SQL.
                if (
                    price is None
                    or (minimum is not None and price < minimum)
                    or (maximum is not None and price > maximum)
                ):
                    continue
            results.append(result)
            if len(results) == limit:
                break
    return tuple(results)


def get_product_details(
    engine: Engine,
    product_id: int,
    *,
    payment_method: PaymentMethod | str | None = None,
) -> ProductDetails | None:
    """Retrieve an explicit product, including unavailable or out-of-scope records.

    Args:
        engine: Existing catalog engine.
        product_id: Source product identifier.
        payment_method: Optional explicit quotation context.

    Returns:
        Complete catalog details with eligibility flags, or None for an unknown ID.

    Raises:
        ValueError: If the payment method is unsupported.
        sqlalchemy.exc.SQLAlchemyError: If a database query fails.
    """
    method = PaymentMethod(payment_method) if payment_method is not None else None
    with create_session_factory(engine)() as session:
        product = find_product(session, product_id)
        return _details(product, method) if product is not None else None


def get_product_pricing(
    engine: Engine,
    product_id: int,
    *,
    payment_method: PaymentMethod | str | None = None,
) -> ProductPricing | None:
    """Retrieve eligibility and exact prices without unrelated product details.

    Args:
        engine: Existing catalog engine.
        product_id: Source product identifier.
        payment_method: Optional explicit context, including PIX.

    Returns:
        Structured pricing and availability, or None for an unknown identifier.

    Raises:
        ValueError: If the payment method is unsupported.
        sqlalchemy.exc.SQLAlchemyError: If a database query fails.
    """
    details = get_product_details(engine, product_id, payment_method=payment_method)
    if details is None:
        return None
    return ProductPricing(
        **details.model_dump(include=set(ProductPricing.model_fields))
    )


def get_active_promotions(
    engine: Engine,
    *,
    limit: int = 20,
) -> tuple[ProductDetails, ...]:
    """Inspect active promotions on available in-scope products in source-ID order.

    Args:
        engine: Existing catalog engine.
        limit: Maximum number of products from 1 through 100.

    Returns:
        Products with active records. Conflicts are flagged as non-purchasable and
        not verified offers; 0% records never claim a reduction. Unavailable and
        accessory records are excluded, and inactive records are never exposed.

    Raises:
        ValueError: If the limit is invalid.
        sqlalchemy.exc.SQLAlchemyError: If a database query fails.
    """
    _validate_limit(limit)
    results: list[ProductDetails] = []
    with create_session_factory(engine)() as session:
        for product in iter_products(session, active_promotions_only=True):
            result = _details(product, None)
            if not result.in_store_scope:
                continue
            results.append(result)
            if len(results) == limit:
                break
    return tuple(results)
