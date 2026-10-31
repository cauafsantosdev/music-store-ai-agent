"""Read-only, parameterized catalog queries with eager loading and bounded batches."""

from collections.abc import Iterator

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session, contains_eager, selectinload

from music_store_agent.database.models import Category, Product, Promotion

CATALOG_BATCH_SIZE: int = 100


def _product_query() -> Select[tuple[Product]]:
    """Build a catalog-only query loading categories and active promotions eagerly.

    Returns:
        A query without customer, order, or order-item data.
    """
    return (
        select(Product)
        .join(Product.category)
        .options(
            contains_eager(Product.category),
            selectinload(Product.promotions.and_(Promotion.is_active.is_(True))),
        )
    )


def find_product(session: Session, product_id: int) -> Product | None:
    """Load a source product with its category and active promotion records.

    Args:
        session: Caller-owned read session.
        product_id: Source product identifier.

    Returns:
        A fully loaded catalog record, or None if absent.
    """
    return session.scalar(_product_query().where(Product.product_id == product_id))


def iter_products(
    session: Session,
    *,
    query: str | None = None,
    category: str | None = None,
    available_only: bool = True,
    active_promotions_only: bool = False,
) -> Iterator[Product]:
    """Yield matching products in ascending source-ID order using bounded queries.

    Args:
        session: Caller-owned read session.
        query: Literal case-insensitive substring of name, description, or category.
        category: Literal case-insensitive category-name substring.
        available_only: Require active status and positive stock when True.
        active_promotions_only: Require at least one active promotion when True.

    Yields:
        Products with eagerly loaded categories and active promotions.
        Percent, underscore, and escape characters are treated literally.
    """
    statement = _product_query()
    if query:
        statement = statement.where(
            or_(
                Product.name.icontains(query, autoescape=True),
                Product.description.icontains(query, autoescape=True),
                Category.name.icontains(query, autoescape=True),
            )
        )
    if category:
        statement = statement.where(Category.name.icontains(category, autoescape=True))
    if available_only:
        statement = statement.where(
            Product.status == "active", Product.stock_quantity > 0
        )
    if active_promotions_only:
        statement = statement.where(
            Product.promotions.any(Promotion.is_active.is_(True))
        )
    last_id: int | None = None
    while True:
        batch_query = statement
        if last_id is not None:
            batch_query = batch_query.where(Product.product_id > last_id)
        products = session.scalars(
            batch_query.order_by(Product.product_id).limit(CATALOG_BATCH_SIZE)
        ).all()
        if not products:
            return
        yield from products
        last_id = products[-1].product_id
        if len(products) < CATALOG_BATCH_SIZE:
            return
