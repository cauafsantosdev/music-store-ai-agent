"""Serializable public results containing only operation-specific source facts."""

from datetime import date

from pydantic import BaseModel, JsonValue

from music_store_agent.services.availability import Availability
from music_store_agent.services.pricing import PriceQuote


class CategoryDetails(BaseModel, frozen=True):
    """Source category identity, name, and description."""

    category_id: int
    name: str
    description: str


class ProductPricing(BaseModel, frozen=True):
    """Attach a quote to explicit product eligibility and offer decisions."""

    product_id: int
    availability: Availability
    in_store_scope: bool
    purchasable: bool
    verified_offer: bool
    quote: PriceQuote


class ProductDetails(ProductPricing, frozen=True):
    """Complete source catalog details plus evaluated pricing and availability."""

    name: str
    description: str
    category: CategoryDetails
    stock_quantity: int
    status: str
    specs: JsonValue
    created_at: date


class OrderTrackingResult(BaseModel, frozen=True):
    """Minimal tracking snapshot after demonstration-only association verification.

    The result does not establish authenticated ownership. Delivery dates remain
    estimates, and no current shipment events or actual delivery dates are inferred.
    """

    order_id: int
    status: str
    order_date: date
    estimated_delivery: date | None
    tracking_code: str | None


class PolicyExcerpt(BaseModel, frozen=True):
    """A bounded source passage with a safe label and one-based physical PDF pages."""

    document_id: str
    document: str
    section: str
    title: str
    page_start: int
    page_end: int
    excerpt: str


class PolicySearchResult(BaseModel, frozen=True):
    """Ranked reference passages, not generated answers or calibrated confidence."""

    matches: tuple[PolicyExcerpt, ...]
