"""Serializable catalog results containing only catalog and commercial information."""

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
