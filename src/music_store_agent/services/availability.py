"""Commercial availability and explicit catalog scope decisions."""

import re
from enum import StrEnum
from unicodedata import combining, normalize

from pydantic import BaseModel


class AvailabilityReason(StrEnum):
    """Machine-readable status and stock outcomes without future-date promises."""

    AVAILABLE = "available"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"
    COMING_SOON = "coming_soon"
    UNKNOWN_STATUS = "unknown_status"


class Availability(BaseModel, frozen=True):
    """Describe whether source status and stock permit a current sale."""

    available: bool
    reason: AvailabilityReason


def assess_availability(status: str, stock_quantity: int) -> Availability:
    """Require an active status and positive stock for commercial availability.

    Args:
        status: Exact source product status; unknown values fail closed.
        stock_quantity: Source inventory quantity.

    Returns:
        A structured availability decision, independent of pricing and catalog scope.
    """
    if status == "discontinued":
        reason = AvailabilityReason.DISCONTINUED
    elif status == "coming_soon":
        reason = AvailabilityReason.COMING_SOON
    elif status != "active":
        reason = AvailabilityReason.UNKNOWN_STATUS
    elif stock_quantity <= 0:
        reason = AvailabilityReason.OUT_OF_STOCK
    else:
        reason = AvailabilityReason.AVAILABLE
    return Availability(available=reason == AvailabilityReason.AVAILABLE, reason=reason)


def is_in_store_scope(category_name: str, category_description: str = "") -> bool:
    """Exclude explicitly labeled accessory and equipment categories.

    Args:
        category_name: Source category name in Portuguese or English.
        category_description: Retained for caller compatibility; excluded from scope
            decisions because descriptions can reference compatible equipment.

    Returns:
        False for explicit out-of-scope category names; otherwise True.
        Product identifiers and catalog-specific names are never consulted.
    """
    # Compatible accessories in descriptive text do not change category scope.
    decomposed = normalize("NFKD", category_name)
    text = "".join(
        character for character in decomposed if not combining(character)
    ).casefold()
    words = set(re.findall(r"[a-z]+", text))
    # These are excluded merchandise families, not a list of catalog records.
    # "Strings"/"cordas" alone is ambiguous with orchestral string instruments.
    excluded = {
        "accessory",
        "accessories",
        "acessorio",
        "acessorios",
        "pick",
        "picks",
        "palheta",
        "palhetas",
        "encordoamento",
        "encordoamentos",
        "cable",
        "cables",
        "cabo",
        "cabos",
        "case",
        "cases",
        "capa",
        "capas",
        "pedal",
        "pedals",
        "pedais",
        "amplifier",
        "amplifiers",
        "amplificador",
        "amplificadores",
    }
    return words.isdisjoint(excluded)
