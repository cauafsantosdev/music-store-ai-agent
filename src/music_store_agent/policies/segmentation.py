"""Numbered policy sections that persist across physical PDF page boundaries."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from music_store_agent.policies.errors import PolicyExtractionError
from music_store_agent.policies.extraction import PolicyPage

HEADING: re.Pattern[str] = re.compile(
    r"^(\d+(?:\.\d+)*)(?:\.\s+|\s*[-–—]\s+|\s+)(\S.*)$"
)


@dataclass(frozen=True)
class PolicyUnit:
    """A complete numbered section with source hierarchy and physical provenance."""

    document_id: str
    section_number: str
    title: str
    heading_path: str
    content: str
    page_start: int
    page_end: int
    document_order: int


def segment_pages(
    pages: Sequence[PolicyPage], *, document_id: str
) -> tuple[PolicyUnit, ...]:
    """Group source lines by progressive numbered headings, never by page boundaries.

    Args:
        pages: Extracted physical pages in their original order.
        document_id: Stable logical source identifier assigned by explicit indexing.

    Returns:
        Nonempty section units in source order. Parent titles remain in heading paths
        even when a parent has no introductory body. Cover text is excluded.

    Raises:
        PolicyExtractionError: If no meaningful numbered sections can be identified.
    """
    units: list[PolicyUnit] = []
    headings: dict[tuple[int, ...], str] = {}
    current: tuple[int, ...] = ()
    body: list[str] = []
    start = end = 0

    def finish() -> None:
        """Emit accumulated source content while retaining the complete heading path."""
        if current and body:
            units.append(
                PolicyUnit(
                    document_id=document_id,
                    section_number=".".join(map(str, current)),
                    title=headings[current],
                    heading_path=" > ".join(
                        headings[current[:depth]]
                        for depth in range(1, len(current) + 1)
                    ),
                    content="\n".join(body),
                    page_start=start,
                    page_end=end,
                    document_order=len(units),
                )
            )

    for page in pages:
        for line in page.lines:
            match = HEADING.match(line)
            parts = tuple(map(int, match.group(1).split("."))) if match else ()
            # Main headings progress sequentially; child headings belong to the
            # active parent. Workflow items such as "1." under section 7 stay text.
            root = current[0] if current else 0
            is_main = len(parts) == 1 and parts[0] == root + 1
            is_child = (
                len(parts) > 1
                and parts[0] == root
                and parts > current
                and parts[:-1] in headings
            )
            if match and (is_main or is_child):
                finish()
                current = parts
                headings[current] = match.group(2).strip()
                body = []
                start = end = page.page_number
            elif current:
                body.append(line)
                end = page.page_number
    finish()
    if not units or not any(len(unit.content.split()) >= 3 for unit in units):
        raise PolicyExtractionError("PDF contains no usable numbered policy sections.")
    return tuple(units)
