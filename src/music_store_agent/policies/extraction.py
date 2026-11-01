"""Layout-preserving PDF extraction with conservative removal of repeated margins."""

import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from music_store_agent.policies.errors import PolicyExtractionError


@dataclass(frozen=True)
class PolicyPage:
    """Normalized source lines tied to a one-based physical PDF page."""

    page_number: int
    lines: tuple[str, ...]


def extract_pages(source: bytes) -> tuple[PolicyPage, ...]:
    """Extract Unicode text and preserve row structure without OCR or source logging.

    Args:
        source: Complete PDF bytes, also used by the caller for checksum calculation.

    Returns:
        Physical pages with normalized lines. Horizontal column gaps become pipes;
        text, punctuation, accents, and numeric values remain source-derived.

    Raises:
        PolicyExtractionError: If the PDF is invalid, encrypted, or contains a page
            without usable text. Textless pages fail closed rather than losing rules.
    """
    try:
        reader = PdfReader(BytesIO(source), strict=True)
        if reader.is_encrypted or not reader.pages:
            raise PolicyExtractionError("PDF requires an accessible text layer.")
        raw_pages = []
        for page in reader.pages:
            text = (
                page.extract_text(
                    extraction_mode="layout",
                    layout_mode_space_vertically=False,
                    layout_mode_strip_rotated=False,
                )
                or ""
            )
            lines = tuple(line.strip() for line in text.splitlines() if line.strip())
            if not lines or not any(character.isalpha() for character in text):
                raise PolicyExtractionError("Every PDF page requires usable text.")
            raw_pages.append(lines)
    except PolicyExtractionError:
        raise
    except Exception:
        # PDF parser diagnostics may contain source bytes; expose a fixed message.
        raise PolicyExtractionError(
            "PDF could not be read as a text document."
        ) from None
    headers = Counter(lines[0] for lines in raw_pages)
    footers = Counter(lines[-1] for lines in raw_pages)
    bare_footer_pages = sum(
        lines[-1].isdigit() and int(lines[-1]) in (number, number - 1)
        for number, lines in enumerate(raw_pages, 1)
    )
    pages = []
    for number, lines in enumerate(raw_pages, 1):
        cleaned = []
        for position, line in enumerate(lines):
            # Only repeated margin positions are removed; identical body text stays.
            if position == 0 and headers[line] >= 2 and not re.match(r"^\d", line):
                continue
            if position == len(lines) - 1:
                explicit_page = re.fullmatch(
                    r"(?i)p[aá]gina\s+\d+(?:\s*(?:de|/)\s*\d+)?", line
                )
                bare_page = (
                    bare_footer_pages >= max(2, len(raw_pages) - 1)
                    and line.isdigit()
                    and int(line) in (number, number - 1)
                )
                repeated_footer = footers[line] >= 2 and not re.match(r"^\d", line)
                if explicit_page or bare_page or repeated_footer:
                    continue
            cleaned.append(re.sub(r"[ \t]{3,}", " | ", line))
        if not cleaned:
            raise PolicyExtractionError("Every PDF page requires searchable body text.")
        pages.append(PolicyPage(number, tuple(cleaned)))
    return tuple(pages)
