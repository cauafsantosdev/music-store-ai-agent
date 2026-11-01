"""Validate source bytes before transactionally replacing one logical policy index."""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import Engine, delete, func, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.database.policies import (
    PolicyDocument,
    PolicySection,
    initialize_policy_schema,
)
from music_store_agent.policies.errors import PolicyExtractionError, PolicyIndexError
from music_store_agent.policies.extraction import extract_pages
from music_store_agent.policies.segmentation import segment_pages


@dataclass(frozen=True)
class PolicyIndexSummary:
    """Non-sensitive completion metadata for an explicit policy index operation."""

    document_id: str
    checksum: str
    page_count: int
    section_count: int


def index_policy_pdf(
    engine: Engine,
    source: Path,
    *,
    document_id: str = "store-policy-manual",
    label: str = "Store Policy Manual",
) -> PolicyIndexSummary:
    """Atomically replace one document after extracting and validating its sections.

    Args:
        engine: Existing PostgreSQL engine, optionally targeting an isolated schema.
        source: Explicit local PDF path; never persisted or included in errors.
        document_id: Stable logical identifier, independent of paths and checksums.
        label: Safe display label; PDF metadata is deliberately not trusted for labels.

    Returns:
        Logical identity, checksum, and extracted page/section counts.

    Raises:
        ValueError: If logical source metadata is invalid.
        PolicyExtractionError: If source access, extraction, or segmentation fails.
        PolicyIndexError: If persistence fails; all previous content remains intact.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", document_id):
        raise ValueError("Document ID must be a safe logical identifier.")
    if (
        not label.strip()
        or len(label) > 120
        or not label.isprintable()
        or "/" in label
        or "\\" in label
    ):
        raise ValueError("Document label must be a short display name without paths.")
    try:
        source_bytes = source.read_bytes()
    except OSError:
        raise PolicyExtractionError("Policy PDF could not be read.") from None
    checksum = hashlib.sha256(source_bytes).hexdigest()
    pages = extract_pages(source_bytes)
    sections = segment_pages(pages, document_id=document_id)
    try:
        # DDL, source metadata, deletion, and replacement share one transaction.
        # The document upsert locks its row, serializing concurrent replacements.
        with engine.begin() as connection:
            initialize_policy_schema(connection)
            statement = pg_insert(PolicyDocument).values(
                document_id=document_id,
                label=label.strip(),
                checksum=checksum,
            )
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[PolicyDocument.document_id],
                    set_={
                        "label": statement.excluded.label,
                        "checksum": checksum,
                        "indexed_at": func.now(),
                    },
                )
            )
            connection.execute(
                delete(PolicySection).where(PolicySection.document_id == document_id)
            )
            payloads = [asdict(section) for section in sections]
            connection.execute(insert(PolicySection), payloads)
    except SQLAlchemyError:
        raise PolicyIndexError(
            "Policy indexing failed; previous index preserved."
        ) from None
    return PolicyIndexSummary(document_id, checksum, len(pages), len(sections))
