"""Separate policy metadata and PostgreSQL generated full-text vectors."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Computed,
    Connection,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.schema import CreateSchema


class PolicyBase(DeclarativeBase):
    """Keep policy tables outside the operational six-table metadata registry."""


class PolicyDocument(PolicyBase):
    """Logical source identity and checksum without private filesystem metadata."""

    __tablename__: str = "policy_documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(120))
    checksum: Mapped[str] = mapped_column(String(64))
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PolicySection(PolicyBase):
    """Source section with physical pages and an indexed weighted Portuguese vector."""

    __tablename__: str = "policy_sections"
    __table_args__: tuple[UniqueConstraint | CheckConstraint | Index, ...] = (
        UniqueConstraint(
            "document_id", "document_order", name="policy_section_order_unique"
        ),
        CheckConstraint(
            "page_start > 0 AND page_end >= page_start", name="policy_page_range_valid"
        ),
        CheckConstraint(
            "document_order >= 0", name="policy_document_order_nonnegative"
        ),
        Index("policy_sections_search_gin", "search_vector", postgresql_using="gin"),
    )

    section_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("policy_documents.document_id"))
    section_number: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    document_order: Mapped[int] = mapped_column(Integer)
    # The explicit regconfig makes the generated expression immutable. GIN indexes
    # the stored vector; source titles outrank body terms without computed answers.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('portuguese'::regconfig, title), 'A') || "
            "setweight(to_tsvector('portuguese'::regconfig, heading_path), 'B') || "
            "setweight(to_tsvector('portuguese'::regconfig, content), 'C')",
            persisted=True,
        ),
    )


def initialize_policy_schema(connection: Connection) -> None:
    """Create only policy tables within the caller's explicit transaction.

    Args:
        connection: Connection carrying the existing schema translation mapping.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If schema or policy table creation fails.
    """
    schema = (
        connection.get_execution_options().get("schema_translate_map", {}).get(None)
    )
    if schema is not None:
        connection.execute(CreateSchema(schema, if_not_exists=True))
    PolicyBase.metadata.create_all(connection)
