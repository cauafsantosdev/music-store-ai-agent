"""Shared SQLAlchemy declarative metadata without connection side effects."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Collect application table mappings without creating database objects."""
