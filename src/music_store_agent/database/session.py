"""Lazy engine configuration, session creation, and explicit schema initialization."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateSchema

from music_store_agent.database import models


def create_database_engine(*, schema: str = "public") -> Engine:
    """Configure a lazy psycopg engine using the existing libpq PG variables.

    Args:
        schema: Target namespace for mapped tables, including isolated test schemas.

    Returns:
        An engine that has not yet connected to PostgreSQL.

    Raises:
        ValueError: If the schema name is empty.
    """
    if not schema.strip():
        raise ValueError("A nonempty PostgreSQL schema name is required.")
    # An empty URL delegates credentials to psycopg's PG* environment handling.
    # Schema translation qualifies both ORM queries and FK targets without changing
    # the connection search path or falling back to development tables in tests.
    return create_engine(
        "postgresql+psycopg://",
        connect_args={"connect_timeout": 5},
        pool_pre_ping=True,
        hide_parameters=True,
        execution_options={"schema_translate_map": {None: schema}},
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory without opening a connection.

    Args:
        engine: Configured PostgreSQL engine.

    Returns:
        A factory supporting explicit transactions through ``factory.begin()``.
    """
    return sessionmaker(bind=engine)


def initialize_schema(engine: Engine) -> None:
    """Create missing tables transactionally without altering or deleting data.

    Args:
        engine: Engine configured with the desired schema translation.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If PostgreSQL rejects schema creation.
    """
    schema = engine.get_execution_options().get("schema_translate_map", {}).get(None)
    with engine.begin() as connection:
        if schema is not None:
            connection.execute(CreateSchema(schema, if_not_exists=True))
        models.Base.metadata.create_all(connection)
