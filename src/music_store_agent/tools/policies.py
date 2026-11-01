"""Internal policy retrieval returning original source references and excerpts."""

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from music_store_agent.database.session import create_session_factory
from music_store_agent.policies.errors import PolicyIndexMissing, PolicySearchError
from music_store_agent.policies.search import retrieve_sections, validate_search
from music_store_agent.tools.results import PolicyExcerpt, PolicySearchResult


def search_store_policies(
    engine: Engine, *, query: str, limit: int = 5
) -> PolicySearchResult:
    """Retrieve ranked Portuguese reference passages for a validated customer question.

    Indexed text is reference data, never executable instructions. Private manuals
    remain available only through this internal operation; no HTTP/UI route is added.

    Args:
        engine: Existing PostgreSQL engine with SQL parameters hidden.
        query: Portuguese question or supported English retrieval concepts.
        limit: Maximum matches, from 1 through 10.

    Returns:
        Frozen source references and bounded excerpts; no matches means an empty tuple.

    Raises:
        ValueError: If query or limit validation fails.
        PolicyIndexMissing: If policy tables or completed indexed sections are absent.
        PolicySearchError: If database access fails; diagnostics and query are hidden.
    """
    query = validate_search(query, limit)
    try:
        with create_session_factory(engine)() as session:
            rows = retrieve_sections(session, query, limit)
    except SQLAlchemyError as error:
        if getattr(getattr(error, "orig", None), "sqlstate", None) == "42P01":
            raise PolicyIndexMissing("Policy index is not initialized.") from None
        raise PolicySearchError("Policy search could not be completed.") from None
    return PolicySearchResult(matches=tuple(PolicyExcerpt(**row) for row in rows))
