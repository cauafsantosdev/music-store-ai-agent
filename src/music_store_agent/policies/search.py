"""Small bilingual retrieval vocabulary and safe PostgreSQL full-text queries."""

import re

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from music_store_agent.database.policies import PolicyDocument, PolicySection
from music_store_agent.policies.errors import PolicyIndexMissing

MAX_QUERY_LENGTH: int = 500
MAX_POLICY_RESULTS: int = 10
MAX_EXCERPT_LENGTH: int = 1200
# Vocabulary expands retrieval concepts only; no deadlines, answers, or rules live here.
LEXICAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "opening hours": ("horário funcionamento",),
    "hours": ("horário funcionamento",),
    "address": ("endereço",),
    "contact": ("contato",),
    "customer service": ("atendimento",),
    "support": ("atendimento", "contato"),
    "payment methods": ("pagamento",),
    "payment": ("pagamento",),
    "pay": ("pagamento",),
    "installments": ("parcelamento",),
    "installment": ("parcelamento",),
    "exchanges": ("troca",),
    "exchange": ("troca",),
    "returns": ("devolução", "devoluções", "troca"),
    "return": ("devolução", "devoluções", "troca"),
    "refunds": ("reembolso", "estorno", "devolução", "devoluções"),
    "refund": ("reembolso", "estorno", "devolução", "devoluções"),
    "shipping": ("entrega", "frete", "envio"),
    "delivery": ("entrega", "frete", "envio"),
    "tracking": ("rastreamento", "rastreio"),
    # Portuguese Snowball keeps distinct lexemes for these singular/plural forms.
    "promotions": ("promoção", "promoções", "desconto"),
    "promotion": ("promoção", "promoções", "desconto"),
    "promoção": ("promoção", "promoções"),
    "promoções": ("promoção", "promoções"),
    "devolução": ("devolução", "devoluções"),
    "devoluções": ("devolução", "devoluções"),
    "discounts": ("desconto",),
    "discount": ("desconto",),
    "warranty": ("garantia",),
    "privacy": ("privacidade", "dados pessoais"),
}
ENGLISH_FILLER: frozenset[str] = frozenset(
    {
        "what",
        "when",
        "where",
        "how",
        "which",
        "are",
        "is",
        "do",
        "does",
        "can",
        "i",
        "my",
        "your",
        "you",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "by",
        "and",
        "please",
        "tell",
        "me",
        "about",
        "store",
        "shop",
        "policy",
        "policies",
        "long",
        "or",
        "on",
        "in",
        "accept",
        "accepted",
        "available",
        "offer",
        "work",
        "works",
        "information",
    }
)


def validate_search(query: str, limit: int) -> str:
    """Validate bounded public search inputs without echoing potentially private text.

    Args:
        query: Customer question in Portuguese or supported English vocabulary.
        limit: Maximum returned units, from 1 through 10.

    Returns:
        Trimmed query text.

    Raises:
        ValueError: If query text or the result limit is invalid.
    """
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > MAX_QUERY_LENGTH
        or "\x00" in query
    ):
        raise ValueError("Policy query must contain 1 to 500 characters of text.")
    if type(limit) is not int or not 1 <= limit <= MAX_POLICY_RESULTS:
        raise ValueError("Policy result limit must be an integer from 1 through 10.")
    return query.strip()


def build_text_query(query: str) -> ColumnElement[object]:
    """Construct AND groups of literal terms and tightly scoped bilingual concepts.

    Args:
        query: Validated query; values are always passed as bound parameters.

    Returns:
        A tsquery expression. Alternative concept vocabulary uses OR within one
        group; distinct concepts and remaining natural-language terms use AND.
    """
    words = re.findall(r"[^\W_]+", query.casefold())
    groups: list[tuple[str, ...]] = []
    literal: list[str] = []
    position = 0
    while position < len(words):
        phrase = " ".join(words[position : position + 2])
        if phrase in LEXICAL_CONCEPTS:
            groups.append(LEXICAL_CONCEPTS[phrase])
            position += 2
        elif words[position] in LEXICAL_CONCEPTS:
            groups.append(LEXICAL_CONCEPTS[words[position]])
            position += 1
        else:
            if words[position] not in ENGLISH_FILLER:
                literal.append(words[position])
            position += 1
    if literal or not groups:
        groups.append((" ".join(literal),))
    expression = None
    for alternatives in groups:
        concept = None
        for term in alternatives:
            value = func.plainto_tsquery("portuguese", term)
            concept = value if concept is None else concept.op("||")(value)
        expression = concept if expression is None else expression.op("&&")(concept)
    return expression


def retrieve_sections(
    session: Session, query: str, limit: int
) -> list[dict[str, object]]:
    """Rank bounded source excerpts using indexed Portuguese full-text vectors.

    Args:
        session: Caller-owned read session with private SQL parameters hidden.
        query: Validated natural-language question.
        limit: Validated result count.

    Returns:
        Safe source labels, numbered sections, physical pages, and original excerpts.

    Raises:
        PolicyIndexMissing: If no completed policy source is available.
        sqlalchemy.exc.SQLAlchemyError: If database access fails.
    """
    if session.scalar(select(PolicySection.section_id).limit(1)) is None:
        raise PolicyIndexMissing("Policy index is not initialized.")
    terms = build_text_query(query)
    excerpt = func.ts_headline(
        "portuguese",
        PolicySection.content,
        terms,
        'StartSel="", StopSel="", MaxWords=120, MinWords=30, MaxFragments=1',
    )
    statement = (
        select(
            PolicyDocument.document_id,
            PolicyDocument.label.label("document"),
            PolicySection.section_number.label("section"),
            PolicySection.title,
            PolicySection.page_start,
            PolicySection.page_end,
            excerpt.label("excerpt"),
        )
        .join(PolicyDocument, PolicyDocument.document_id == PolicySection.document_id)
        .where(PolicySection.search_vector.bool_op("@@")(terms))
        .order_by(
            func.ts_rank_cd(PolicySection.search_vector, terms).desc(),
            PolicyDocument.document_id,
            PolicySection.document_order,
        )
        .limit(limit)
    )
    results = []
    for row in session.execute(statement).mappings():
        result = dict(row)
        text = str(result["excerpt"])
        if len(text) > MAX_EXCERPT_LENGTH:
            text = text[: MAX_EXCERPT_LENGTH - 1].rsplit(" ", 1)[0] + "…"
        result["excerpt"] = text
        results.append(result)
    return results
