"""Synthetic PDF and PostgreSQL retrieval tests independent of restricted manuals."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import (
    CheckConstraint,
    Engine,
    MetaData,
    Table,
    create_engine,
    func,
    inspect,
    select,
)
from sqlalchemy.schema import AddConstraint

from music_store_agent.data.ingestion import seed_data
from music_store_agent.database.models import Base
from music_store_agent.database.policies import (
    PolicyBase,
    PolicyDocument,
    PolicySection,
    initialize_policy_schema,
)
from music_store_agent.policies.cli import main
from music_store_agent.policies.errors import (
    PolicyExtractionError,
    PolicyIndexError,
    PolicyIndexMissing,
    PolicySearchError,
)
from music_store_agent.policies.extraction import extract_pages
from music_store_agent.policies.indexing import index_policy_pdf
from music_store_agent.policies.segmentation import segment_pages
from music_store_agent.tools.policies import search_store_policies

SAMPLE_PDF: Path = (
    Path(__file__).resolve().parents[1] / "data/sample/store_policies.pdf"
)


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Generate compact synthetic text PDFs with the existing pypdf dependency.

    Args:
        path: Temporary output path inside the container.
        pages: Invented page lines; no restricted text may be passed here.
    """
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    for number, lines in enumerate(pages):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                )
            }
        )
        contents = []
        all_lines = [(760, "Manual sintético de políticas")]
        all_lines.extend((720 - offset * 20, text) for offset, text in enumerate(lines))
        all_lines.append((20, f"Página {number}"))
        for y, text in all_lines:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            contents.append(
                f"BT /F1 11 Tf 42 {y} Td ({escaped}) Tj ET\n".encode("cp1252")
            )
        stream = DecodedStreamObject()
        stream.set_data(b"".join(contents))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.fixture
def policy_engine(database_engine: Engine) -> Engine:
    """Index only the public synthetic PDF in the existing isolated schema fixture.

    Args:
        database_engine: Disposable PostgreSQL schema with deterministic cleanup.

    Returns:
        An engine containing a complete synthetic policy index.
    """
    index_policy_pdf(database_engine, SAMPLE_PDF)
    return database_engine


def _sections(engine: Engine) -> list[dict[str, object]]:
    """Snapshot indexed source facts while excluding unstable internal identifiers.

    Args:
        engine: Isolated policy engine.

    Returns:
        Source section fields in deterministic logical order.
    """
    with engine.connect() as connection:
        rows = connection.execute(
            select(
                PolicySection.document_id,
                PolicySection.section_number,
                PolicySection.title,
                PolicySection.content,
                PolicySection.page_start,
                PolicySection.page_end,
                PolicySection.document_order,
            ).order_by(PolicySection.document_id, PolicySection.document_order)
        ).mappings()
        return [dict(row) for row in rows]


def test_pdf_hierarchy_pages_tables_and_accents() -> None:
    """Preserve synthetic hierarchy, accents, tables, numbers, and cross-page rules."""
    pages = extract_pages(SAMPLE_PDF.read_bytes())
    assert len(pages) == 4
    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    units = segment_pages(pages, document_id="synthetic")
    by_number = {unit.section_number: unit for unit in units}
    assert "MANUAL FICTÍCIO" not in " ".join(unit.content for unit in units)
    assert by_number["3.1"].title == "Parcelamento fictício"
    assert "Formas de pagamento" in by_number["3.1"].heading_path
    assert "13,37" in by_number["3.1"].content
    assert "3 parcelas" in by_number["3.1"].content
    assert by_number["2"].page_start == by_number["2"].page_end == 2
    assert "Segunda a sexta" in by_number["2"].content
    assert "10h às 16h" in by_number["2"].content
    assert " | " in by_number["2"].content
    assert by_number["5.2"].page_start == 3
    assert by_number["5.2"].page_end == 4
    assert "constelações" in by_number["5.2"].content
    assert "23 ciclos" in by_number["5.2"].content
    assert "1." in by_number["5.2"].content
    assert "2." in by_number["5.2"].content
    assert "Manual sintético de políticas" not in "\n".join(
        unit.content for unit in units
    )
    assert all("Página" not in unit.content for unit in units)
    assert [unit.document_order for unit in units] == list(range(len(units)))


@pytest.mark.parametrize("source", [b"", b"not a PDF", b"%PDF-1.7\ninvalid"])
def test_invalid_pdf_is_rejected(source: bytes) -> None:
    """Reject unreadable bytes with sanitized parser diagnostics.

    Args:
        source: Deliberately malformed public test bytes.
    """
    with pytest.raises(PolicyExtractionError):
        extract_pages(source)


def test_textless_and_headingless_pdfs_are_rejected(tmp_path: Path) -> None:
    """Reject a scanned-style blank page and a document without policy structure.

    Args:
        tmp_path: Container-local temporary files.
    """
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(blank)
    with pytest.raises(PolicyExtractionError):
        extract_pages(blank.read_bytes())
    headingless = tmp_path / "headingless.pdf"
    _write_pdf(headingless, [["Artificial content without numbered section headings."]])
    with pytest.raises(PolicyExtractionError, match="numbered"):
        segment_pages(extract_pages(headingless.read_bytes()), document_id="test")


def test_header_only_page_is_not_silently_discarded(tmp_path: Path) -> None:
    """Reject partial extraction instead of ignoring a page with margins only.

    Args:
        tmp_path: Container-local synthetic PDF path.
    """
    source = tmp_path / "partial.pdf"
    _write_pdf(
        source,
        [["1. Política sintética", "A regra artificial permanece disponível."], []],
    )
    with pytest.raises(PolicyExtractionError, match="body text"):
        extract_pages(source.read_bytes())


def test_policy_schema_is_separate_and_repeatable(database_engine: Engine) -> None:
    """Initialize only policy metadata, leaving the operational registry unchanged.

    Args:
        database_engine: Disposable engine without application tables.
    """
    for _ in range(2):
        with database_engine.begin() as connection:
            initialize_policy_schema(connection)
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    inspector = inspect(database_engine)
    assert set(inspector.get_table_names(schema=schema)) == set(
        PolicyBase.metadata.tables
    )
    assert len(Base.metadata.tables) == 6
    indexes = inspector.get_indexes("policy_sections", schema=schema)
    assert any(
        index["dialect_options"].get("postgresql_using") == "gin" for index in indexes
    )


def test_index_checksum_and_repeatability(policy_engine: Engine) -> None:
    """Repeat indexing without duplicates while preserving source bytes and checksum.

    Args:
        policy_engine: Complete synthetic index.
    """
    source_bytes = SAMPLE_PDF.read_bytes()
    before = _sections(policy_engine)
    summary = index_policy_pdf(policy_engine, SAMPLE_PDF)
    assert summary.checksum == hashlib.sha256(source_bytes).hexdigest()
    assert summary.page_count == 4
    assert summary.section_count == len(before)
    assert _sections(policy_engine) == before
    assert SAMPLE_PDF.read_bytes() == source_bytes
    with policy_engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(PolicyDocument)) == 1
        assert connection.scalar(select(PolicyDocument.checksum)) == summary.checksum


def test_changed_source_replaces_only_its_index(
    policy_engine: Engine, tmp_path: Path
) -> None:
    """Replace obsolete sections atomically while retaining an unrelated logical source.

    Args:
        policy_engine: Complete synthetic index.
        tmp_path: Container-local synthetic replacement path.
    """
    index_policy_pdf(
        policy_engine,
        SAMPLE_PDF,
        document_id="unrelated",
        label="Other Synthetic Manual",
    )
    unrelated = [
        row for row in _sections(policy_engine) if row["document_id"] == "unrelated"
    ]
    changed = tmp_path / "changed.pdf"
    _write_pdf(
        changed,
        [
            [
                "1. Garantia artificial",
                "Somente a cobertura experimental azul permanece neste cenário.",
            ]
        ],
    )
    summary = index_policy_pdf(policy_engine, changed)
    assert summary.section_count == 1
    assert summary.checksum != hashlib.sha256(SAMPLE_PDF.read_bytes()).hexdigest()
    rows = _sections(policy_engine)
    assert [row for row in rows if row["document_id"] == "unrelated"] == unrelated
    own = [row for row in rows if row["document_id"] == "store-policy-manual"]
    assert len(own) == 1
    assert own[0]["section_number"] == "1"
    assert "azul" in str(own[0]["content"])


def test_database_failure_rolls_back_replacement(
    policy_engine: Engine, tmp_path: Path
) -> None:
    """Restore old sections and metadata after PostgreSQL rejects replacement rows.

    Args:
        policy_engine: Complete synthetic index.
        tmp_path: Container-local synthetic replacement path.
    """
    before = _sections(policy_engine)
    with policy_engine.begin() as connection:
        checksum = connection.scalar(select(PolicyDocument.checksum))
        schema = connection.get_execution_options()["schema_translate_map"][None]
        table = Table(
            "policy_sections", MetaData(), schema=schema, autoload_with=connection
        )
        constraint = CheckConstraint(
            table.c.document_order < len(before), name="synthetic_replacement_limit"
        )
        connection.execute(AddConstraint(constraint))
    changed = tmp_path / "changed.pdf"
    lines = []
    for number in range(1, len(before) + 2):
        lines.extend(
            [
                f"{number}. Artificial section",
                "Invented replacement content for rollback testing.",
            ]
        )
    _write_pdf(changed, [lines])
    with pytest.raises(PolicyIndexError, match="previous index preserved"):
        index_policy_pdf(policy_engine, changed)
    assert _sections(policy_engine) == before
    with policy_engine.connect() as connection:
        assert connection.scalar(select(PolicyDocument.checksum)) == checksum


def test_parse_failure_preserves_previous_index(
    policy_engine: Engine, tmp_path: Path
) -> None:
    """Validate PDF content before changing a previously successful index.

    Args:
        policy_engine: Complete synthetic index.
        tmp_path: Container-local invalid source path.
    """
    before = _sections(policy_engine)
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a PDF")
    with pytest.raises(PolicyExtractionError):
        index_policy_pdf(policy_engine, invalid)
    with pytest.raises(PolicyExtractionError, match="could not be read"):
        index_policy_pdf(policy_engine, tmp_path / "missing.pdf")
    assert _sections(policy_engine) == before


@pytest.mark.parametrize(
    ("query", "section"),
    [
        ("horário de funcionamento", "2"),
        ("formas de pagamento", "3"),
        ("trocas e devoluções", "4"),
        ("garantia", "8"),
        ("What are your opening hours?", "2"),
        ("What payment methods do you accept?", "3"),
        ("installments", "3.1"),
        ("refunds", "4.1"),
        ("returns", "4"),
        ("shipping", "5"),
        ("tracking", "5.3"),
        ("warranty", "8"),
        ("privacy", "9"),
        ("address", "1"),
        ("contact", "7"),
        ("promotions", "6"),
        ("promoção", "6"),
        ("How do returns work?", "4"),
        ("How can I contact support?", "7"),
        ("PIX", "3"),
        ("embalagem lilás constelações", "5.2"),
    ],
)
def test_portuguese_and_english_retrieval(
    policy_engine: Engine, query: str, section: str
) -> None:
    """Rank relevant source sections first for explicit bilingual customer concepts.

    Args:
        policy_engine: Complete synthetic index.
        query: Natural-language Portuguese or supported English query.
        section: Expected most relevant source section identifier.
    """
    result = search_store_policies(policy_engine, query=query)
    assert result.matches
    assert result.matches[0].section == section
    assert all(match.document == "Store Policy Manual" for match in result.matches)


def test_cross_page_provenance_serialization_and_bounded_output(
    policy_engine: Engine,
) -> None:
    """Return original bounded excerpts with source hierarchy and physical page numbers.

    Args:
        policy_engine: Complete synthetic index.
    """
    result = search_store_policies(policy_engine, query="constelações")
    match = result.matches[0]
    assert match.section == "5.2"
    assert match.title == "Entregas distantes"
    assert (match.page_start, match.page_end) == (3, 4)
    assert "constelações" in match.excerpt
    assert len(match.excerpt) <= 1200
    serialized = json.loads(result.model_dump_json())
    assert serialized["matches"][0]["document_id"] == "store-policy-manual"
    assert set(serialized["matches"][0]) == {
        "document_id",
        "document",
        "section",
        "title",
        "page_start",
        "page_end",
        "excerpt",
    }
    assert "/app" not in result.model_dump_json()


def test_long_section_returns_a_bounded_excerpt(
    database_engine: Engine, tmp_path: Path
) -> None:
    """Keep a long coherent source section intact while bounding the public excerpt.

    Args:
        database_engine: Disposable PostgreSQL target.
        tmp_path: Container-local generated synthetic PDF.
    """
    source = tmp_path / "long.pdf"
    line = "Extraordinariamente sintético extraordinariamente ilustrativo."
    _write_pdf(source, [["1. Apêndice experimental", *([line] * 25)], [line] * 25])
    index_policy_pdf(database_engine, source)
    result = search_store_policies(database_engine, query="extraordinariamente")
    assert len(result.matches) == 1
    assert len(result.matches[0].excerpt) <= 1200
    assert len(result.matches[0].excerpt) < len(
        str(_sections(database_engine)[0]["content"])
    )
    assert "extraordinariamente" in result.matches[0].excerpt.casefold()
    assert (result.matches[0].page_start, result.matches[0].page_end) == (1, 2)


@pytest.mark.parametrize(
    ("document_id", "label"),
    [
        ("../private", "Safe label"),
        ("safe", "/app/data/private/manual.pdf"),
        ("safe", ""),
    ],
)
def test_source_metadata_rejects_filesystem_paths(
    database_engine: Engine, document_id: str, label: str
) -> None:
    """Reject unsafe public metadata before creating policy tables or source records.

    Args:
        database_engine: Disposable PostgreSQL engine without tables.
        document_id: Invalid identifier or valid control identifier.
        label: Invalid path/empty label or valid control label.
    """
    with pytest.raises(ValueError):
        index_policy_pdf(
            database_engine, SAMPLE_PDF, document_id=document_id, label=label
        )
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    assert not inspect(database_engine).has_schema(schema)


@pytest.mark.parametrize(
    "query",
    [
        "bitcoin mining",
        "astronaut insurance",
        "!!!",
        "' OR 1=1; DROP TABLE policy_sections; --",
    ],
)
def test_unsupported_queries_return_empty(policy_engine: Engine, query: str) -> None:
    """Return no fabricated substitute for unsupported topics or hostile literal input.

    Args:
        policy_engine: Complete synthetic index.
        query: Unsupported or injection-like query text.
    """
    assert search_store_policies(policy_engine, query=query).matches == ()


@pytest.mark.parametrize("query", ["", " \t\n", "x" * 501, "private\x00query"])
def test_invalid_queries_fail_before_database(
    database_engine: Engine, query: str
) -> None:
    """Reject invalid query text without echoing it or accessing missing tables.

    Args:
        database_engine: Engine without initialized tables.
        query: Invalid public input.
    """
    with pytest.raises(ValueError, match="Policy query"):
        search_store_policies(database_engine, query=query)


@pytest.mark.parametrize("limit", [0, -1, 11, True])
def test_invalid_limits_fail_before_database(
    database_engine: Engine, limit: int
) -> None:
    """Reject unbounded limits and bool values before accessing PostgreSQL.

    Args:
        database_engine: Engine without initialized tables.
        limit: Invalid result bound.
    """
    with pytest.raises(ValueError, match="Policy result limit"):
        search_store_policies(database_engine, query="garantia", limit=limit)


def test_missing_and_empty_indexes_are_distinct_from_no_matches(
    database_engine: Engine,
) -> None:
    """Distinguish missing or empty indexes from a completed index with no matches.

    Args:
        database_engine: Disposable PostgreSQL engine without initial tables.
    """
    with pytest.raises(PolicyIndexMissing, match="not initialized"):
        search_store_policies(database_engine, query="garantia")
    with database_engine.begin() as connection:
        initialize_policy_schema(connection)
    with pytest.raises(PolicyIndexMissing, match="not initialized"):
        search_store_policies(database_engine, query="garantia")


def test_operational_failure_is_sanitized(policy_engine: Engine) -> None:
    """Exercise a real PostgreSQL failure without leaking source paths or query text.

    Args:
        policy_engine: Complete synthetic index.
    """
    failing = create_engine(
        "postgresql+psycopg://",
        connect_args={
            "user": f"missing_policy_role_{uuid4().hex}",
            "connect_timeout": 2,
        },
        hide_parameters=True,
    )
    query = "private-marker@example.test"
    try:
        with pytest.raises(PolicySearchError, match="could not be completed") as raised:
            search_store_policies(failing, query=query)
    finally:
        failing.dispose()
    assert query not in str(raised.value)
    assert "SELECT" not in str(raised.value)


def test_search_is_stable_read_only_and_limited(policy_engine: Engine) -> None:
    """Maintain deterministic rank ties, result bounds, and unchanged indexed content.

    Args:
        policy_engine: Complete synthetic index.
    """
    before = _sections(policy_engine)
    first = search_store_policies(policy_engine, query="shipping", limit=2)
    assert first == search_store_policies(policy_engine, query="shipping", limit=2)
    assert len(first.matches) == 2
    assert _sections(policy_engine) == before


def test_policy_index_does_not_change_operational_records(
    initialized_engine: Engine, source_dir: Path
) -> None:
    """Add policy metadata while retaining every existing operational source record.

    Args:
        initialized_engine: Disposable operational schema from commit 02.
        source_dir: Public synthetic operational CSVs.
    """
    seed_data(initialized_engine, source_dir)
    with initialized_engine.connect() as connection:
        before = {
            name: connection.execute(
                select(table).order_by(*table.primary_key.columns)
            ).all()
            for name, table in Base.metadata.tables.items()
        }
    index_policy_pdf(initialized_engine, SAMPLE_PDF)
    search_store_policies(initialized_engine, query="horário")
    with initialized_engine.connect() as connection:
        after = {
            name: connection.execute(
                select(table).order_by(*table.primary_key.columns)
            ).all()
            for name, table in Base.metadata.tables.items()
        }
    assert after == before


def test_index_cli(database_engine: Engine, capsys: pytest.CaptureFixture[str]) -> None:
    """Run explicit CLI indexing repeatedly with concise source-free completion output.

    Args:
        database_engine: Disposable schema target.
        capsys: Captured command output.
    """
    schema = database_engine.get_execution_options()["schema_translate_map"][None]
    for _ in range(2):
        assert main(["index", "--source", str(SAMPLE_PDF), "--schema", schema]) == 0
    output = capsys.readouterr()
    assert output.out.count("Policy index committed") == 2
    assert str(SAMPLE_PDF) not in output.out
    assert "constelações" not in output.out
    assert not output.err
