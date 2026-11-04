"""Explicit dataset selection and optional approved-source PostgreSQL integration."""

import csv
import json
import os
from pathlib import Path
from shutil import copytree

import pytest
from conftest import ScriptedModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy import Engine, func, inspect, select, update
from sqlalchemy.orm import Session

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.datasets import CHALLENGE_CHECKSUMS
from music_store_agent.agent.demo import (
    initialize_demo,
    select_dataset,
    verified_snapshot,
    verify_demo,
)
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.agent.service import _validate_message
from music_store_agent.agent.tool_registry import build_tools
from music_store_agent.api.main import create_app
from music_store_agent.database.models import Base, Customer, Order, Product
from music_store_agent.database.policies import PolicyDocument, PolicySection
from music_store_agent.tools import products


def selected_settings(source: Path) -> AgentSettings:
    """Build explicit offline challenge settings without external credentials.

    Args:
        source: Trusted mounted source directory or an intentional missing fixture.

    Returns:
        Mode-specific configuration with independent schema defaults.
    """
    return AgentSettings(
        ai_dataset_mode="challenge",
        ai_challenge_directory=source,
        groq_api_key=None,
        ai_demo_schema="",
        ai_state_schema="",
    )


@pytest.fixture
def challenge_source() -> Path:
    """Require actual local approved originals; public CI never fabricates them.

    Returns:
        The explicit read-only mounted directory.
    """
    source = Path(os.getenv("AI_CHALLENGE_DIRECTORY", "/run/music-store/challenge"))
    if not source.is_dir():
        pytest.skip("Approved challenge sources are not mounted.")
    return source


@pytest.fixture
def challenge_engine(database_engine: Engine, challenge_source: Path) -> Engine:
    """Import all six original CSVs and the original PDF into a disposable schema.

    Args:
        database_engine: Empty generated namespace with automatic cleanup.
        challenge_source: Actual approved local source mount.

    Returns:
        Real PostgreSQL engine with challenge provenance verified.
    """
    engine = select_dataset(database_engine, selected_settings(challenge_source))
    initialize_demo(engine)
    return engine


def test_default_mode_is_public_and_challenge_is_explicit() -> None:
    """Preserve public defaults and choose independent challenge schemas explicitly."""
    public = AgentSettings(groq_api_key=None, ai_demo_schema="", ai_state_schema="")
    challenge = selected_settings(Path("/missing"))
    assert public.ai_dataset_mode == "sample"
    assert public.ai_demo_schema == "agent_demo"
    assert public.ai_state_schema == "agent_state"
    assert challenge.ai_demo_schema == "agent_demo_challenge"
    assert challenge.ai_state_schema == "agent_state_challenge"
    with pytest.raises(ValidationError):
        AgentSettings(ai_dataset_mode="unknown")
    with pytest.raises(ValidationError):
        AgentSettings(ai_dataset_mode="challenge", ai_demo_schema="agent_demo")
    with pytest.raises(ValidationError):
        AgentSettings(ai_dataset_mode="sample", ai_demo_schema="agent_demo_challenge")


def test_missing_sources_fail_before_schema_creation(
    database_engine: Engine, tmp_path: Path
) -> None:
    """Refuse missing originals without importing public substitutes or creating tables.

    Args:
        database_engine: Empty generated PostgreSQL namespace.
        tmp_path: Missing local source root.
    """
    engine = select_dataset(database_engine, selected_settings(tmp_path))
    for operation in (initialize_demo, verify_demo):
        with pytest.raises(AgentFailure) as failure:
            operation(engine)
        assert failure.value.code == ErrorCode.DEMO_UNVERIFIED
    assert not inspect(engine).has_schema(
        engine.get_execution_options()["schema_translate_map"][None]
    )


def test_mode_cannot_relabel_sample_records(
    demo_engine: Engine, tmp_path: Path
) -> None:
    """Reject changing selection as a substitute for provenance or source verification.

    Args:
        demo_engine: Original pinned public dataset.
        tmp_path: Unapproved source root.
    """
    with pytest.raises(AgentFailure):
        verify_demo(select_dataset(demo_engine, selected_settings(tmp_path)))
    verify_demo(demo_engine)


def test_capabilities_cannot_cross_dataset_boundaries(
    state_store: ConversationStore,
) -> None:
    """Reject cross-mode resume even when the same checkpoint schema is selected.

    Args:
        state_store: Existing real sample state store with isolated cleanup.
    """
    identifier, token = state_store.create()
    with ConversationStore(state_store.schema, dataset_mode="challenge") as challenge:
        challenge.check_ready()
        with pytest.raises(AgentFailure) as failure:
            with challenge.authorized(identifier, token):
                pytest.fail("Cross-dataset authorization succeeded.")
        assert failure.value.code == ErrorCode.SESSION_UNVERIFIED
        other_id, other_token = challenge.create()
        with pytest.raises(AgentFailure):
            with state_store.authorized(other_id, other_token):
                pytest.fail("Cross-dataset authorization succeeded.")
        with challenge.authorized(other_id, other_token):
            pass
    with state_store.authorized(identifier, token):
        pass


def test_challenge_counts_pdf_and_repeatable_initialization(
    challenge_engine: Engine,
) -> None:
    """Verify complete original records, indexed pages and non-destructive repetition.

    Args:
        challenge_engine: Real approved original records in an isolated namespace.
    """
    expected = {
        "products": 65,
        "categories": 9,
        "promotions": 25,
        "customers": 50,
        "orders": 20,
        "order_items": 22,
    }
    with challenge_engine.connect() as connection:
        actual = {
            name: connection.scalar(select(func.count()).select_from(table))
            for name, table in Base.metadata.tables.items()
        }
        assert actual == expected
        assert (
            connection.scalar(select(PolicyDocument.checksum))
            == CHALLENGE_CHECKSUMS["políticas_da_loja.pdf"]
        )
        assert connection.scalar(select(func.count()).select_from(PolicySection)) == 25
        assert connection.scalar(select(func.max(PolicySection.page_end))) == 8
        assert "checkpoints" not in inspect(connection).get_table_names(
            schema=challenge_engine.get_execution_options()["schema_translate_map"][
                None
            ]
        )
    initialize_demo(challenge_engine)
    verify_demo(challenge_engine)


def test_all_original_products_and_pricing_are_queryable(
    challenge_engine: Engine,
) -> None:
    """Use real catalog and bounded adapters while preserving availability rules.

    Args:
        challenge_engine: Approved original catalog with real pricing operations.
    """
    with verified_snapshot(challenge_engine) as bind, Session(bind) as session:
        identifiers = list(session.scalars(select(Product.product_id)))
        assert len(identifiers) == 65
        for identifier in identifiers:
            assert products.get_product_details(bind, identifier) is not None
        results = products.search_products(bind, available_only=False, limit=100)
        assert len(results) == 65
        assert all(item.product_id in identifiers for item in results)
    tools = {tool.name: tool for tool in build_tools(challenge_engine)}
    assert len(tools) == 6
    search = tools["search_products"].invoke({"query": "violão", "limit": 10})
    assert search["status"] == "ok"
    identifier = search["data"][0]["product_id"]
    assert (
        tools["get_product_pricing"].invoke(
            {"product_id": identifier, "payment_method": "pix"}
        )["status"]
        == "ok"
    )
    assert tools["get_active_promotions"].invoke({"limit": 5})["status"] == "ok"


def test_original_orders_and_message_privacy(
    challenge_engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    """Allow approved synthetic emails while retaining exact association and privacy.

    Args:
        challenge_engine: Real original synthetic customers and orders.
        caplog: Captured application logs without customer values.
    """
    with Session(challenge_engine) as session:
        order, email = session.execute(
            select(Order.order_id, Customer.email).join(Customer)
        ).first()
    assert email.rsplit("@", 1)[-1] != "example.test"
    tool = {tool.name: tool for tool in build_tools(challenge_engine)}[
        "get_order_status"
    ]
    result = tool.invoke({"order_id": order, "customer_email": email})
    assert result["status"] == "ok"
    assert set(result["data"]) == {
        "order_id",
        "status",
        "order_date",
        "estimated_delivery",
        "tracking_code",
    }
    mismatch = tool.invoke(
        {"order_id": order, "customer_email": "incorrect@example.test"}
    )
    unknown = tool.invoke(
        {"order_id": 2_147_483_647, "customer_email": "incorrect@example.test"}
    )
    assert mismatch == unknown
    assert mismatch["code"] == ErrorCode.ORDER_UNVERIFIED
    _validate_message(f"Consulte o pedido {order} com {email}", challenge_engine)
    with pytest.raises(AgentFailure):
        _validate_message("Consulte unknown@unapproved.invalid", challenge_engine)
    with pytest.raises(AgentFailure):
        _validate_message(f"Consulte {email}")
    assert email not in json.dumps(result) + caplog.text


def test_original_pdf_retrieval(challenge_engine: Engine) -> None:
    """Retrieve bounded original passages with physical page references.

    Args:
        challenge_engine: Real PostgreSQL index of the complete approved PDF.
    """
    tool = {tool.name: tool for tool in build_tools(challenge_engine)}[
        "search_store_policies"
    ]
    result = tool.invoke({"query": "formas de pagamento PIX", "limit": 3})
    assert result["status"] == "ok"
    excerpts = result["data"]["matches"]
    assert 1 <= len(excerpts) <= 3
    assert any("PIX" in item["excerpt"] for item in excerpts)
    assert all(1 <= item["page_start"] <= item["page_end"] <= 8 for item in excerpts)


@pytest.mark.parametrize("filename", tuple(CHALLENGE_CHECKSUMS))
def test_changed_original_source_fails_closed(
    challenge_engine: Engine, challenge_source: Path, tmp_path: Path, filename: str
) -> None:
    """Reject every modified approved source even when database records are unchanged.

    Args:
        challenge_engine: Originally verified database records.
        challenge_source: Read-only original mount, never modified by the test.
        tmp_path: Disposable mutable source copy.
        filename: One of the seven pinned original filenames.
    """
    copied = Path(copytree(challenge_source, tmp_path / "sources"))
    with (copied / filename).open("ab") as handle:
        handle.write(b"\nmodified")
    engine = select_dataset(challenge_engine, selected_settings(copied))
    for operation in (initialize_demo, verify_demo):
        with pytest.raises(AgentFailure) as failure:
            operation(engine)
        assert failure.value.code == ErrorCode.DEMO_UNVERIFIED
    verify_demo(challenge_engine)


@pytest.mark.parametrize("policy", [False, True])
def test_modified_database_cannot_be_reapproved(
    challenge_engine: Engine, policy: bool
) -> None:
    """Refuse changed operational records or indexed policy text without overwriting.

    Args:
        challenge_engine: Real approved challenge namespace.
        policy: Whether to change indexed text instead of a product.
    """
    with challenge_engine.begin() as connection:
        if policy:
            connection.execute(
                update(PolicySection).values(content="Unapproved policy")
            )
        else:
            connection.execute(update(Product).values(stock_quantity=99999))
    for operation in (verify_demo, initialize_demo):
        with pytest.raises(AgentFailure) as failure:
            operation(challenge_engine)
        assert failure.value.code == ErrorCode.DEMO_UNVERIFIED


def test_sample_and_challenge_are_isolated(
    demo_engine: Engine, challenge_source: Path
) -> None:
    """Reject importing originals over populated sample tables and retain their pin.

    Args:
        demo_engine: Approved existing sample records.
        challenge_source: Original approved local sources.
    """
    engine = select_dataset(demo_engine, selected_settings(challenge_source))
    with pytest.raises(AgentFailure):
        initialize_demo(engine)
    verify_demo(demo_engine)


@pytest.mark.parametrize("transport", ["messages", "stream"])
def test_challenge_agent_and_api_preserve_checkpoints(
    challenge_engine: Engine,
    challenge_source: Path,
    state_namespace: str,
    transport: str,
) -> None:
    """Exercise original orders through real LangGraph, API and native checkpoints.

    Args:
        challenge_engine: Approved original operational namespace.
        challenge_source: Explicit approved original mount.
        state_namespace: Generated state namespace with automatic test cleanup.
        transport: Ordinary JSON reply or existing completed-response SSE delivery.
    """
    with Session(challenge_engine) as session:
        order, email = session.execute(
            select(Order.order_id, Customer.email).join(Customer)
        ).first()
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_status",
                        "args": {"order_id": order, "customer_email": email},
                        "id": "challenge-order",
                    }
                ],
            ),
            AIMessage(content="Pedido sintético consultado com segurança."),
        ]
    )
    settings = selected_settings(challenge_source)
    with ConversationStore(state_namespace, dataset_mode="challenge") as store:
        store.initialize()
        with TestClient(
            create_app(
                settings=settings, model=model, engine=challenge_engine, store=store
            )
        ) as client:
            assert client.get("/ready").status_code == 200
            response = client.post("/api/v1/conversations")
            assert response.status_code == 201
            data = response.json()
            identifier = data["conversation_id"]
            headers = {"Authorization": "Bearer " + data["access_token"]}
            sent = client.post(
                f"/api/v1/conversations/{identifier}/{transport}",
                headers=headers,
                json={"message": f"Consulte o pedido {order} com {email}"},
            )
            assert sent.status_code == 200
            if transport == "stream":
                assert "event: done" in sent.text
            assert len(model.seen) == 2
            history = client.get(f"/api/v1/conversations/{identifier}", headers=headers)
            assert history.status_code == 200
            assert len(model.seen) == 2
            assert len(history.json()["messages"]) == 2
            rejected = client.post(
                f"/api/v1/conversations/{identifier}/{transport}",
                headers=headers,
                json={"message": "Consulte unknown@unapproved.invalid"},
            )
            assert rejected.status_code == 422
            assert len(model.seen) == 2
        with pytest.raises(AgentFailure):
            create_store_agent(
                challenge_engine,
                settings=AgentSettings(groq_api_key=None),
                model=model,
                state_store=store,
            )
    with ConversationStore(state_namespace, dataset_mode="challenge") as fresh:
        with create_store_agent(
            challenge_engine, settings=settings, model=model, state_store=fresh
        ) as agent:
            resumed = agent.resume_conversation(identifier, data["access_token"])
            assert (
                len(
                    agent.get_conversation_history(
                        resumed.conversation_id, resumed.access_token
                    )
                )
                == 2
            )
            assert len(model.seen) == 2


def test_original_filenames_are_the_manifest_contract(challenge_source: Path) -> None:
    """Recognize all six original CSV filenames without shipping their records.

    Args:
        challenge_source: Original read-only source directory.
    """
    sources = challenge_source
    assert len([name for name in CHALLENGE_CHECKSUMS if name.endswith(".csv")]) == 6
    for name in CHALLENGE_CHECKSUMS:
        assert (sources / name).is_file()
        if name.endswith(".csv"):
            with (sources / name).open(encoding="utf-8-sig", newline="") as handle:
                assert list(csv.DictReader(handle))
