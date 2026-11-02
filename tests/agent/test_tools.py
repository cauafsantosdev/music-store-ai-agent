"""Real PostgreSQL tool-adapter and synthetic provenance integration tests."""

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, MetaData, Table, select, update
from sqlalchemy.orm import Session
from sqlalchemy.schema import DropTable

from music_store_agent.agent.demo import initialize_demo, verified_snapshot, verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.tool_registry import build_tools
from music_store_agent.database.models import Customer, Product


def test_six_unique_safe_tool_schemas(demo_engine: Engine) -> None:
    """Expose only six bounded public operations without engine/session arguments.

    Args:
        demo_engine: Confirmed synthetic PostgreSQL dataset.
    """
    tools = build_tools(demo_engine)
    assert {tool.name for tool in tools} == {
        "search_products",
        "get_product_details",
        "get_product_pricing",
        "get_active_promotions",
        "get_order_status",
        "search_store_policies",
    }
    assert len(tools) == 6
    for tool in tools:
        properties = tool.get_input_jsonschema()["properties"]
        assert (
            not {"engine", "session", "credentials", "api_key", "customer_id"}
            & properties.keys()
        )


def test_real_catalog_pricing_and_availability(demo_engine: Engine) -> None:
    """Reuse deterministic catalog filtering, exact prices, and PIX non-cumulativity.

    Args:
        demo_engine: Confirmed synthetic PostgreSQL dataset.
    """
    tools = {tool.name: tool for tool in build_tools(demo_engine)}
    search = tools["search_products"].invoke({})
    assert search["status"] == "ok"
    assert [item["product_id"] for item in search["data"]] == [101]
    price = tools["get_product_pricing"].invoke(
        {"product_id": 101, "payment_method": "pix"}
    )
    assert price["data"]["quote"]["quoted_price"] == "1104.93"
    assert price["data"]["quote"]["pix_price"] is None
    assert price["data"]["quote"]["discount_source"] == "promotion"
    unavailable = tools["get_product_details"].invoke({"product_id": 103})
    assert unavailable["data"]["availability"]["reason"] == "coming_soon"
    assert not unavailable["data"]["purchasable"]
    assert (
        tools["get_product_details"].invoke({"product_id": 999999})["code"]
        == "not_found"
    )
    assert (
        tools["search_products"].invoke({"query": "absent synthetic instrument"})[
            "data"
        ]
        == []
    )
    promotions = tools["get_active_promotions"].invoke({})
    assert [row["product_id"] for row in promotions["data"]] == [101]
    assert json.loads(json.dumps(price))["data"]["quote"]["list_price"] == "1234.56"


@pytest.mark.parametrize(
    "arguments",
    [
        {"limit": True},
        {"limit": 100},
        {"engine": "public"},
        {"max_price_brl": 12.3},
        {"payment_method": "cash"},
    ],
)
def test_invalid_model_arguments_are_sanitized(
    demo_engine: Engine, arguments: dict[str, object]
) -> None:
    """Reject coercions, unsupported payment methods, and hidden dependency injection.

    Args:
        demo_engine: Confirmed synthetic PostgreSQL dataset.
        arguments: Invalid model-generated tool arguments.
    """
    tool = next(
        tool for tool in build_tools(demo_engine) if tool.name == "search_products"
    )
    result = json.loads(tool.invoke(arguments))
    assert result["code"] == "invalid_arguments"
    assert "ValidationError" not in json.dumps(result)


@pytest.mark.parametrize(
    "order_id,email",
    [
        (999999, "customer-203@example.test"),
        (303, "wrong@example.test"),
        (301, "shared@example.test"),
        (302, "SHARED@EXAMPLE.TEST"),
    ],
)
def test_order_failures_remain_uniform(
    demo_engine: Engine, order_id: int, email: str
) -> None:
    """Preserve uniform failures for unknown, mismatched, and ambiguous order lookups.

    Args:
        demo_engine: Confirmed synthetic dataset retaining original ambiguous samples.
        order_id: Requested synthetic identifier.
        email: Incorrect or ambiguous reserved address.
    """
    tool = next(
        tool for tool in build_tools(demo_engine) if tool.name == "get_order_status"
    )
    result = tool.invoke({"order_id": order_id, "customer_email": email})
    assert result["code"] == "order_unverified"
    assert "data" not in result


def test_valid_order_minimizes_data(demo_engine: Engine) -> None:
    """Return only permitted tracking facts for a uniquely associated synthetic order.

    Args:
        demo_engine: Confirmed synthetic PostgreSQL dataset.
    """
    tool = next(
        tool for tool in build_tools(demo_engine) if tool.name == "get_order_status"
    )
    result = tool.invoke(
        {"order_id": 303, "customer_email": "CUSTOMER-203@EXAMPLE.TEST"}
    )
    assert result["status"] == "ok"
    assert set(result["data"]) == {
        "order_id",
        "status",
        "order_date",
        "estimated_delivery",
        "tracking_code",
    }
    assert "example.test" not in json.dumps(result)
    assert result["data"]["tracking_code"] is None
    invalid = json.loads(
        tool.invoke({"order_id": 303, "customer_email": "private@example.org"})
    )
    assert invalid["code"] == "invalid_arguments"
    assert "private@example.org" not in json.dumps(invalid)


def test_real_policy_retrieval_and_empty_results(demo_engine: Engine) -> None:
    """Preserve real source references and bounded excerpts without fabricated answers.

    Args:
        demo_engine: Confirmed synthetic policy index.
    """
    tool = next(
        tool
        for tool in build_tools(demo_engine)
        if tool.name == "search_store_policies"
    )
    result = tool.invoke({"query": "opening hours"})
    match = result["data"]["matches"][0]
    assert match["section"] == "2"
    assert match["page_start"] == 2
    assert len(match["excerpt"]) <= 1200
    assert len(result["data"]["matches"]) <= 3
    missing = tool.invoke({"query": "bitcoin mining"})
    assert missing["data"]["matches"] == []


def test_modified_or_private_records_are_never_forwarded(demo_engine: Engine) -> None:
    """Check actual rows instead of trusting a previously verified namespace name.

    Args:
        demo_engine: Initially confirmed synthetic PostgreSQL dataset.
    """
    tools = {tool.name: tool for tool in build_tools(demo_engine)}
    with demo_engine.begin() as connection:
        connection.execute(
            update(Product)
            .where(Product.product_id == 101)
            .values(description="private-record-marker")
        )
    result = tools["get_product_details"].invoke({"product_id": 101})
    assert result["code"] == "demo_unverified"
    assert "private-record-marker" not in json.dumps(result)
    with pytest.raises(AgentFailure):
        initialize_demo(demo_engine)


def test_missing_policy_index_is_safe(demo_engine: Engine) -> None:
    """Reject a missing policy table without reading another schema or inventing rules.

    Args:
        demo_engine: Initially confirmed synthetic PostgreSQL dataset.
    """
    schema = demo_engine.get_execution_options()["schema_translate_map"][None]
    with demo_engine.begin() as connection:
        table = Table(
            "policy_sections", MetaData(), schema=schema, autoload_with=connection
        )
        connection.execute(DropTable(table))
    tool = next(
        tool
        for tool in build_tools(demo_engine)
        if tool.name == "search_store_policies"
    )
    assert tool.invoke({"query": "returns"})["code"] == "missing_policy"


def test_public_and_uninitialized_sources_fail_closed(database_engine: Engine) -> None:
    """Refuse default/private namespaces and an unproven empty schema.

    Args:
        database_engine: Empty disposable schema.
    """
    public = database_engine.execution_options(schema_translate_map={None: "public"})
    with pytest.raises(AgentFailure) as raised:
        verify_demo(public)
    assert raised.value.code == ErrorCode.DEMO_UNVERIFIED
    with pytest.raises(AgentFailure):
        verify_demo(database_engine)


def test_demo_setup_is_repeatable_and_snapshots_are_read_only(
    demo_engine: Engine,
) -> None:
    """Repeat initialization without duplicates and execute real reads in one snapshot.

    Args:
        demo_engine: Confirmed synthetic PostgreSQL dataset.
    """
    initialize_demo(demo_engine)
    verify_demo(demo_engine)
    with pytest.raises(AgentFailure) as raised:
        with verified_snapshot(demo_engine) as bind, Session(bind) as session:
            session.execute(update(Customer).values(name="must-not-persist"))
    assert raised.value.code == ErrorCode.DATABASE
    verify_demo(demo_engine)
    with demo_engine.connect() as connection:
        assert len(connection.execute(select(Customer.customer_id)).all()) == 3


def test_changed_fixture_bytes_cannot_initialize_demo(
    database_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify pinned public bytes before any source is copied into the selected schema.

    Args:
        database_engine: Empty disposable schema.
        tmp_path: Directory lacking the pinned public fixture assets.
        monkeypatch: Scoped fixture-location override.
    """
    monkeypatch.setattr("music_store_agent.agent.demo.SAMPLE_DIRECTORY", tmp_path)
    with pytest.raises(AgentFailure) as raised:
        initialize_demo(database_engine)
    assert raised.value.code == ErrorCode.DEMO_UNVERIFIED
