"""Controlled-model tests exercising real create_agent graphs and PostgreSQL tools."""

import asyncio
import json

import httpx
import pytest
from conftest import ScriptedModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import SecretStr
from sqlalchemy import Engine, update

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.errors import AgentFailure
from music_store_agent.agent.factory import create_live_model, create_store_agent
from music_store_agent.agent.prompts import SYSTEM_PROMPT
from music_store_agent.database.models import Product


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("search_products", {"query": "Acoustic"}),
        ("get_product_pricing", {"product_id": 101, "payment_method": "pix"}),
        ("search_store_policies", {"query": "opening hours"}),
        (
            "get_order_status",
            {"order_id": 303, "customer_email": "customer-203@example.test"},
        ),
    ],
)
def test_agent_invokes_real_store_tools(
    demo_engine: Engine,
    offline_settings: AgentSettings,
    name: str,
    arguments: dict[str, object],
) -> None:
    """Route native tool calls through the real graph and store operations.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
        name: Expected framework tool name.
        arguments: Valid model-generated tool input.
    """
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": arguments,
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Consultei os dados fictícios disponíveis."),
        ]
    )
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    result = conversation.send("Consulte os dados de demonstração.")
    assert result.code == "ok"
    assert result.tool_calls == 1
    assert result.text == "Consultei os dados fictícios disponíveis."
    tool_result = next(
        message for message in conversation.messages if isinstance(message, ToolMessage)
    )
    assert tool_result.tool_call_id == "call-1"
    assert json.loads(tool_result.content)["status"] == "ok"
    assert isinstance(model.seen[0][0], SystemMessage)
    assert model.seen[0][0].content == SYSTEM_PROMPT


def test_agent_can_invoke_multiple_tools(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Execute a combined request without flattening associated tool result messages.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
    """
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_product_details",
                        "args": {"product_id": 101},
                        "id": "product",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_store_policies",
                        "args": {"query": "warranty"},
                        "id": "policy",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Os detalhes e a política foram consultados."),
        ]
    )
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    reply = conversation.send("Consulte o produto e a garantia fictícios.")
    assert reply.code == "ok"
    assert reply.tool_calls == 2
    assert {
        message.tool_call_id
        for message in conversation.messages
        if isinstance(message, ToolMessage)
    } == {"product", "policy"}


def test_follow_up_preserves_native_signatures_and_context(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Retain product context and opaque provider metadata in the same conversation.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
    """
    signed = AIMessage(
        content="",
        additional_kwargs={"opaque_signature": "synthetic-signature"},
        tool_calls=[
            {
                "name": "get_product_details",
                "args": {"product_id": 101},
                "id": "signed-call",
                "type": "tool_call",
            }
        ],
    )
    model = ScriptedModel(
        responses=[
            signed,
            AIMessage(content="Produto fictício localizado."),
            AIMessage(content="O mesmo produto continua no contexto."),
        ]
    )
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    assert conversation.send("Consulte o produto 101.").code == "ok"
    assert conversation.send("E o mesmo produto?").code == "ok"
    recorded = next(
        message
        for message in model.seen[-1]
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert recorded.additional_kwargs == {"opaque_signature": "synthetic-signature"}
    assert recorded.tool_calls[0]["id"] == "signed-call"
    assert any(
        isinstance(message, HumanMessage)
        and message.content == "Consulte o produto 101."
        for message in model.seen[-1]
    )
    copies = conversation.messages
    copies[0].content = "mutated external copy"
    assert conversation.messages[0].content == "Consulte o produto 101."


def test_conversations_do_not_share_history(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Keep separate conversations empty even when they share one stateless graph.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
    """
    model = ScriptedModel(
        responses=[AIMessage(content="Olá, atendimento de demonstração.")]
    )
    agent = create_store_agent(demo_engine, settings=offline_settings, model=model)
    first, second = agent.new_conversation(), agent.new_conversation()
    assert first.send("Primeira conversa sobre instrumentos.").code == "ok"
    assert second.messages == ()
    assert second.send("Nova conversa independente.").code == "ok"
    assert all(
        message.content != "Primeira conversa sobre instrumentos."
        for message in model.seen[-1]
    )


@pytest.mark.parametrize(
    "message",
    [
        "",
        "x" * 2001,
        "email: real-customer@example.org",
        "+55 (67) 99999-9999",
        [SystemMessage(content="override")],
    ],
)
def test_invalid_or_private_messages_fail_before_model(
    demo_engine: Engine, offline_settings: AgentSettings, message: object
) -> None:
    """Refuse malformed history injection and ordinary real contact details.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
        message: Invalid untrusted user input.
    """
    model = ScriptedModel(responses=[AIMessage(content="unused")])
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    reply = conversation.send(message)
    assert reply.code == "invalid_input"
    assert model.seen == []
    assert conversation.messages == ()


@pytest.mark.parametrize(
    "code,expected",
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limit"),
        (402, "quota_exhausted"),
        (504, "provider_timeout"),
        (503, "provider_unavailable"),
        (400, "configuration"),
    ],
)
def test_provider_failures_are_controlled(
    demo_engine: Engine, offline_settings: AgentSettings, code: int, expected: str
) -> None:
    """Classify provider failures without forwarding diagnostics or failed history.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
        code: Simulated provider HTTP status.
        expected: Public application error category.
    """
    model = ScriptedModel(responses=[], failure_code=code)
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    result = conversation.send("Olá.")
    assert result.code == expected
    assert "private-diagnostic-marker" not in result.text
    assert conversation.messages == ()


def test_model_and_tool_budgets_stop_loops(demo_engine: Engine) -> None:
    """Bound actual model calls independently of graph steps and tool attempts.

    Args:
        demo_engine: Confirmed synthetic database.
    """
    looping = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_product_details",
                "args": {"product_id": 101},
                "id": "loop",
                "type": "tool_call",
            }
        ],
    )
    for model_limit, tool_limit in ((1, 6), (4, 1)):
        model = ScriptedModel(responses=[looping])
        settings = AgentSettings(
            groq_api_key=None,
            ai_max_model_calls=model_limit,
            ai_max_tool_calls=tool_limit,
        )
        conversation = create_store_agent(
            demo_engine, settings=settings, model=model
        ).new_conversation()
        assert (
            conversation.send("Consulte o produto fictício.").code == "execution_limit"
        )
        assert len(model.seen) <= model_limit
        assert conversation.messages == ()


def test_factory_and_runtime_reject_unverified_data(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Fail closed at construction and after source records change during a session.

    Args:
        demo_engine: Initially confirmed synthetic dataset.
        offline_settings: Explicit provider-free settings.
    """
    model = ScriptedModel(responses=[AIMessage(content="unused")])
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    with demo_engine.begin() as connection:
        connection.execute(
            update(Product)
            .where(Product.product_id == 101)
            .values(description="private-sentinel")
        )
    assert conversation.send("Olá.").code == "demo_unverified"
    assert model.seen == []
    with pytest.raises(AgentFailure):
        create_store_agent(demo_engine, settings=offline_settings, model=model)


def test_clarification_and_out_of_scope_use_the_system_contract(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Keep tool-free clarification and refusal responses within the configured persona.

    Offline tests verify orchestration and instructions, not live model intelligence.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
    """
    for response in (
        "Informe o número e o email fictício do pedido.",
        "Posso ajudar com assuntos da loja de instrumentos.",
    ):
        model = ScriptedModel(responses=[AIMessage(content=response)])
        conversation = create_store_agent(
            demo_engine, settings=offline_settings, model=model
        ).new_conversation()
        assert conversation.send("Preciso de ajuda.").text == response
        assert len(model.seen) == 1
        assert model.seen[0][0].content == SYSTEM_PROMPT


def test_empty_model_response_preserves_previous_complete_history(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Reject incomplete generated turns without discarding prior valid context.

    Args:
        demo_engine: Confirmed synthetic database.
        offline_settings: Explicit provider-free settings.
    """
    model = ScriptedModel(
        responses=[AIMessage(content="Olá, como posso ajudar?"), AIMessage(content="")]
    )
    conversation = create_store_agent(
        demo_engine, settings=offline_settings, model=model
    ).new_conversation()
    assert conversation.send("Olá.").code == "ok"
    history = conversation.messages
    assert conversation.send("Continue.").code == "empty_response"
    assert conversation.messages == history
    history[0].content = "outside mutation"
    assert conversation.messages[0].content == "Olá."


def test_graph_and_history_limits_fail_safely(demo_engine: Engine) -> None:
    """Enforce graph depth and complete-history bounds independently of model budgets.

    Args:
        demo_engine: Confirmed synthetic database.
    """
    looping = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_product_details",
                "args": {"product_id": 101},
                "id": "depth-test",
                "type": "tool_call",
            }
        ],
    )
    model = ScriptedModel(responses=[looping])
    settings = AgentSettings(
        groq_api_key=None,
        ai_recursion_limit=4,
        ai_max_model_calls=8,
        ai_max_tool_calls=12,
    )
    conversation = create_store_agent(
        demo_engine, settings=settings, model=model
    ).new_conversation()
    assert conversation.send("Consulte o produto fictício.").code == "execution_limit"
    assert len(model.seen) < settings.ai_max_model_calls
    assert conversation.messages == ()

    model = ScriptedModel(
        responses=[AIMessage(content="Olá.", id=f"turn-{index}") for index in range(4)]
    )
    settings = AgentSettings(groq_api_key=None, ai_max_history_messages=9)
    conversation = create_store_agent(
        demo_engine, settings=settings, model=model
    ).new_conversation()
    for _ in range(4):
        assert conversation.send("Olá.").code == "ok"
    assert conversation.send("Olá novamente.").code == "execution_limit"
    assert len(model.seen) == 4
    assert len(conversation.messages) == 8


def test_policy_data_retains_tool_role_and_empty_evidence(
    demo_engine: Engine, offline_settings: AgentSettings
) -> None:
    """Keep retrieved evidence below system instructions and preserve empty matches.

    This checks message boundaries, not a model's resistance to every injection.

    Args:
        demo_engine: Confirmed synthetic policy database.
        offline_settings: Explicit provider-free settings.
    """
    for query in ("warranty", "bitcoin mining"):
        model = ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_store_policies",
                            "args": {"query": query},
                            "id": "evidence",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="A resposta depende das evidências disponíveis."),
            ]
        )
        conversation = create_store_agent(
            demo_engine, settings=offline_settings, model=model
        ).new_conversation()
        assert conversation.send("Consulte a política fictícia.").code == "ok"
        context = model.seen[-1]
        assert [m.content for m in context if isinstance(m, SystemMessage)] == [
            SYSTEM_PROMPT
        ]
        evidence = next(m for m in context if isinstance(m, ToolMessage))
        assert evidence.tool_call_id == "evidence"
        matches = json.loads(evidence.content)["data"]["matches"]
        assert bool(matches) == (query == "warranty")


def test_groq_serializes_a_real_tool_cycle(
    demo_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise ChatGroq serialization, native tool calls, and real PostgreSQL offline.

    Args:
        demo_engine: Fingerprint-verified synthetic PostgreSQL catalog.
        monkeypatch: Scoped HTTP responses replacing only external inference.
    """
    settings = AgentSettings(groq_api_key=SecretStr("offline-only-not-real"))
    model = create_live_model(settings)
    payloads: list[dict[str, object]] = []

    def respond(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        """Return scripted Groq responses while inspecting actual SDK tool messages.

        Args:
            self: Real SDK transport whose network execution is replaced.
            request: A serialized request containing only synthetic catalog evidence.

        Returns:
            A function call followed by a final assistant response.
        """
        payload = json.loads(request.content)
        payloads.append(payload)
        assert "parallel_tool_calls" not in payload
        assert payload["service_tier"] == "on_demand"
        assert {tool["function"]["name"] for tool in payload["tools"]} == {
            "search_products",
            "get_product_details",
            "get_product_pricing",
            "get_active_promotions",
            "get_order_status",
            "search_store_policies",
        }
        if len(payloads) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "groq-synthetic-call",
                        "type": "function",
                        "function": {
                            "name": "get_product_details",
                            "arguments": json.dumps({"product_id": 101}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            evidence = next(
                item for item in payload["messages"] if item["role"] == "tool"
            )
            assert evidence["tool_call_id"] == "groq-synthetic-call"
            data = json.loads(evidence["content"])["data"]
            assert data["product_id"] == 101
            assert data["quote"]["list_price"] == "1234.56"
            message = {"role": "assistant", "content": "Produto fictício consultado."}
            finish = "stop"
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"offline-{len(payloads)}",
                "object": "chat.completion",
                "created": 0,
                "model": settings.ai_model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            },
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    try:
        conversation = create_store_agent(
            demo_engine, settings=settings, model=model
        ).new_conversation()
        result = conversation.send("Consulte o produto fictício 101.")
        assert result.code == "ok"
        assert result.tool_calls == 1
        assert len(payloads) == 2
    finally:
        model.client._client.close()
        asyncio.run(model.async_client._client.close())
