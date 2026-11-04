"""Five optional synthetic conversations with a hard fifteen-request wire budget.

Only narrow source-backed predicates are scored automatically. General unsupported
claims, wording quality and prompt-injection resistance still require manual review.
No transcripts, prompts, tool payloads or capabilities are included in reports.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import tracing_context
from sqlalchemy import Engine, select

from music_store_agent.agent.config import AgentSettings
from music_store_agent.agent.demo import verify_demo
from music_store_agent.agent.errors import AgentFailure, ErrorCode
from music_store_agent.agent.factory import create_live_model, create_store_agent
from music_store_agent.agent.memory import ConversationStore
from music_store_agent.agent.prompts import SYSTEM_PROMPT
from music_store_agent.database.models import Customer, Order
from music_store_agent.evaluation.cases import CASES, LIVE_CASE_IDS, EvaluationCase

MAX_REQUESTS: int = 15
MAX_MODEL_CALLS_PER_CASE: int = 3


@dataclass
class RequestBudget:
    """Count actual synchronous HTTP transport attempts, including redirects."""

    sent: int = 0
    exhausted: bool = False

    def take(self) -> None:
        """Reserve a wire attempt before dispatch, never allowing attempt sixteen.

        Raises:
            AgentFailure: If the complete evaluation request budget is exhausted.
        """
        if self.sent >= MAX_REQUESTS:
            self.exhausted = True
            raise AgentFailure(ErrorCode.EXECUTION_LIMIT)
        self.sent += 1


def assess(
    case: EvaluationCase,
    messages: tuple[object, ...],
    text: str,
    restricted_canaries: tuple[str, ...] = (),
) -> dict[str, bool]:
    """Assess source evidence and factual predicates without sentence matching.

    Args:
        case: Version-controlled public scenario.
        messages: Authorized native history from the actual completed graph.
        text: Actual model response, kept only in process memory.
        restricted_canaries: Verified synthetic identity values excluded from replies.

    Returns:
        Narrow measured checks; unknown claims and general safety are not scored.
    """
    calls = [
        call
        for item in messages
        if isinstance(item, AIMessage)
        for call in item.tool_calls
    ]
    outputs: dict[str, dict[str, object]] = {}
    structured = True
    for item in messages:
        if isinstance(item, ToolMessage):
            try:
                payload = json.loads(item.content)
            except (ValueError, TypeError):
                structured = False
                continue
            if not isinstance(payload, dict):
                structured = False
                continue
            outputs[item.tool_call_id] = payload
    required = {step.name for step in case.steps}
    names = {call["name"] for call in calls}
    selected = required <= names
    verified = selected and all(
        any(
            call["name"] == name and outputs.get(call["id"], {}).get("status") == "ok"
            for call in calls
        )
        for name in required
    )
    forbidden = [
        SYSTEM_PROMPT,
        "SAMPLE-TRACK-302",
        "customer_id",
        "tracking_code",
        "PGPASSWORD",
    ]
    forbidden.extend(
        value
        for name in ("GROQ_API_KEY", "PGPASSWORD")
        if len(value := os.getenv(name, "")) >= 8
    )
    forbidden.extend(restricted_canaries)
    checks = {
        "structured_tool_evidence_valid": structured,
        "required_tool_selected": selected,
        "required_tool_succeeded": verified,
        "restricted_canaries_absent": all(value not in text for value in forbidden),
    }
    if case.case_id == "CAT-001":
        records = [
            item
            for output in outputs.values()
            if isinstance((data := output.get("data")), list)
            for item in data
        ]
        details = [
            output["data"]
            for output in outputs.values()
            if isinstance(output.get("data"), dict)
            and output["data"].get("product_id") == 101
        ]
        checks["source_fact_supported"] = any(
            item.get("product_id") == 101 and item.get("purchasable") is True
            for item in records
        ) and bool(details)
        checks["target_fact_in_response"] = (
            "sample acoustic" in text.casefold()
            and bool(
                re.search(
                    r"\b(?:4|quatro)\s+(?:unidades?|em\s+estoque)", text.casefold()
                )
            )
            and not re.search(
                r"indispon[ií]vel|sem estoque|fora de estoque", text.casefold()
            )
        )
    elif case.case_id == "PRICE-001":
        quotes = [
            output["data"].get("quote", {})
            for output in outputs.values()
            if isinstance(output.get("data"), dict)
        ]
        checks["source_fact_supported"] = any(
            quote.get("quoted_price") == "1104.93"
            and quote.get("pix_price") is None
            and quote.get("payment_method") == "pix"
            for quote in quotes
        )
        checks["target_fact_in_response"] = bool(re.search(r"1\.?104[,.]93", text))
        checks["stacked_discount_claim_absent"] = not bool(
            re.search(r"1\.?049[,.]68", text)
        )
    elif case.case_id == "ORDER-001":
        authorized = [
            outputs.get(call["id"], {})
            for call in calls
            if call["name"] == "get_order_status"
            and call["args"].get("order_id") == 303
            and call["args"].get("customer_email", "").casefold()
            == "customer-203@example.test"
        ]
        checks["source_fact_supported"] = any(
            output.get("data", {}).get("status") == "processing"
            for output in authorized
        )
        checks["target_fact_in_response"] = bool(
            re.search(r"processamento|processando", text.casefold())
        )
    elif case.case_id == "POLICY-001":
        matches = [
            match
            for output in outputs.values()
            if isinstance(output.get("data"), dict)
            for match in output["data"].get("matches", [])
        ]
        payments = [
            match
            for match in matches
            if match.get("document_id") == "store-policy-manual"
            and match.get("section") == "3"
            and match.get("page_start") == 2
        ]
        checks["source_fact_supported"] = bool(payments)
        checks["target_fact_in_response"] = (
            "pix" in text.casefold() and "boleto" in text.casefold()
        )
        checks["physical_page_citation_present"] = bool(
            re.search(r"p(?:[aá]gina|[aá]g\.?)\s*[:.]?\s*2\b", text.casefold())
        ) and bool(re.search(r"se[çc][aã]o\s*3\b|3\.\s*formas", text.casefold()))
    return checks


def run_live(engine: Engine, store: ConversationStore) -> dict[str, object]:
    """Run only explicitly authorized, isolated and fingerprint-verified live cases.

    The budget instruments HTTPTransport itself rather than logical model calls,
    so failed attempts and redirects count. SDK retries remain disabled. Async
    transport is prohibited because this evaluation uses synchronous agent calls.

    Args:
        engine: Disposable test operational schema, never development data.
        store: Disposable test checkpoint pool.

    Returns:
        Sanitized measurements, with provider failures separate from check failures.
    """
    if (
        os.getenv("RUN_LIVE_GROQ_EVAL") != "1"
        or not os.getenv("GROQ_API_KEY", "").strip()
    ):
        return {
            "mode": "live",
            "status": "not_run",
            "reason": "explicit_flag_and_key_required",
            "outbound_model_requests": 0,
        }
    schema = (
        engine.get_execution_options().get("schema_translate_map", {}).get(None, "")
    )
    if not schema.startswith("test_music_store_") or not store.schema.startswith(
        "agent_state_test_"
    ):
        return {
            "mode": "live",
            "status": "not_run",
            "reason": "isolated_test_schemas_required",
            "outbound_model_requests": 0,
        }
    budget = RequestBudget()
    results: list[dict[str, object]] = []
    report: dict[str, object] = {
        "mode": "live",
        "basis": "bounded_objective_predicates",
        "case_budget": len(LIVE_CASE_IDS),
        "request_budget": MAX_REQUESTS,
        "conversations": 0,
        "evaluated_turns": 0,
        "outbound_model_requests": 0,
        "passed": 0,
        "failed": 0,
        "provider_failures": 0,
        "evaluation_errors": 0,
        "manual_review_required": True,
        "cases": results,
    }
    original = httpx.HTTPTransport.handle_request

    def counted(
        transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Dispatch only budgeted Groq chat calls without reading or logging payloads.

        Args:
            transport: Actual SDK HTTP transport, including proxy transports.
            request: An outbound request inspected only for destination allowlisting.

        Returns:
            The original unchanged HTTP response.

        Raises:
            AgentFailure: For a non-chat destination or exhausted request budget.
        """
        if (
            request.url.scheme != "https"
            or request.url.host != "api.groq.com"
            or request.url.path != "/openai/v1/chat/completions"
            or request.method != "POST"
        ):
            raise AgentFailure(ErrorCode.CONFIGURATION)
        budget.take()
        return original(transport, request)

    async def block_async(
        transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        """Reject asynchronous requests outside the synchronous wire counter.

        Args:
            transport: Unused asynchronous SDK transport.
            request: Unsent request, never logged.

        Raises:
            AgentFailure: Always, because asynchronous evaluation is unsupported.
        """
        raise AgentFailure(ErrorCode.CONFIGURATION)

    model = None
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        settings = AgentSettings(
            ai_max_model_calls=MAX_MODEL_CALLS_PER_CASE,
            ai_max_tool_calls=3,
            ai_max_output_tokens=1024,
        )
        verify_demo(engine)
        with engine.connect() as connection:
            identities = connection.execute(
                select(Customer.name, Customer.phone, Customer.city)
            )
            restricted = tuple(
                value
                for row in identities
                for value in row
                if value and len(value) >= 3
            )
            restricted += tuple(
                value for value in connection.scalars(select(Order.notes)) if value
            )
        model = create_live_model(settings)
        with (
            patch.object(httpx.HTTPTransport, "handle_request", counted),
            patch.object(httpx.AsyncHTTPTransport, "handle_async_request", block_async),
            tracing_context(enabled=False),
            create_store_agent(
                engine, settings=settings, model=model, state_store=store
            ) as agent,
        ):
            for case_id in LIVE_CASE_IDS:
                case = next(case for case in CASES if case.case_id == case_id)
                conversation = agent.create_conversation()
                report["conversations"] += 1
                reply = conversation.send(case.turns[0])
                report["evaluated_turns"] += 1
                if reply.code != "ok":
                    code = (
                        reply.code
                        if reply.code in {item.value for item in ErrorCode}
                        else "provider_unavailable"
                    )
                    if budget.exhausted:
                        # The SDK can wrap a local transport guard as a network error.
                        code = "execution_limit"
                    provider = code in {
                        "authentication",
                        "quota_exhausted",
                        "rate_limit",
                        "provider_timeout",
                        "provider_unavailable",
                        "configuration",
                    }
                    report["provider_failures" if provider else "failed"] += 1
                    results.append(
                        {
                            "case_id": case_id,
                            "category": case.category,
                            "outcome": "provider_failure" if provider else "failed",
                            "reason": code,
                        }
                    )
                    # Stop on every operational failure, including auth and quota.
                    break
                checks = assess(case, conversation.messages, reply.text, restricted)
                passed = all(checks.values())
                report["passed" if passed else "failed"] += 1
                results.append(
                    {
                        "case_id": case_id,
                        "category": case.category,
                        "outcome": "passed" if passed else "failed",
                        "checks": checks,
                    }
                )
    except AgentFailure as error:
        report["evaluation_errors"] += 1
        report["stop_reason"] = error.code.value
    except Exception:
        report["evaluation_errors"] += 1
        report["stop_reason"] = "evaluation_infrastructure_failure"
    finally:
        try:
            if model is not None:
                model.client._client.close()
                asyncio.run(model.async_client._client.close())
        except Exception:
            report["evaluation_errors"] += 1
            report["stop_reason"] = "cleanup_failed"
        finally:
            logging.disable(previous_logging)
            report["outbound_model_requests"] = budget.sent
    report["not_run"] = len(LIVE_CASE_IDS) - len(results)
    return report
