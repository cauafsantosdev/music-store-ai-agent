"""Offline checks for measured reporting, gating and sanitized failure summaries."""

import json
import sys
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from music_store_agent.evaluation.cases import CASES, LIVE_CASE_IDS
from music_store_agent.evaluation.live import MAX_REQUESTS, RequestBudget, assess
from music_store_agent.evaluation.runner import (
    EvaluationResults,
    evaluation_environment,
    main,
    run,
)


def test_case_contracts_have_explicit_criteria_and_all_six_tools() -> None:
    """Keep the case table grounded, unique and ready for separate live examples."""
    assert len({case.case_id for case in CASES}) == len(CASES)
    assert {step.name for case in CASES for step in case.steps} == {
        "search_products",
        "get_product_details",
        "get_product_pricing",
        "get_active_promotions",
        "get_order_status",
        "search_store_policies",
    }
    for case in CASES:
        assert (
            case.turns
            and case.expected_behavior
            and case.criteria
            and case.evidence
            and case.checks
        )
    assert len(LIVE_CASE_IDS) == 5 and len(LIVE_CASE_IDS) * 3 == MAX_REQUESTS == 15
    budget = RequestBudget()
    for _ in range(MAX_REQUESTS):
        budget.take()
    with pytest.raises(RuntimeError):
        budget.take()
    assert budget.sent == 15


@pytest.mark.parametrize(
    "phase,outcome", [("call", "failed"), ("setup", "skipped"), ("teardown", "failed")]
)
def test_report_never_counts_failed_or_skipped_evidence_as_passed(
    phase: str, outcome: str
) -> None:
    """Compute case metrics from all evidence phases, including parametrized checks.

    Args:
        phase: A lifecycle phase that failed or skipped.
        outcome: The controlled phase result.
    """
    case = replace(CASES[0], checks=("synthetic::test_case",))
    success = {"setup": "passed", "call": "passed", "teardown": "passed"}
    results = EvaluationResults(
        nodes={
            "synthetic::test_case[one]": success,
            "synthetic::test_case[two]": {**success, phase: outcome},
        }
    )
    report = results.summarize((case,))
    assert report["total"] == report["failed"] == 1 and report["passed"] == 0
    assert report["pass_rate"] == 0
    assert report["categories"]["catalog"] == {"total": 1, "passed": 0, "failed": 1}
    assert report["failures"][0]["case_id"] == "CAT-001"
    assert "synthetic::" not in json.dumps(report)


def test_missing_collection_and_partial_execution_are_not_successes() -> None:
    """Fail closed on missing selectors, collection errors and interrupted teardown."""
    case = replace(CASES[0], checks=("synthetic::test_case",))
    for results in (
        EvaluationResults(),
        EvaluationResults(collection_failed=True),
        EvaluationResults(
            nodes={"synthetic::test_case": {"setup": "passed", "call": "passed"}}
        ),
    ):
        assert results.summarize((case,))["failed"] == 1
    complete = EvaluationResults(
        nodes={
            "synthetic::test_case": {
                "setup": "passed",
                "call": "passed",
                "teardown": "passed",
            }
        }
    )
    assert complete.summarize((case,))["pass_rate"] == 1


def test_offline_environment_removes_credentials_and_restores_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent inherited smoke/evaluation flags or tracing from enabling offline calls.

    Args:
        monkeypatch: Scoped synthetic inherited environment.
    """
    import os

    for name in (
        "GROQ_API_KEY",
        "RUN_LIVE_GROQ",
        "RUN_LIVE_GROQ_EVAL",
        "LANGSMITH_TRACING",
        "PYTEST_ADDOPTS",
    ):
        monkeypatch.setenv(name, "1")
    with evaluation_environment("offline"):
        assert "GROQ_API_KEY" not in os.environ
        assert (
            "RUN_LIVE_GROQ" not in os.environ and "RUN_LIVE_GROQ_EVAL" not in os.environ
        )
        assert "PYTEST_ADDOPTS" not in os.environ
        assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["GROQ_API_KEY"] == os.environ["RUN_LIVE_GROQ"] == "1"


def test_live_runner_requires_distinct_flag_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the original smoke opt-in from authorizing the new live evaluation.

    Args:
        monkeypatch: Scoped flags without constructing a provider client.
    """
    monkeypatch.setenv("RUN_LIVE_GROQ", "1")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic-unused-key")
    monkeypatch.delenv("RUN_LIVE_GROQ_EVAL", raising=False)
    report, code = run("live")
    assert (
        code == 2
        and report["status"] == "not_run"
        and report["outbound_model_requests"] == 0
    )
    monkeypatch.setenv("RUN_LIVE_GROQ_EVAL", "1")
    monkeypatch.delenv("GROQ_API_KEY")
    report, code = run("live")
    assert code == 2 and report["outbound_model_requests"] == 0


def test_runner_suppresses_raw_collection_errors_and_preserves_safe_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Publish JSON failure criteria without raw pytest output or exception details.

    Args:
        monkeypatch: Scoped failing pytest invocation and CLI arguments.
        capsys: Captured public terminal output.
    """

    def fail_collection(*args: object, **kwargs: object) -> int:
        """Simulate a failed collection containing synthetic private diagnostics.

        Args:
            args: Uninspected pytest positional inputs.
            kwargs: Uninspected pytest keyword inputs.

        Raises:
            RuntimeError: Always, with a value excluded from the public report.
        """
        print("private-collection-canary")
        print("private-collection-canary", file=sys.stderr)
        raise RuntimeError("private-collection-canary")

    monkeypatch.setattr(
        "music_store_agent.evaluation.runner.pytest.main", fail_collection
    )
    monkeypatch.setattr(sys, "argv", ["evaluation"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["passed"] == 0 and report["failed"] == len(CASES)
    assert "private-collection-canary" not in output.out + output.err


def test_live_predicates_reject_stacking_and_restricted_source_values() -> None:
    """Evaluate meaningful numeric and privacy failures without judging LLM prose."""
    case = next(case for case in CASES if case.case_id == "PRICE-001")
    messages = (
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_product_pricing",
                    "args": {"product_id": 101, "payment_method": "pix"},
                    "id": "pricing",
                    "type": "tool_call",
                },
                {
                    "name": "get_active_promotions",
                    "args": {},
                    "id": "promotions",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "ok",
                    "data": {
                        "quote": {
                            "quoted_price": "1104.93",
                            "pix_price": None,
                            "payment_method": "pix",
                        }
                    },
                }
            ),
            tool_call_id="pricing",
        ),
        ToolMessage(
            content=json.dumps({"status": "ok", "data": []}), tool_call_id="promotions"
        ),
    )
    valid = assess(case, messages, "Preço verificado: R$ 1.104,93.")
    assert all(valid.values())
    wrong = assess(
        case,
        messages,
        "R$ 1.104,93, com PIX adicional R$ 1.049,68. Synthetic identity",
        ("Synthetic identity",),
    )
    assert not wrong["stacked_discount_claim_absent"]
    assert not wrong["restricted_canaries_absent"]
    assert not assess(case, (), "Preço verificado: R$ 1.104,93.")[
        "required_tool_selected"
    ]
    malformed = (ToolMessage(content="Malformed tool output", tool_call_id="invalid"),)
    assert not assess(case, malformed, "Sem evidências.")[
        "structured_tool_evidence_valid"
    ]
