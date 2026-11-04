"""Run sanitized pytest-based evaluations inside the API development container.

Commands:
    uv run --no-sync python -m music_store_agent.evaluation.runner
    With RUN_LIVE_GROQ_EVAL=1 and credentials explicitly supplied:
    uv run --no-sync python -m music_store_agent.evaluation.runner --mode live

The offline report measures assertion groups, not LLM reasoning quality. Standard
output contains only summarized JSON. No report or transcript is written to disk.
"""

import argparse
import io
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

import pytest

from music_store_agent.evaluation.cases import CASES, EvaluationCase


def matches(nodeid: str, selector: str) -> bool:
    """Match a collected test, parametrized function or selected test module.

    Args:
        nodeid: Collected pytest identifier, never included in public reports.
        selector: Version-controlled evidence selector.

    Returns:
        Whether the collected test belongs to the selected evidence group.
    """
    return nodeid == selector or nodeid.startswith((selector + "[", selector + "::"))


@dataclass(eq=False)
class EvaluationResults:
    """Collect phase outcomes without retaining assertion values or raw diagnostics."""

    nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    collection_failed: bool = False
    live_report: dict[str, object] | None = None

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Record expected checks so missing execution cannot count as passed.

        Args:
            items: Actual collected checks.
        """
        self.nodes.update((item.nodeid, {}) for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        """Retain only whether collection failed.

        Args:
            report: Pytest collection result; diagnostic contents are ignored.
        """
        self.collection_failed |= report.failed

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Record safe phase outcomes and an explicitly constructed live summary.

        Args:
            report: Test outcome without copying captured output or exception text.
        """
        self.nodes.setdefault(report.nodeid, {})[report.when] = report.outcome
        if report.when == "call":
            for name, value in report.user_properties:
                if name == "live_evaluation" and isinstance(value, dict):
                    self.live_report = value

    def summarize(self, cases: tuple[EvaluationCase, ...] = CASES) -> dict[str, object]:
        """Compute case-level metrics only from fully completed evidence assertions.

        Args:
            cases: Version-controlled deterministic case groups.

        Returns:
            Summarized counts and fixed failure reasons, without test diagnostics.
        """
        failures: list[dict[str, str]] = []
        categories: dict[str, dict[str, int]] = {}
        for case in cases:
            category = categories.setdefault(
                case.category, {"total": 0, "passed": 0, "failed": 0}
            )
            category["total"] += 1
            reason = ""
            for selector in case.checks:
                phases = [
                    value
                    for node, value in self.nodes.items()
                    if matches(node, selector)
                ]
                if (
                    self.collection_failed
                    or not phases
                    or any(
                        set(phase) != {"setup", "call", "teardown"} for phase in phases
                    )
                ):
                    reason = "checks_not_completed"
                    break
                if any(
                    outcome != "passed"
                    for phase in phases
                    for outcome in phase.values()
                ):
                    reason = "deterministic_check_failed_or_skipped"
                    break
            category["failed" if reason else "passed"] += 1
            if reason:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "reason": reason,
                    }
                )
        passed = len(cases) - len(failures)
        return {
            "mode": "offline",
            "basis": "deterministic_pipeline_checks",
            "total": len(cases),
            "passed": passed,
            "failed": len(failures),
            "pass_rate": round(passed / len(cases), 4) if cases else 0,
            "categories": categories,
            "failures": failures,
            "checks_collected": len(self.nodes),
        }


@contextmanager
def evaluation_environment(mode: str) -> Iterator[None]:
    """Disable tracing, unrelated smoke tests and inherited pytest customizations.

    Args:
        mode: Offline checks or explicitly gated live evaluation.

    Yields:
        A scoped environment restored after evaluation.
    """
    names = (
        "GROQ_API_KEY",
        "RUN_LIVE_GROQ",
        "RUN_LIVE_GROQ_EVAL",
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in ("RUN_LIVE_GROQ", "PYTEST_ADDOPTS"):
            os.environ.pop(name, None)
        if mode == "offline":
            os.environ.pop("GROQ_API_KEY", None)
            os.environ.pop("RUN_LIVE_GROQ_EVAL", None)
        os.environ.update(
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            LANGCHAIN_TRACING_V2="false",
            LANGSMITH_TRACING="false",
        )
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(mode: str) -> tuple[dict[str, object], int]:
    """Execute real checks with raw pytest output captured rather than published.

    Args:
        mode: Offline or live mode selected explicitly by the CLI.

    Returns:
        Safe JSON-compatible measurements and a success/failure exit code.
    """
    if mode not in ("offline", "live"):
        return {"mode": "unknown", "status": "not_run", "reason": "invalid_mode"}, 2
    if mode == "live" and (
        os.getenv("RUN_LIVE_GROQ_EVAL") != "1"
        or not os.getenv("GROQ_API_KEY", "").strip()
    ):
        return {
            "mode": "live",
            "status": "not_run",
            "reason": "explicit_flag_and_key_required",
            "outbound_model_requests": 0,
        }, 2
    selectors = list(
        dict.fromkeys(selector for case in CASES for selector in case.checks)
    )
    if mode == "live":
        selectors = ["tests/agent/test_evaluation_live.py::test_live_agent_evaluation"]
    results = EvaluationResults()
    with (
        evaluation_environment(mode),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        try:
            exit_code = int(
                pytest.main(["-p", "no:terminal", *selectors], plugins=[results])
            )
        except Exception:
            # Never render collection/setup exceptions or chained provider payloads.
            exit_code = 2
    if mode == "live":
        if results.live_report is not None:
            return results.live_report, exit_code
        return {
            "mode": "live",
            "status": "not_completed",
            "reason": "evaluation_setup_failed",
        }, 2
    report = results.summarize()
    return report, 0 if exit_code == 0 and report["failed"] == 0 else 1


def main() -> None:
    """Print only measured, sanitized JSON and exit with the evaluation outcome."""
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic pipelines or opted-in Groq behavior."
    )
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    arguments = parser.parse_args()
    report, exit_code = run(arguments.mode)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
