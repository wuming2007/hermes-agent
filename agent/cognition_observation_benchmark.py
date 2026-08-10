"""Fixed cognition observation benchmark cases (PR12).

This module is intentionally pure: no IO, no model/tool calls, and no runtime
side effects. It provides stable prompt cases and JSON-friendly evaluation
helpers so tests and future offline tooling can detect cognition metadata drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CognitionObservationCase:
    """One fixed cognition observation scenario."""

    name: str
    prompt: str
    expected_mode: str
    expected_dialogue_mode: str
    expected_answer_density: str
    expected_retrieval_plan: str | None = None
    expected_verification_plan: str | None = None
    expected_allow_cheap_model: bool | None = None
    expected_uncertainty_action: str | None = None


_CASES: tuple[CognitionObservationCase, ...] = (
    CognitionObservationCase(
        name="fast_query",
        prompt="what time is it in tokyo?",
        expected_mode="fast",
        expected_dialogue_mode="query",
        expected_answer_density="brief",
        expected_retrieval_plan="principles_only",
        expected_verification_plan="none",
        expected_allow_cheap_model=True,
        expected_uncertainty_action="answer",
    ),
    CognitionObservationCase(
        name="project_status",
        prompt="目前 cognition stack 狀態如何？",
        expected_mode="standard",
        expected_dialogue_mode="status",
        expected_answer_density="brief",
        expected_retrieval_plan="principles_plus_semantic",
        expected_verification_plan="light",
        expected_allow_cheap_model=False,
        expected_uncertainty_action="answer",
    ),
    CognitionObservationCase(
        name="history_lookup",
        prompt="上次我們討論的 root cause 是什麼？",
        expected_mode="deep",
        expected_dialogue_mode="query",
        expected_answer_density="standard",
        expected_retrieval_plan="principles_plus_semantic_plus_episodic",
        expected_verification_plan="full",
        expected_allow_cheap_model=False,
        expected_uncertainty_action="answer",
    ),
    CognitionObservationCase(
        name="debate",
        prompt="你覺得這個架構取捨應不應該反駁？",
        expected_mode="deep",
        expected_dialogue_mode="debate",
        expected_answer_density="standard",
        expected_retrieval_plan="principles_plus_semantic_plus_episodic",
        expected_verification_plan="full",
        expected_allow_cheap_model=False,
        expected_uncertainty_action="answer",
    ),
    CognitionObservationCase(
        name="execution",
        prompt="幫我做 PR12，跑測試後 commit",
        expected_mode="deep",
        expected_dialogue_mode="execution",
        expected_answer_density="brief",
        expected_retrieval_plan="principles_plus_semantic_plus_episodic",
        expected_verification_plan="full",
        expected_allow_cheap_model=False,
        expected_uncertainty_action="answer",
    ),
)


def cognition_observation_cases() -> tuple[CognitionObservationCase, ...]:
    """Return the fixed PR12 observation prompt set in stable order."""

    return _CASES


def expected_case_names() -> tuple[str, ...]:
    """Return fixed case names in benchmark order."""

    return tuple(case.name for case in _CASES)


def _nested_mapping(trace: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = trace.get(key)
    return value if isinstance(value, Mapping) else {}


def _append_failure(
    failures: list[str],
    path: str,
    expected: Any,
    observed: Any,
) -> None:
    if expected is None:
        return
    if observed != expected:
        failures.append(f"{path} expected {expected!r} got {observed!r}")


def evaluate_cognition_trace(
    case: CognitionObservationCase,
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one cognition trace against one fixed benchmark case.

    The result is JSON-friendly and never raises for missing/malformed traces.
    """

    failures: list[str] = []
    observed: dict[str, Any] = {}

    if not isinstance(trace, Mapping):
        return {
            "name": case.name,
            "passed": False,
            "failures": ["trace missing or malformed"],
            "observed": observed,
        }

    route = _nested_mapping(trace, "route")
    interaction = _nested_mapping(trace, "interaction")
    uncertainty = _nested_mapping(trace, "uncertainty")

    observed = {
        "enabled": trace.get("enabled"),
        "schema_version": trace.get("schema_version"),
        "mode": route.get("mode"),
        "retrieval_plan": route.get("retrieval_plan"),
        "verification_plan": route.get("verification_plan"),
        "allow_cheap_model": route.get("allow_cheap_model"),
        "dialogue_mode": interaction.get("dialogue_mode"),
        "answer_density": interaction.get("answer_density"),
        "uncertainty_action": uncertainty.get("action"),
    }

    _append_failure(failures, "trace.enabled", True, observed["enabled"])
    _append_failure(failures, "trace.schema_version", 1, observed["schema_version"])
    _append_failure(failures, "route.mode", case.expected_mode, observed["mode"])
    _append_failure(
        failures,
        "route.retrieval_plan",
        case.expected_retrieval_plan,
        observed["retrieval_plan"],
    )
    _append_failure(
        failures,
        "route.verification_plan",
        case.expected_verification_plan,
        observed["verification_plan"],
    )
    _append_failure(
        failures,
        "route.allow_cheap_model",
        case.expected_allow_cheap_model,
        observed["allow_cheap_model"],
    )
    _append_failure(
        failures,
        "interaction.dialogue_mode",
        case.expected_dialogue_mode,
        observed["dialogue_mode"],
    )
    _append_failure(
        failures,
        "interaction.answer_density",
        case.expected_answer_density,
        observed["answer_density"],
    )
    _append_failure(
        failures,
        "uncertainty.action",
        case.expected_uncertainty_action,
        observed["uncertainty_action"],
    )

    return {
        "name": case.name,
        "passed": not failures,
        "failures": failures,
        "observed": observed,
    }


def evaluate_cognition_traces(
    traces_by_case: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Evaluate all fixed benchmark cases against a name -> trace mapping."""

    results = [
        evaluate_cognition_trace(case, traces_by_case.get(case.name))
        for case in _CASES
    ]
    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }
