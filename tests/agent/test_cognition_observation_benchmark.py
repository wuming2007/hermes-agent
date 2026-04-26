"""Tests for PR12 cognition observation benchmark helpers."""

from __future__ import annotations

from agent.cognition_observation_benchmark import (
    cognition_observation_cases,
    evaluate_cognition_trace,
    evaluate_cognition_traces,
    expected_case_names,
)


def _matching_trace(case):
    return {
        "schema_version": 1,
        "enabled": True,
        "route": {
            "mode": case.expected_mode,
            "retrieval_plan": case.expected_retrieval_plan,
            "verification_plan": case.expected_verification_plan,
            "allow_cheap_model": case.expected_allow_cheap_model,
        },
        "interaction": {
            "dialogue_mode": case.expected_dialogue_mode,
            "answer_density": case.expected_answer_density,
            "stance_reasons": ["test"],
        },
        "uncertainty": {
            "present": case.expected_uncertainty_action is not None,
            "action": case.expected_uncertainty_action,
        },
        "verification": {"ladder_enabled": False},
    }


def test_case_names_are_fixed_and_ordered():
    assert expected_case_names() == (
        "fast_query",
        "project_status",
        "history_lookup",
        "debate",
        "execution",
    )
    assert tuple(case.name for case in cognition_observation_cases()) == expected_case_names()


def test_evaluate_matching_trace_passes():
    case = cognition_observation_cases()[1]
    result = evaluate_cognition_trace(case, _matching_trace(case))

    assert result["name"] == case.name
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["observed"]["mode"] == case.expected_mode
    assert result["observed"]["dialogue_mode"] == case.expected_dialogue_mode


def test_evaluate_mismatched_trace_reports_failures_without_raising():
    case = cognition_observation_cases()[0]
    trace = _matching_trace(case)
    trace["route"]["mode"] = "standard"
    trace["interaction"]["answer_density"] = "expanded"

    result = evaluate_cognition_trace(case, trace)

    assert result["passed"] is False
    assert "route.mode expected 'fast' got 'standard'" in result["failures"]
    assert "interaction.answer_density expected 'brief' got 'expanded'" in result["failures"]


def test_evaluate_missing_or_malformed_trace_does_not_raise():
    case = cognition_observation_cases()[0]

    missing = evaluate_cognition_trace(case, None)
    malformed = evaluate_cognition_trace(case, "not-a-dict")

    assert missing["passed"] is False
    assert malformed["passed"] is False
    assert missing["failures"]
    assert malformed["failures"]


def test_evaluate_cognition_traces_summarizes_all_cases():
    traces = {case.name: _matching_trace(case) for case in cognition_observation_cases()}
    report = evaluate_cognition_traces(traces)

    assert report["total"] == 5
    assert report["passed"] == 5
    assert report["failed"] == 0
    assert all(item["passed"] for item in report["results"])
