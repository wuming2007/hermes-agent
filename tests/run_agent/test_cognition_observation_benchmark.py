"""Runtime PR12 cognition observation benchmark tests.

These tests keep model calls fake while verifying that the production turn path
emits cognition_trace and that PR8 trajectory persistence can be analyzed by the
PR9 offline report.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.cognition_observation_benchmark import (
    cognition_observation_cases,
    evaluate_cognition_trace,
)
from agent.cognition_trace_report import analyze_cognition_trace_jsonl
from agent.trajectory import save_trajectory
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


_FAST_COGNITION_CFG = {
    "enabled": True,
    "fast_mode": {
        "max_chars": 160,
        "max_words": 28,
        "allow_urls": False,
        "allow_code_blocks": False,
    },
    "deep_mode_triggers": {
        "historical_questions": True,
        "code_changes": True,
        "risky_external_actions": True,
        "architecture_decisions": True,
    },
    "consistency_guard": {"enabled": True, "deep_mode_only": True},
}


def _mock_response(content="ok", finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None, reasoning=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _setup_agent(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._cognition_config = _FAST_COGNITION_CFG


def _run_one_turn(agent, message: str):
    agent.client.chat.completions.create.return_value = _mock_response()
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


def test_observation_cases_produce_expected_runtime_cognition_traces(agent):
    _setup_agent(agent)

    for case in cognition_observation_cases():
        result = _run_one_turn(agent, case.prompt)
        trace = result["cognition_trace"]
        evaluation = evaluate_cognition_trace(case, trace)
        assert evaluation["passed"], evaluation["failures"]
        assert agent._current_turn_cognition_metadata["cognition_trace"] == trace


def test_observation_trace_persists_to_jsonl_and_report_counts_it(agent, tmp_path):
    _setup_agent(agent)
    case = cognition_observation_cases()[1]  # project_status
    result = _run_one_turn(agent, case.prompt)
    trace = result["cognition_trace"]
    assert evaluate_cognition_trace(case, trace)["passed"] is True

    path = tmp_path / "trajectory_samples.jsonl"
    save_trajectory(
        [{"from": "human", "value": case.prompt}, {"from": "gpt", "value": "ok"}],
        model="test-model",
        completed=True,
        filename=str(path),
        metadata={"cognition_trace": trace},
    )

    report = analyze_cognition_trace_jsonl([path])
    assert report["total_entries"] == 1
    assert report["cognition_trace"]["present"] == 1
    assert report["route"]["modes"] == {case.expected_mode: 1}
    assert report["interaction"]["dialogue_modes"] == {case.expected_dialogue_mode: 1}
    assert report["interaction"]["answer_densities"] == {case.expected_answer_density: 1}
