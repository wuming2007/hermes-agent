"""Behavior contract: the PR3 consistency guard's revision is the answer that
becomes durable, and the guard runs at most once per turn.

Since #81641 the ordinary text-response exit flushes the final assistant row
before ``finalize_turn`` runs, and SQLite treats a non-blank assistant row as
settled. The guard therefore revises at that seam (``finish_text_response``,
ahead of the output transform), and ``finalize_turn`` only catches exits that
never reached it (budget summaries, recovery breaks, an answer recovered from
the stream buffer) — before its persist step.
The text-exit half is pinned on a real turn in ``tests/agent/test_run_agent.py``
(``TestConsistencyGuardWiring``); these tests pin the finalizer half.
"""

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.cognitive_router import CognitiveRoute
from agent.consistency_guard import VerificationResult
from agent.turn_finalizer import apply_cognition_consistency_guard, finalize_turn


def _deep_route() -> CognitiveRoute:
    return CognitiveRoute(
        mode="deep",
        retrieval_plan="principles_plus_semantic_plus_episodic",
        verification_plan="full",
        allow_cheap_model=False,
        consistency_check=True,
        routing_reasons=["historical_question"],
    )


class _CognitionAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        for key in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
        ):
            setattr(self, f"session_{key}", 0)
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict[str, Any]] | None = None
        self._current_cognitive_route = _deep_route()
        self._current_turn_cognition_metadata: dict[str, Any] = {"mode": "deep"}

    def _default_consistency_verifier(self, prompt):
        raise AssertionError("verifier is patched out")

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(message) for message in messages]

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _revised(**overrides):
    fields = dict(
        applied=True, plan="full", original_response="candidate",
        final_response="REVISED", changed=True, notes=("issue:x",),
    )
    fields.update(overrides)
    return VerificationResult(**fields)


def _finalize(
    agent, *, turn_id="turn-1", final_response="candidate", messages=None,
    interrupted=False, failed=False, exit_reason="fallback_prior_turn_content",
):
    if messages is None:
        messages = [
            {"role": "user", "content": "上次的設計回顧"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [{"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
        ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=2,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id=turn_id,
        user_message="上次的設計回顧",
        original_user_message="上次的設計回顧",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


def test_recovery_exit_revision_is_what_gets_persisted(monkeypatch):
    """An exit that never reached the text-response seam is guarded before persist."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CognitionAgent()

    with patch("agent.turn_finalizer.run_full_consistency_check", return_value=_revised()) as full:
        result = _finalize(agent)

    full.assert_called_once()
    assert result["final_response"] == "REVISED"
    assert agent.persisted_messages[-1]["role"] == "assistant"
    assert agent.persisted_messages[-1]["content"] == result["final_response"]
    assert result["cognition_metadata"]["verification_changed"] is True
    assert result["cognition_trace"]["route"]["mode"] == "deep"


def test_revision_of_a_promoted_reasoning_row_lands_in_its_replay_sidecar():
    """A reasoning-only clean stop keeps ``content`` empty and replays ``api_content``
    (#111761); a finalizer-side revision must update that sidecar, not invent content."""
    agent = _CognitionAgent()
    messages = [
        {"role": "user", "content": "上次的設計回顧"},
        {"role": "assistant", "content": "", "api_content": "candidate", "reasoning": "candidate"},
    ]

    with patch("agent.turn_finalizer.run_full_consistency_check", return_value=_revised()):
        revised = apply_cognition_consistency_guard(
            agent, "candidate", turn_id="turn-1", user_message="上次的設計回顧",
            logger=SimpleNamespace(warning=lambda *a, **k: None), messages=messages,
        )

    assert revised == "REVISED"
    assert messages[-1]["api_content"] == "REVISED"
    assert messages[-1]["content"] == ""


def test_guard_already_run_this_turn_is_not_repeated(monkeypatch):
    """The text-response seam's guard pass counts for the whole turn."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CognitionAgent()

    with patch("agent.turn_finalizer.run_full_consistency_check", return_value=_revised()) as full:
        guarded = apply_cognition_consistency_guard(
            agent, "candidate", turn_id="turn-1", user_message="上次的設計回顧",
            logger=SimpleNamespace(warning=lambda *a, **k: None),
        )
        _finalize(agent, turn_id="turn-1")
        _finalize(agent, turn_id="turn-2")

    assert guarded == "REVISED"
    # Once for turn-1 (the seam), once for turn-2 (the finalizer fallback).
    assert full.call_count == 2


# ── Answers recovered from the stream buffer (#95514) ──────────────────────
# An empty terminal completion is replaced by the text the stream already
# delivered. That text never reached the text-response seam, so the
# finalizer's guard pass is the only one it gets.

_STREAMED = "STREAM_CANDIDATE"


def _stream_recovery_agent():
    agent = _CognitionAgent()
    agent._current_streamed_assistant_text = _STREAMED
    return agent


def test_stream_recovered_answer_is_guarded_before_it_is_stored(monkeypatch):
    """The recovered text is guarded once, and the guarded answer is what the caller, the
    transcript, the result trace and the trace the trajectory export reads all carry."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _stream_recovery_agent()
    traces_at_trajectory_save = []
    agent._save_trajectory = lambda *_a, **_kw: traces_at_trajectory_save.append(
        copy.deepcopy(agent._current_turn_cognition_metadata.get("cognition_trace"))
    )
    revised = "Revised first point. Revised second point."

    with patch(
        "agent.turn_finalizer.run_full_consistency_check",
        return_value=_revised(original_response=_STREAMED, final_response=revised),
    ) as full:
        result = _finalize(agent, final_response=None)

    full.assert_called_once()
    assert full.call_args.kwargs["candidate_response"] == _STREAMED
    assert result["final_response"] == revised
    assert agent.persisted_messages[-1]["role"] == "assistant"
    assert agent.persisted_messages[-1]["content"] == revised
    assert result["cognition_metadata"]["verification_changed"] is True
    for trace in (result["cognition_trace"], traces_at_trajectory_save[0]):
        assert trace["verification"]["applied"] is True
        assert trace["verification"]["changed"] is True
        # The process monitor read the revised answer (two sentences), not the streamed one.
        assert trace["process_monitor"]["claim_count"] == 2


def _real_agent(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "logs").mkdir(parents=True)
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "web_search", "description": "web_search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_stream_recovered_revision_replaces_the_blank_row_in_state_db(monkeypatch, tmp_path):
    """#95514 on a real store: the incremental flush already wrote a blank assistant row.
    The reloaded session must hold exactly the answer the caller got — guarded, then
    transformed — in that row: no blank row left behind and no second answer row."""
    from hermes_state import SessionDB

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kw: [f"[t] {kw['response_text']}"] if name == "transform_llm_output" else [],
    )
    agent = _real_agent(tmp_path)
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="sess-stream", source="tui", model="test/model")
    agent._session_db, agent._session_db_created, agent.session_id = db, True, "sess-stream"
    agent._last_flushed_db_idx, agent._flushed_db_message_ids = 0, set()
    agent._current_cognitive_route = _deep_route()
    agent._current_turn_cognition_metadata = {"mode": "deep"}
    agent._current_streamed_assistant_text = _STREAMED
    messages = [
        {"role": "user", "content": "上次的設計回顧"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "ok"},
        {"role": "assistant", "content": ""},
    ]
    agent._flush_messages_to_session_db(messages)

    with patch(
        "agent.turn_finalizer.run_full_consistency_check",
        return_value=_revised(original_response=_STREAMED),
    ) as full:
        result = finalize_turn(
            agent,
            final_response="",
            api_call_count=2,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn-1",
            user_message="上次的設計回顧",
            original_user_message="上次的設計回顧",
            _should_review_memory=False,
            _turn_exit_reason="text_response(final)",
        )

    reopened = SessionDB(db_path=db_path)
    try:
        reloaded = reopened.get_messages_as_conversation("sess-stream")
    finally:
        reopened.close()
        db.close()
    full.assert_called_once()
    assert result["final_response"] == "[t] REVISED"
    assistants = [m for m in reloaded if m.get("role") == "assistant"]
    assert [m.get("content") or "" for m in assistants] == ["", "[t] REVISED"]
    assert assistants[0].get("tool_calls")


def test_text_exit_keeps_its_seam_guarded_answer_over_the_raw_stream_buffer(monkeypatch):
    """A streamed text turn was guarded at the text-response seam while the stream buffer
    still holds the raw model text: the finalizer neither guards again nor trades the
    settled revision for that buffer."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CognitionAgent()
    agent._current_streamed_assistant_text = "candidate"

    with patch("agent.turn_finalizer.run_full_consistency_check", return_value=_revised()) as full:
        seam_answer = apply_cognition_consistency_guard(
            agent, "candidate", turn_id="turn-1", user_message="上次的設計回顧",
            logger=SimpleNamespace(warning=lambda *a, **k: None),
        )
        result = _finalize(
            agent,
            final_response=seam_answer,
            messages=[
                {"role": "user", "content": "上次的設計回顧"},
                {"role": "assistant", "content": seam_answer},
            ],
            exit_reason="text_response(finish_reason=stop)",
        )

    assert full.call_count == 1
    assert result["final_response"] == "REVISED"
    assert [m["content"] for m in agent.persisted_messages if m["role"] == "assistant"] == ["REVISED"]


def test_stream_recovery_with_cognition_disabled_is_not_guarded(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _stream_recovery_agent()
    agent._current_cognitive_route = None
    agent._current_turn_cognition_metadata = {"mode": "disabled"}

    with (
        patch("agent.turn_finalizer.run_light_consistency_check") as light,
        patch("agent.turn_finalizer.run_full_consistency_check") as full,
    ):
        result = _finalize(agent, final_response=None)

    light.assert_not_called()
    full.assert_not_called()
    assert result["final_response"] == _STREAMED
    assert agent.persisted_messages[-1]["content"] == _STREAMED
    assert result["cognition_trace"]["enabled"] is False
    assert "verification_changed" not in result["cognition_metadata"]


@pytest.mark.parametrize("interrupted, failed", [(True, False), (False, True)])
def test_interrupted_or_failed_turn_neither_recovers_nor_guards_the_stream_buffer(
    monkeypatch, interrupted, failed,
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _stream_recovery_agent()

    with patch("agent.turn_finalizer.run_full_consistency_check") as full:
        result = _finalize(agent, final_response=None, interrupted=interrupted, failed=failed)

    full.assert_not_called()
    assert result["final_response"] is None
    assert not any(m.get("content") == _STREAMED for m in agent.persisted_messages)


def test_guard_error_on_a_stream_recovered_answer_keeps_that_answer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _stream_recovery_agent()

    with patch(
        "agent.turn_finalizer.run_full_consistency_check", side_effect=RuntimeError("verifier down"),
    ) as full:
        result = _finalize(agent, final_response=None)

    full.assert_called_once()
    assert result["final_response"] == _STREAMED
    assert agent.persisted_messages[-1]["content"] == _STREAMED
    assert "cleanup_errors" not in result
