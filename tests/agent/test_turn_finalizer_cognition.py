"""Behavior contract: the PR3 consistency guard's revision is the answer that
becomes durable, and the guard runs at most once per turn.

Since #81641 the ordinary text-response exit flushes the final assistant row
before ``finalize_turn`` runs, and SQLite treats a non-blank assistant row as
settled. The guard therefore revises at that seam (``finish_text_response``,
ahead of the output transform), and ``finalize_turn`` only catches exits that
never reached it (budget summaries, recovery breaks) — before its persist step.
The text-exit half is pinned on a real turn in ``tests/agent/test_run_agent.py``
(``TestConsistencyGuardWiring``); these tests pin the finalizer half.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

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


def _finalize(agent, *, turn_id="turn-1"):
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
        final_response="candidate",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id=turn_id,
        user_message="上次的設計回顧",
        original_user_message="上次的設計回顧",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
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
