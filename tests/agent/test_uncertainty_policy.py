"""Tests for uncertainty-aware cognition policy (PR5 Task 1)."""

from __future__ import annotations

from agent.cognitive_router import CognitiveRoute
from agent.uncertainty_policy import UncertaintyDecision, resolve_uncertainty_decision


_BASE_CFG = {
    "enabled": True,
    "uncertainty_policy": {
        "enabled": True,
        "escalate_fast_on_mismatch": True,
        "escalate_standard_on_history_or_architecture": True,
        "require_tool_evidence_for_risky_actions": True,
        "seek_human_for_unverified_external_actions": True,
    },
}


def _route(mode: str) -> CognitiveRoute:
    if mode == "fast":
        return CognitiveRoute(
            mode="fast",
            retrieval_plan="principles_only",
            verification_plan="none",
            allow_cheap_model=True,
            consistency_check=False,
            routing_reasons=["short_simple"],
        )
    if mode == "standard":
        return CognitiveRoute(
            mode="standard",
            retrieval_plan="principles_plus_semantic",
            verification_plan="light",
            allow_cheap_model=False,
            consistency_check=False,
            routing_reasons=["over_max_chars"],
        )
    return CognitiveRoute(
        mode="deep",
        retrieval_plan="principles_plus_semantic_plus_episodic",
        verification_plan="full",
        allow_cheap_model=False,
        consistency_check=True,
        routing_reasons=["historical:上次"],
    )


def _resolve(message: str, route: CognitiveRoute | None = None, cfg=None, state=None):
    return resolve_uncertainty_decision(
        user_message=message,
        cognition_route=_route("fast") if route is None else route,
        routing_config=_BASE_CFG if cfg is None else cfg,
        agent_state=state,
    )


def test_disabled_cognition_returns_none():
    cfg = {**_BASE_CFG, "enabled": False}
    assert _resolve("hi", cfg=cfg) is None
    assert resolve_uncertainty_decision(
        user_message="hi",
        cognition_route=_route("fast"),
        routing_config=None,
    ) is None


def test_disabled_uncertainty_policy_returns_none():
    cfg = {**_BASE_CFG, "uncertainty_policy": {"enabled": False}}
    assert _resolve("hi", cfg=cfg) is None


def test_missing_route_returns_none():
    assert resolve_uncertainty_decision(
        user_message="hi",
        cognition_route=None,
        routing_config=_BASE_CFG,
    ) is None


def test_fast_low_risk_short_prompt_answers_with_high_confidence():
    decision = _resolve("ping")
    assert isinstance(decision, UncertaintyDecision)
    assert decision.confidence_band == "high"
    assert decision.action == "answer"
    assert decision.escalate_depth is False
    assert decision.require_tool_evidence is False
    assert decision.seek_human is False
    assert decision.target_mode is None
    assert "low_risk_fast_route" in decision.reasons


def test_fast_route_with_historical_cue_escalates_to_standard():
    decision = _resolve("上次我們討論的方向是什麼？", route=_route("fast"))
    assert decision is not None
    assert decision.confidence_band == "medium"
    assert decision.action == "escalate_depth"
    assert decision.escalate_depth is True
    assert decision.target_mode == "standard"
    assert "historical_cue" in decision.reasons


def test_standard_route_with_architecture_cue_escalates_to_deep():
    decision = _resolve("Let's revisit the architecture design", route=_route("standard"))
    assert decision is not None
    assert decision.action == "escalate_depth"
    assert decision.target_mode == "deep"
    assert "architecture_cue" in decision.reasons


def test_standard_risky_external_action_requires_tool_evidence_and_seek_human():
    decision = _resolve("please send the email now", route=_route("standard"))
    assert decision is not None
    assert decision.confidence_band == "low"
    assert decision.action == "seek_human"
    assert decision.escalate_depth is False
    assert decision.require_tool_evidence is True
    assert decision.seek_human is True
    assert decision.target_mode is None
    assert "risky_external_action" in decision.reasons


def test_risky_external_action_can_seek_tool_without_human_gate():
    cfg = {
        **_BASE_CFG,
        "uncertainty_policy": {
            **_BASE_CFG["uncertainty_policy"],
            "seek_human_for_unverified_external_actions": False,
        },
    }
    decision = _resolve("publish this post", route=_route("standard"), cfg=cfg)
    assert decision is not None
    assert decision.action == "seek_tool"
    assert decision.require_tool_evidence is True
    assert decision.seek_human is False


def test_risky_external_action_with_existing_tool_evidence_can_answer():
    decision = _resolve(
        "publish this post",
        route=_route("standard"),
        state={"has_tool_evidence": True},
    )
    assert decision is not None
    assert decision.action == "answer"
    assert decision.require_tool_evidence is False
    assert decision.seek_human is False
    assert "risky_external_action" in decision.reasons
