"""Tests for PR10 interaction stance planner."""

from __future__ import annotations

from agent.cognitive_router import CognitiveRoute
from agent.interaction_stance import InteractionStance, resolve_interaction_stance


def _route(mode="fast"):
    if mode == "deep":
        return CognitiveRoute(
            mode="deep",
            retrieval_plan="principles_plus_semantic_plus_episodic",
            verification_plan="full",
            allow_cheap_model=False,
            consistency_check=True,
        )
    if mode == "standard":
        return CognitiveRoute(
            mode="standard",
            retrieval_plan="principles_plus_semantic",
            verification_plan="light",
            allow_cheap_model=False,
            consistency_check=False,
        )
    return CognitiveRoute(
        mode="fast",
        retrieval_plan="principles_only",
        verification_plan="none",
        allow_cheap_model=True,
        consistency_check=False,
    )


_DEFAULT_ROUTE = object()


def _stance(message, route=_DEFAULT_ROUTE, cfg=None):
    return resolve_interaction_stance(
        user_message=message,
        cognition_route=_route() if route is _DEFAULT_ROUTE else route,
        routing_config=cfg or {"enabled": True},
        agent_state={"platform": "cli"},
    )


def test_returns_frozen_interaction_stance():
    stance = _stance("ping")
    assert isinstance(stance, InteractionStance)
    assert stance.dialogue_mode == "query"
    assert stance.answer_density == "brief"
    assert isinstance(stance.stance_reasons, tuple)


def test_status_prompt_is_brief():
    stance = _stance("目前 cognition stack 狀態如何？")
    assert stance.dialogue_mode == "status"
    assert stance.answer_density == "brief"
    assert any(reason.startswith("status:") for reason in stance.stance_reasons)


def test_execution_prompt_is_brief():
    stance = _stance("幫我做 PR10，跑測試後 commit")
    assert stance.dialogue_mode == "execution"
    assert stance.answer_density == "brief"
    assert any(reason.startswith("execution:") for reason in stance.stance_reasons)


def test_debate_prompt_is_standard_density():
    stance = _stance("你覺得這個設計應不應該反駁？")
    assert stance.dialogue_mode == "debate"
    assert stance.answer_density == "standard"


def test_exploration_prompt_is_expanded():
    stance = _stance("Let's explore the design philosophy of cognitive agents")
    assert stance.dialogue_mode == "exploration"
    assert stance.answer_density == "expanded"


def test_fallback_density_follows_cognitive_route():
    assert _stance("ping", route=_route("fast")).answer_density == "brief"
    assert _stance("ordinary question", route=_route("standard")).answer_density == "standard"
    assert _stance("ordinary question", route=_route("deep")).answer_density == "standard"
    assert _stance("ordinary question", route=None).answer_density == "standard"


def test_empty_message_defaults_to_query_standard():
    stance = _stance("", route=_route("fast"))
    assert stance.dialogue_mode == "query"
    assert stance.answer_density == "standard"
    assert "empty_message" in stance.stance_reasons


def test_config_overrides_valid_values_and_ignores_invalid_values():
    cfg = {
        "enabled": True,
        "interaction_stance": {
            "mode_overrides": {"status": "query", "execution": "not-a-mode"},
            "density_overrides": {"query": "expanded", "execution": "not-a-density"},
        },
    }
    status = _stance("目前狀態？", cfg=cfg)
    assert status.dialogue_mode == "query"
    assert status.answer_density == "expanded"
    assert "config_override" in status.stance_reasons

    execution = _stance("please commit this", cfg=cfg)
    assert execution.dialogue_mode == "execution"
    assert execution.answer_density == "brief"
