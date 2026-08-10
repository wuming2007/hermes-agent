"""Tests for the cognitive router scaffold (PR1)."""

from __future__ import annotations

from agent.cognitive_router import (
    CognitiveRoute,
    gate_cheap_route,
    resolve_cognitive_route,
)


_BASE_CFG = {
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
    "consistency_guard": {
        "enabled": True,
        "deep_mode_only": True,
    },
}


_DEFAULT = object()


def _resolve(message: str, cfg=_DEFAULT, history=None, state=None):
    return resolve_cognitive_route(
        user_message=message,
        conversation_history=history,
        routing_config=_BASE_CFG if cfg is _DEFAULT else cfg,
        agent_state=state,
    )


def test_disabled_cognition_returns_none():
    cfg = {**_BASE_CFG, "enabled": False}
    assert _resolve("hi there", cfg=cfg) is None


def test_disabled_cognition_returns_none_when_config_missing():
    assert _resolve("hi there", cfg=None) is None
    assert _resolve("hi there", cfg={}) is None


def test_fast_route_for_short_simple_prompt():
    route = _resolve("what time is it in tokyo?")
    assert isinstance(route, CognitiveRoute)
    assert route.mode == "fast"
    assert route.retrieval_plan == "principles_only"
    assert route.verification_plan == "none"
    assert route.allow_cheap_model is True
    assert route.consistency_check is False
    assert route.dialogue_mode == "query"
    assert route.answer_density == "brief"
    assert route.stance_reasons
    assert route.routing_reasons  # populated


def test_standard_route_for_nontrivial_prompt():
    # Long enough to exceed fast-mode thresholds but no deep trigger keywords.
    prompt = (
        "Could you please help me draft a thoughtful and friendly multi-paragraph "
        "note about our upcoming team picnic that covers food preferences, "
        "accessibility needs, transportation options, weather contingencies, "
        "and a few possible time slots so everyone can weigh in early?"
    )
    route = _resolve(prompt)
    assert route is not None
    assert route.mode == "standard"
    assert route.retrieval_plan == "principles_plus_semantic"
    assert route.verification_plan == "light"
    assert route.allow_cheap_model is False
    assert route.consistency_check is False


def test_deep_route_for_historical_prompt():
    route = _resolve("上次我們討論的 root cause 是什麼？")
    assert route is not None
    assert route.mode == "deep"
    assert route.retrieval_plan == "principles_plus_semantic_plus_episodic"
    assert route.verification_plan == "full"
    assert route.allow_cheap_model is False
    assert route.consistency_check is True
    assert any("historical" in r for r in route.routing_reasons)


def test_deep_route_for_architecture_prompt():
    route = _resolve("Let's discuss the architecture tradeoff for this design")
    assert route is not None
    assert route.mode == "deep"
    assert route.retrieval_plan == "principles_plus_semantic_plus_episodic"
    assert route.allow_cheap_model is False
    assert any("architecture" in r for r in route.routing_reasons)


def test_deep_route_for_code_change_prompt():
    route = _resolve("please refactor this module")
    assert route is not None
    assert route.mode == "deep"
    assert route.allow_cheap_model is False
    assert any("code_change" in r for r in route.routing_reasons)


def test_deep_route_for_risky_external_action_prompt():
    route = _resolve("go ahead and publish the tweet now")
    assert route is not None
    assert route.mode == "deep"
    assert route.allow_cheap_model is False
    assert any("risky_external" in r for r in route.routing_reasons)


def test_long_prompt_falls_back_to_standard_not_fast():
    prompt = "please tell me about your day " * 20
    route = _resolve(prompt)
    assert route is not None
    assert route.mode == "standard"
    assert route.allow_cheap_model is False


def test_url_in_prompt_disqualifies_fast():
    route = _resolve("see https://example.com for context")
    assert route is not None
    # Not a deep trigger, has URL so not fast either.
    assert route.mode == "standard"


def test_code_block_disqualifies_fast():
    route = _resolve("hey ```python\nprint('hi')\n``` thoughts?")
    assert route is not None
    assert route.mode == "standard"


def test_consistency_check_off_when_guard_disabled():
    cfg = {
        **_BASE_CFG,
        "consistency_guard": {"enabled": False, "deep_mode_only": True},
    }
    route = _resolve("上次的設計回顧", cfg=cfg)
    assert route is not None
    assert route.mode == "deep"
    assert route.consistency_check is False


def test_deep_trigger_categories_can_be_disabled():
    cfg = {
        **_BASE_CFG,
        "deep_mode_triggers": {
            "historical_questions": False,
            "code_changes": False,
            "risky_external_actions": False,
            "architecture_decisions": False,
        },
    }
    route = _resolve("please refactor this", cfg=cfg)
    assert route is not None
    # No deep triggers, short, no code/url -> fast.
    assert route.mode == "fast"


def test_empty_message_is_safely_standard():
    route = _resolve("")
    assert route is not None
    # Empty messages should not be classified as fast (cheap routing would be
    # meaningless) and have no deep trigger content; standard is the safe default.
    assert route.mode == "standard"
    assert route.allow_cheap_model is False


def test_routing_reasons_includes_fast_signal():
    route = _resolve("ping")
    assert route is not None
    assert route.mode == "fast"
    assert any("simple" in r or "short" in r for r in route.routing_reasons)


def test_route_detects_status_stance():
    route = _resolve("目前 cognition stack 狀態如何？")
    assert route is not None
    assert route.dialogue_mode == "status"
    assert route.answer_density == "brief"
    assert any(reason.startswith("status:") for reason in route.stance_reasons)


def test_project_status_prompt_routes_standard_not_fast():
    route = _resolve("目前 cognition stack 狀態如何？")
    assert route is not None
    assert route.mode == "standard"
    assert route.retrieval_plan == "principles_plus_semantic"
    assert route.verification_plan == "light"
    assert route.allow_cheap_model is False
    assert any("status_lookup" in reason or "project_state" in reason for reason in route.routing_reasons)
    assert route.dialogue_mode == "status"
    assert route.answer_density == "brief"


def test_pr_status_prompt_routes_standard_not_fast():
    route = _resolve("PR11 狀態？")
    assert route is not None
    assert route.mode == "standard"
    assert route.retrieval_plan == "principles_plus_semantic"
    assert route.verification_plan == "light"
    assert route.allow_cheap_model is False
    assert any("status_lookup" in reason or "project_state" in reason for reason in route.routing_reasons)


def test_regular_short_query_stays_fast():
    route = _resolve("what time is it in tokyo?")
    assert route is not None
    assert route.mode == "fast"
    assert route.allow_cheap_model is True
    assert route.retrieval_plan == "principles_only"


def test_cognitive_route_constructor_keeps_old_call_shape():
    route = CognitiveRoute(
        mode="fast",
        retrieval_plan="principles_only",
        verification_plan="none",
        allow_cheap_model=True,
        consistency_check=False,
    )
    assert route.dialogue_mode == "query"
    assert route.answer_density == "standard"
    assert route.stance_reasons == []


# ---- gate_cheap_route ------------------------------------------------------


_SAMPLE_CHEAP_ROUTE = {
    "provider": "openrouter",
    "model": "google/gemini-2.5-flash",
    "routing_reason": "simple_turn",
}


def _route(mode):
    return resolve_cognitive_route(
        user_message={
            "fast": "hi there",
            "deep": "上次的 root cause",
        }.get(mode, "please help me draft a long thoughtful note " * 5),
        conversation_history=None,
        routing_config=_BASE_CFG,
        agent_state=None,
    )


def test_gate_passes_through_when_no_cognition():
    assert gate_cheap_route(None, _SAMPLE_CHEAP_ROUTE) == _SAMPLE_CHEAP_ROUTE
    assert gate_cheap_route(None, None) is None


def test_gate_allows_cheap_route_in_fast_mode():
    fast = _route("fast")
    assert fast.mode == "fast"
    assert gate_cheap_route(fast, _SAMPLE_CHEAP_ROUTE) == _SAMPLE_CHEAP_ROUTE


def test_gate_blocks_cheap_route_in_standard_mode():
    standard = _route("standard")
    assert standard.mode == "standard"
    assert gate_cheap_route(standard, _SAMPLE_CHEAP_ROUTE) is None


def test_gate_blocks_cheap_route_in_deep_mode():
    deep = _route("deep")
    assert deep.mode == "deep"
    assert gate_cheap_route(deep, _SAMPLE_CHEAP_ROUTE) is None
