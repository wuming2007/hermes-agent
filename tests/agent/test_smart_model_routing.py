from agent.smart_model_routing import choose_cheap_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_choose_cheap_model_route_unaffected_when_cognition_route_is_none():
    # Backward compatibility: not passing a cognition_route preserves
    # existing behavior exactly.
    result = choose_cheap_model_route(
        "what time is it in tokyo?", _BASE_CONFIG, cognition_route=None
    )
    assert result is not None
    assert result["model"] == "google/gemini-2.5-flash"


def test_choose_cheap_model_route_allowed_in_fast_cognition_mode():
    from agent.cognitive_router import resolve_cognitive_route

    cognition_cfg = {
        "enabled": True,
        "default_mode": "standard",
        "fast_mode": {"max_chars": 160, "max_words": 28},
        "deep_mode_triggers": {
            "historical_questions": True,
            "code_changes": True,
            "risky_external_actions": True,
            "architecture_decisions": True,
        },
        "consistency_guard": {"enabled": True, "deep_mode_only": True},
    }
    fast_route = resolve_cognitive_route(
        user_message="what time is it in tokyo?",
        conversation_history=None,
        routing_config=cognition_cfg,
        agent_state=None,
    )
    assert fast_route.mode == "fast"
    result = choose_cheap_model_route(
        "what time is it in tokyo?", _BASE_CONFIG, cognition_route=fast_route
    )
    assert result is not None
    assert result["model"] == "google/gemini-2.5-flash"


def test_choose_cheap_model_route_blocked_in_standard_cognition_mode():
    from agent.cognitive_router import CognitiveRoute

    standard_route = CognitiveRoute(
        mode="standard",
        retrieval_plan="principles_plus_semantic",
        verification_plan="light",
        allow_cheap_model=False,
        consistency_check=False,
        routing_reasons=["over_max_chars"],
    )
    # Even a clearly fast-eligible message must be denied when the
    # cognition route says cheap routing is not allowed.
    result = choose_cheap_model_route(
        "ping", _BASE_CONFIG, cognition_route=standard_route
    )
    assert result is None


def test_choose_cheap_model_route_blocked_in_deep_cognition_mode():
    from agent.cognitive_router import CognitiveRoute

    deep_route = CognitiveRoute(
        mode="deep",
        retrieval_plan="principles_plus_semantic_plus_episodic",
        verification_plan="full",
        allow_cheap_model=False,
        consistency_check=True,
        routing_reasons=["historical:上次"],
    )
    result = choose_cheap_model_route(
        "what time is it in tokyo?", _BASE_CONFIG, cognition_route=deep_route
    )
    assert result is None


def test_resolve_turn_route_falls_back_to_primary_when_cognition_blocks_cheap_route():
    from agent.cognitive_router import CognitiveRoute
    from agent.smart_model_routing import resolve_turn_route

    deep_route = CognitiveRoute(
        mode="deep",
        retrieval_plan="principles_plus_semantic_plus_episodic",
        verification_plan="full",
        allow_cheap_model=False,
        consistency_check=True,
        routing_reasons=["code_change:refactor"],
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
        cognition_route=deep_route,
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["label"] is None


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None
