"""Cron-side wiring of the PR1 cognitive router.

Verifies that:
- the helper returns ``None`` when cognition is disabled (legacy passthrough),
- the helper produces the expected mode for fast / deep prompts, and
- when threaded into ``resolve_turn_route(..., cognition_route=...)`` (the
  exact pattern used inside ``run_job``) cheap routing is gated correctly
  on the cron path.
"""

from __future__ import annotations

import sys
import types

# cron.scheduler imports many heavy modules at top-level (fire, firecrawl,
# etc.); stub the ones not present in CI so importing the module doesn't fail.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import pytest

from agent.cognitive_router import CognitiveRoute
from agent.smart_model_routing import resolve_turn_route
from cron.scheduler import _resolve_cron_cognitive_route


_COGNITION_CFG = {
    "enabled": True,
    "fast_mode": {"max_chars": 160, "max_words": 28},
    "deep_mode_triggers": {
        "historical_questions": True,
        "code_changes": True,
        "risky_external_actions": True,
        "architecture_decisions": True,
    },
    "consistency_guard": {"enabled": True, "deep_mode_only": True},
}


_PRIMARY = {
    "model": "anthropic/claude-sonnet-4",
    "api_key": "primary-key",
    "base_url": "https://openrouter.ai/api/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
    "command": None,
    "args": [],
    "credential_pool": None,
}


_SMART_ROUTING_CFG = {
    "enabled": True,
    "cheap_model": {"provider": "zai", "model": "glm-5-air"},
    "max_simple_chars": 160,
    "max_simple_words": 28,
}


# ---- _resolve_cron_cognitive_route ----------------------------------------


def test_helper_returns_none_when_cfg_empty():
    assert _resolve_cron_cognitive_route("hello", {}) is None
    assert _resolve_cron_cognitive_route("hello", None) is None


def test_helper_returns_none_when_cognition_disabled():
    cfg = {**_COGNITION_CFG, "enabled": False}
    assert _resolve_cron_cognitive_route("hello", cfg) is None


def test_helper_returns_fast_route_for_simple_prompt():
    route = _resolve_cron_cognitive_route("ping", _COGNITION_CFG)
    assert isinstance(route, CognitiveRoute)
    assert route.mode == "fast"
    assert route.allow_cheap_model is True


def test_helper_returns_deep_route_for_trigger_prompt():
    route = _resolve_cron_cognitive_route("should I publish this?", _COGNITION_CFG)
    assert isinstance(route, CognitiveRoute)
    assert route.mode == "deep"
    assert route.allow_cheap_model is False


# ---- end-to-end gating through resolve_turn_route -------------------------


def test_cron_disabled_cognition_preserves_cheap_route(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_: {
            "provider": "zai",
            "api_mode": "chat_completions",
            "base_url": "https://open.z.ai/api/v1",
            "api_key": "cheap-key",
            "source": "env/config",
        },
    )
    cognition_route = _resolve_cron_cognitive_route(
        "what time is it in tokyo?", {}, model="anthropic/claude-sonnet-4"
    )
    assert cognition_route is None  # disabled

    result = resolve_turn_route(
        "what time is it in tokyo?",
        _SMART_ROUTING_CFG,
        _PRIMARY,
        cognition_route=cognition_route,
    )
    assert result["model"] == "glm-5-air"


def test_cron_fast_cognition_allows_cheap_route(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_: {
            "provider": "zai",
            "api_mode": "chat_completions",
            "base_url": "https://open.z.ai/api/v1",
            "api_key": "cheap-key",
            "source": "env/config",
        },
    )
    cognition_route = _resolve_cron_cognitive_route(
        "what time is it in tokyo?", _COGNITION_CFG, model="anthropic/claude-sonnet-4"
    )
    assert cognition_route is not None and cognition_route.mode == "fast"

    result = resolve_turn_route(
        "what time is it in tokyo?",
        _SMART_ROUTING_CFG,
        _PRIMARY,
        cognition_route=cognition_route,
    )
    assert result["model"] == "glm-5-air"


def test_cron_deep_cognition_blocks_cheap_route(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_: pytest.fail(
            "runtime provider must NOT be invoked when cheap route is gated off"
        ),
    )
    # "publish" is a risky_external trigger but is NOT in
    # smart_model_routing._COMPLEX_KEYWORDS, so the gate is the only thing
    # keeping cheap routing off on the cron path too.
    cognition_route = _resolve_cron_cognitive_route(
        "should I publish this now?", _COGNITION_CFG, model="anthropic/claude-sonnet-4"
    )
    assert cognition_route is not None and cognition_route.mode == "deep"

    result = resolve_turn_route(
        "should I publish this now?",
        _SMART_ROUTING_CFG,
        _PRIMARY,
        cognition_route=cognition_route,
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None
