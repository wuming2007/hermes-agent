"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
        }
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def test_turn_route_injects_priority_processing_without_changing_runtime():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    with patch("agent.smart_model_routing.resolve_turn_route", return_value={
        "model": "gpt-5.4",
        "runtime": dict(runtime_kwargs),
        "label": None,
        "signature": ("gpt-5.4", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
    }):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openrouter"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}


def test_turn_route_skips_priority_processing_for_unsupported_models():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    with patch("agent.smart_model_routing.resolve_turn_route", return_value={
        "model": "gpt-5.3-codex",
        "runtime": dict(runtime_kwargs),
        "label": None,
        "signature": ("gpt-5.3-codex", "openrouter", "https://openrouter.ai/api/v1", "chat_completions", None, ()),
    }):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.3-codex", runtime_kwargs)

    assert route["request_overrides"] is None


@pytest.mark.asyncio
async def test_handle_fast_command_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"


@pytest.mark.asyncio
async def test_run_agent_passes_priority_processing_to_gateway_agent(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  service_tier: fast\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["service_tier"] == "priority"
    assert _CapturingAgent.last_init["request_overrides"] == {"service_tier": "priority"}


# ---------------------------------------------------------------------------
# Cognitive routing gate (PR1) — gateway production wiring
# ---------------------------------------------------------------------------


_GATEWAY_COGNITION_CFG = {
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


def _make_runner_with_cheap_routing(cognition_cfg):
    runner = _make_runner()
    runner._smart_model_routing = {
        "enabled": True,
        "cheap_model": {"provider": "zai", "model": "glm-5-air"},
        "max_simple_chars": 160,
        "max_simple_words": 28,
    }
    runner._cognition_config = cognition_cfg
    return runner


_PRIMARY_RUNTIME = {
    "api_key": "primary-key",
    "base_url": "https://openrouter.ai/api/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
    "command": None,
    "args": [],
    "credential_pool": None,
}


def test_gateway_cognition_disabled_preserves_cheap_route(monkeypatch):
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
    runner = _make_runner_with_cheap_routing({"enabled": False})
    result = runner._resolve_turn_agent_config(
        "what time is it in tokyo?", "anthropic/claude-sonnet-4", _PRIMARY_RUNTIME
    )
    assert result["model"] == "glm-5-air"
    assert result["runtime"]["provider"] == "zai"


def test_gateway_cognition_fast_mode_allows_cheap_route(monkeypatch):
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
    runner = _make_runner_with_cheap_routing(_GATEWAY_COGNITION_CFG)
    result = runner._resolve_turn_agent_config(
        "what time is it in tokyo?", "anthropic/claude-sonnet-4", _PRIMARY_RUNTIME
    )
    assert result["model"] == "glm-5-air"


def test_gateway_cognition_deep_trigger_blocks_cheap_route(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_: pytest.fail(
            "runtime provider must NOT be invoked when cheap route is gated off"
        ),
    )
    runner = _make_runner_with_cheap_routing(_GATEWAY_COGNITION_CFG)
    # "publish" is a risky_external trigger but is NOT in
    # smart_model_routing._COMPLEX_KEYWORDS, so the gate is the only thing
    # keeping cheap routing off.
    result = runner._resolve_turn_agent_config(
        "should I publish this now?", "anthropic/claude-sonnet-4", _PRIMARY_RUNTIME
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


# ---------------------------------------------------------------------------
# Cognition config loading parity (PR4)
# ---------------------------------------------------------------------------


def test_gateway_load_cognition_config_uses_shared_helper(tmp_path, monkeypatch):
    """gateway/run.py's _load_cognition_config must delegate to the
    shared loader so all entry points share normalization semantics."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "cognition:\n  enabled: true\n  fast_mode:\n    max_chars: 99\n",
        encoding="utf-8",
    )
    result = gateway_run.GatewayRunner._load_cognition_config()
    assert result["enabled"] is True
    assert result["fast_mode"]["max_chars"] == 99


def test_gateway_load_cognition_config_normalizes_malformed_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "cognition:\n  enabled: true\n  fast_mode: broken_string\n  deep_mode_triggers: 42\n",
        encoding="utf-8",
    )
    result = gateway_run.GatewayRunner._load_cognition_config()
    # Malformed sub-blocks must be coerced to {} via the shared helper.
    assert result["enabled"] is True
    assert result["fast_mode"] == {}
    assert result["deep_mode_triggers"] == {}


def test_gateway_malformed_cognition_config_does_not_crash_routing(monkeypatch):
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
    runner = _make_runner_with_cheap_routing(
        {
            "enabled": True,
            "fast_mode": "broken",
            "deep_mode_triggers": [1, 2, 3],
            "consistency_guard": 42,
        }
    )
    # Routing must complete without raising on the malformed sub-blocks.
    result = runner._resolve_turn_agent_config(
        "ping", "anthropic/claude-sonnet-4", _PRIMARY_RUNTIME
    )
    assert "model" in result
