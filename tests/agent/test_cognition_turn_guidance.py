"""Behavior contract: a ``pre_llm_call`` hook may raise this turn's answer check through
``turn_guidance`` — never lower it, never touch the route's mode or recall — and both turn hooks
name the provider the turn runs on.

Real agent loop against a scripted provider; the plugin hooks are a recording stand-in for
``hermes_cli.lifecycle.invoke_hook``.
"""

import pytest

from agent.cognition_routing import apply_verification_floor
from agent.cognitive_router import CognitiveRoute
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

_CONFIG = "cognition:\n  enabled: true\nauxiliary:\n  title_generation:\n    enabled: false\n"
_VERIFIER = "You are a strict consistency verifier."


class _Hooks:
    """Stands in for the plugin hooks: records every call, answers ``pre_llm_call`` from a queue."""

    def __init__(self, guidance: list):
        self.calls: list[tuple[str, dict]] = []
        self._guidance = list(guidance)

    def __call__(self, hook_name: str, **kwargs):
        self.calls.append((hook_name, kwargs))
        if hook_name == "pre_llm_call" and self._guidance:
            guidance = self._guidance.pop(0)
            return [{"turn_guidance": guidance}] if guidance else []
        return []


def _run_turns(monkeypatch, tmp_path, guidance: list, messages: list[str]):
    """Run ``messages`` as consecutive turns; returns (per-turn cognition metadata, hooks, server)."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    hooks = _Hooks(guidance)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hooks)
    home = tmp_path / "home"
    (tmp_path / "work").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setenv("HERMES_HOME", str(home))
    metas = []
    with FakeLLMServer([Text("整理好了。") for _ in messages], aux=lambda _r: Text('{"verdict":"ok"}')) as server:
        write_hermes_home(home, server.base_url, extra_config=_CONFIG)
        db = SessionDB(db_path=home / "state.db")
        history: list = []
        for message in messages:
            agent = AIAgent(provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model",
                            session_db=db, session_id="sess-guidance", quiet_mode=True, platform="cli",
                            enabled_toolsets=["file"], skip_context_files=True, skip_memory=True)
            agent.run_conversation(message, conversation_history=history)
            metas.append(dict(agent._current_turn_cognition_metadata))
            history = db.get_messages_as_conversation("sess-guidance")
    return metas, hooks, server


def _verifier_calls(server) -> int:
    return sum(_VERIFIER in str(body.get("messages")) for body in server.aux_requests())


def test_a_floor_raises_the_answer_check_and_says_why(tmp_path, monkeypatch):
    (base,), _hooks, base_server = _run_turns(monkeypatch, tmp_path / "base", [None], ["整理一下這段文字"])
    (meta,), _hooks, server = _run_turns(
        monkeypatch, tmp_path / "floor", [{"verification_floor": "full", "reasons": ["weak_record"]}],
        ["整理一下這段文字"])

    assert base["verification_plan"] != "full" and _verifier_calls(base_server) == 0
    assert meta["verification_plan"] == "full" and _verifier_calls(server) == 1
    assert "track_record:weak_record" in meta["routing_reasons"]
    # only the check is raised: how the turn recalls and which mode it runs in stay the same
    assert (meta["mode"], meta["retrieval_plan"]) == (base["mode"], base["retrieval_plan"])


def test_guidance_holds_for_its_own_turn_only(tmp_path, monkeypatch):
    metas, _hooks, server = _run_turns(monkeypatch, tmp_path, [{"verification_floor": "full"}, None],
                                       ["整理一下這段文字", "再整理一次"])

    assert metas[0]["verification_plan"] == "full"
    assert metas[1]["verification_plan"] != "full"
    assert _verifier_calls(server) == 1


def test_a_track_record_band_reaches_the_turn_metadata(tmp_path, monkeypatch):
    (meta,), _hooks, _server = _run_turns(monkeypatch, tmp_path, [{"track_record_band": "low"}],
                                          ["整理一下這段文字"])

    assert meta["track_record_band"] == "low"


def test_both_turn_hooks_name_the_provider(tmp_path, monkeypatch):
    _metas, hooks, _server = _run_turns(monkeypatch, tmp_path, [None], ["整理一下這段文字"])

    seen = {name: kwargs.get("provider") for name, kwargs in hooks.calls if name in ("pre_llm_call", "post_llm_call")}
    assert seen == {"pre_llm_call": "custom", "post_llm_call": "custom"}


def _route(plan: str) -> CognitiveRoute:
    return CognitiveRoute(mode="standard", retrieval_plan="principles_plus_semantic", verification_plan=plan,
                          allow_cheap_model=False, consistency_check=False, routing_reasons=["base"])


@pytest.mark.parametrize("plan, floor, expected", [
    ("none", "full", "full"),
    ("light", "full", "full"),
    ("none", "light", "light"),
    ("full", "light", "full"),   # never lowered
    ("light", "none", "light"),
    ("light", "bogus", "light"),  # an unknown floor says nothing
    ("light", None, "light"),
])
def test_a_floor_only_ever_raises_the_plan(plan, floor, expected):
    route = apply_verification_floor(_route(plan), {"verification_floor": floor} if floor else {})

    assert route.verification_plan == expected
    assert (route.mode, route.retrieval_plan, route.allow_cheap_model) == ("standard", "principles_plus_semantic", False)
    assert ("base" in route.routing_reasons) and (len(route.routing_reasons) > 1) == (expected != plan)
