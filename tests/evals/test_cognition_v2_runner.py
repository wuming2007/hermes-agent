"""Behavior contract: the cognition v2 eval runner measures both arms in a dedicated eval home and
leaves that home's config as it found it; it never runs in the everyday home.

Hermetic: the eval home's model is the scripted loopback provider.
"""

import json
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

_NO_BORROWED_LOGINS = "auth:\n  adopt_external_logins: false\n"


def test_both_arms_are_measured_and_the_config_is_restored(tmp_path, monkeypatch):
    from evals.cognition_v2 import runner

    home = tmp_path / "eval-home"
    runner.init_home(home)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "chat", "set": "dev", "turns": ["今天適合出門嗎？", "謝謝"]}) + "\n",
                     encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    with FakeLLMServer([], default_text="今天天氣不錯。") as server:
        write_hermes_home(home, server.base_url, extra_config=_NO_BORROWED_LOGINS)
        before = (home / "config.yaml").read_text(encoding="utf-8")
        runner.run(home, cases, out, arms=["off", "on"], repeats=1, case_set="dev")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["arm"], r["turn"]) for r in rows] == [("off", 0), ("off", 1), ("on", 0), ("on", 1)]
    assert all(r["api_calls"] == 1 and r["final_response"] == "今天天氣不錯。" for r in rows)
    assert (home / "config.yaml").read_text(encoding="utf-8") == before
    summary = runner.summarize(out)
    assert summary["on"]["api_calls"]["median"] == summary["off"]["api_calls"]["median"] == 1


def test_the_everyday_home_is_refused(tmp_path, monkeypatch):
    from evals.cognition_v2 import runner

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    everyday = tmp_path / ".hermes"
    everyday.mkdir()
    (everyday / runner.MARKER).write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.run(everyday, tmp_path / "cases.jsonl", tmp_path / "out.jsonl", arms=["off"], repeats=1, case_set="dev")


def test_a_new_eval_home_never_borrows_another_programs_login(tmp_path, monkeypatch):
    from agent.credential_sources import adopt_external_logins_enabled
    from evals.cognition_v2 import runner

    home = tmp_path / "eval-home"
    runner.init_home(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert adopt_external_logins_enabled() is False


def test_an_eval_home_that_could_borrow_a_login_is_refused(tmp_path, monkeypatch):
    from evals.cognition_v2 import runner

    home = tmp_path / "eval-home"
    runner.init_home(home)
    (home / "config.yaml").write_text("auth:\n  adopt_external_logins: true\n", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "chat", "turns": ["hi"]}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.run(home, cases, tmp_path / "out.jsonl", arms=["off"], repeats=1, case_set="dev")
