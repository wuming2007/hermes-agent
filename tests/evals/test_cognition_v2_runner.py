"""Behavior contract: the cognition v2 eval runner measures both arms in a dedicated eval home and
leaves that home's config as it found it; it never runs in the everyday home.

Hermetic: the eval home's model is the scripted loopback provider.
"""

import json
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

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


def _confinement_available():
    from evals.cognition_v2 import sandbox

    return sandbox.available()


@pytest.mark.skipif(not _confinement_available(), reason="the evaluation's confinement is macOS sandbox-exec")
def test_the_confinement_keeps_a_run_inside_the_eval_home(tmp_path):
    import subprocess
    import sys

    from evals.cognition_v2 import sandbox

    home, data = tmp_path / "home", tmp_path / "data"
    eval_home = home / "eval-home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_test").write_text("not for the run", encoding="utf-8")
    eval_home.mkdir()
    data.mkdir()
    agent = tmp_path / "bin" / "claude"
    agent.parent.mkdir()
    agent.write_text(f"#!/bin/sh\ntouch '{tmp_path / 'agent-ran'}'\n", encoding="utf-8")
    agent.chmod(0o755)
    profile = tmp_path / "run.sb"
    profile.write_text(sandbox.profile(eval_home=eval_home, repo=Path(sandbox.__file__).resolve().parents[2], home=home,
                                       python_root=Path(sys.executable).resolve().parents[1], denied=[agent],
                                       protected=[data]), encoding="utf-8")
    script = (f"echo ok > '{eval_home}/w' && echo EVAL_HOME_WRITABLE\n"
              f"echo x > '{home}/w' 2>/dev/null || echo HOME_WRITE_DENIED\n"
              f"echo x > '{data}/w' 2>/dev/null || echo DATA_WRITE_DENIED\n"
              f"cat '{home}/.ssh/id_test' >/dev/null 2>&1 || echo HOME_READ_DENIED\n"
              f"'{agent}' >/dev/null 2>&1 || echo AGENT_DENIED\n"
              f"'{sys.executable}' -c 'import hermes_cli, run_agent' && echo CODE_LOADS\n")

    out = subprocess.run([str(sandbox.SANDBOX_EXEC), "-f", str(profile), "/bin/sh", "-c", script],
                         capture_output=True, text=True, timeout=120).stdout

    for marker in ("EVAL_HOME_WRITABLE", "HOME_WRITE_DENIED", "DATA_WRITE_DENIED", "HOME_READ_DENIED", "AGENT_DENIED",
                   "CODE_LOADS"):
        assert marker in out
    assert not (home / "w").exists() and not (data / "w").exists() and not (tmp_path / "agent-ran").exists()


@pytest.mark.skipif(not _confinement_available(), reason="the evaluation's confinement is macOS sandbox-exec")
def test_the_evaluation_runs_confined_and_a_coding_agent_is_only_a_stand_in(tmp_path):
    from evals.cognition_v2 import runner, sandbox

    home = tmp_path / "eval-home"
    runner.init_home(home)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "prompt-only", "turns": ["幫我寫一段給 Claude 的 prompt"]}) + "\n", encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    script = [ToolCall("terminal", {"command": "claude -p 'refactor the parser'"}), Text("這是 prompt：……")]

    with FakeLLMServer(script) as server:
        write_hermes_home(home, server.base_url, extra_config=_NO_BORROWED_LOGINS)
        with pytest.raises(SystemExit) as exit_:
            runner.main(["--home", str(home), "--cases", str(cases), "--out", str(out), "--arms", "off",
                         "--repeats", "1"])

    assert exit_.value.code == 0
    (row,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert any("claude -p" in call for call in row["calls"])
    assert sandbox.command_log(home) == ["claude -p refactor the parser"]


def test_without_confinement_the_evaluation_does_not_run(tmp_path, monkeypatch):
    from evals.cognition_v2 import runner, sandbox

    home = tmp_path / "eval-home"
    runner.init_home(home)
    monkeypatch.setattr(sandbox, "available", lambda: False)
    monkeypatch.delenv(sandbox.MARKER_ENV, raising=False)

    with pytest.raises(SystemExit) as exit_:
        runner.main(["--home", str(home), "--cases", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "o.jsonl")])

    assert "confined" in str(exit_.value.code)
