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
    rows = [r for r in rows if r["kind"] == "turn"]
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
    (row,) = [r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if r["kind"] == "turn"]
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


def _run_scripted(tmp_path, monkeypatch, case: dict, responder, arms, repeats=1):
    from evals.cognition_v2 import runner

    home = tmp_path / "eval-home"
    runner.init_home(home)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    with FakeLLMServer(responder, aux=lambda _r: Text('{"verdict":"ok"}')) as server:
        write_hermes_home(home, server.base_url, extra_config=_NO_BORROWED_LOGINS)
        runner.run(home, cases, out, arms=arms, repeats=repeats, case_set="dev")
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def _test_runner(record):
    """Runs the tests the way the request's lessons say, if any."""
    body = record["body"]
    rows = [str(m.get("content")) for m in body["messages"] if m.get("role") == "user"]
    asked = rows[-1]
    if body["messages"][-1].get("role") == "tool":
        return Text("測試都通過了。")
    if "測試" in asked.split("\n", 1)[0]:
        return ToolCall("terminal", {"command": "sh run_tests.sh" if "sh run_tests.sh" in asked else "sh old_tests.sh"})
    return Text("好，記下了。")


_LESSON_CASE = {
    "id": "carry-over", "set": "dev",
    "files": {"run_tests.sh": "echo '3 passed'\n", "old_tests.sh": "echo '3 passed'\n"},
    "sessions": [
        {"turns": ["幫我跑測試", "不對，要用 sh run_tests.sh 跑測試，不要用 old_tests.sh"]},
        {"turns": ["跑一下測試"]},
    ],
    "expect": [{"session": 1, "turn": 0, "ran": "run_tests.sh"}, {"session": 1, "turn": 0, "not_ran": "old_tests.sh"}],
}


def test_three_arms_are_measured_across_conversations_and_only_the_full_v2_carries_a_lesson(tmp_path, monkeypatch):
    rows = [r for r in _run_scripted(tmp_path, monkeypatch, _LESSON_CASE, _test_runner, ["off", "on", "v2"])
            if r.get("kind") == "turn"]

    assert [(r["arm"], r["session"], r["turn"]) for r in rows] == [
        (arm, s, t) for arm in ("off", "on", "v2") for s, t in ((0, 0), (0, 1), (1, 0))]
    day2 = {r["arm"]: r for r in rows if r["session"] == 1}
    assert [c["ok"] for c in day2["v2"]["checks"]] == [True, True]
    assert [c["ok"] for c in day2["on"]["checks"]] == [False, False]  # phase 1 alone forgets
    assert day2["v2"]["lessons_chars"] > 0 and day2["on"]["lessons_chars"] == 0


def test_each_case_run_starts_without_what_earlier_runs_learned(tmp_path, monkeypatch):
    rows = _run_scripted(tmp_path, monkeypatch, _LESSON_CASE, _test_runner, ["v2"], repeats=2)

    # the second run counts only what it learned itself, not the first run's lesson too
    assert [r["lessons"] for r in rows if r.get("kind") == "case"] == [{"candidate": 1}, {"candidate": 1}]


def test_a_case_summary_counts_what_was_learned_and_how_claims_held_up(tmp_path, monkeypatch):
    rows = _run_scripted(tmp_path, monkeypatch, _LESSON_CASE, _test_runner, ["v2"])
    (summary,) = [r for r in rows if r.get("kind") == "case"]

    assert summary["lessons"] == {"candidate": 1}
    assert summary["outcomes"]["self"]["success"] >= 1


def test_a_blocked_command_did_not_run():
    from evals.cognition_v2 import runner

    def call(call_id, command):
        return {"id": call_id, "type": "function",
                "function": {"name": "terminal", "arguments": json.dumps({"command": command})}}

    rows = [{"role": "user", "content": "跑測試"},
            {"role": "assistant", "tool_calls": [call("a", "sh deploy.sh"), call("b", "sh run_tests.sh")]},
            {"role": "tool", "tool_call_id": "a", "content": "BLOCKED: the user paused this task"},
            {"role": "tool", "tool_call_id": "b", "content": "3 passed"}]

    assert runner._ran(rows) == ["sh run_tests.sh"]


def test_the_run_uses_the_eval_homes_own_model_and_provider(tmp_path, monkeypatch):
    """Like `hermes chat -q`: the configured provider, model and API mode, never an unresolved
    default (an unresolved agent sent chat completions to the Codex backend and got 404s)."""
    rows = _run_scripted(tmp_path, monkeypatch, {"id": "chat", "set": "dev", "turns": ["今天適合出門嗎？"]},
                         lambda _r: Text("今天天氣不錯。"), ["off"])

    (row,) = [r for r in rows if r["kind"] == "turn"]
    assert (row["model"], row["provider"]) == ("fake-model", "custom")
