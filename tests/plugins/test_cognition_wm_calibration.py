"""Behavior contract: the track record says nothing until there are enough results, never pools
models or versions, and changes a turn only by making it more careful — a line saying to check,
and the full answer check — for work like the one that went badly.

Pure checks of the estimate, then the real agent loop against a scripted provider, in the test's
isolated HERMES_HOME. Earlier results are recorded straight into the store.
"""

import argparse
import os
import time
from pathlib import Path

import pytest

from plugins.cognition_wm import calibration, cli, runtime
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_ON = ("plugins:\n  enabled: [cognition_wm]\n  entries:\n    cognition_wm:\n      settings:\n"
       "        learning: true\n        calibration: true\n"
       "cognition:\n  enabled: true\nauxiliary:\n  title_generation:\n    enabled: false\n")
_VERIFIER = "You are a strict consistency verifier."


@pytest.mark.parametrize("successes, n", [(0, 5), (3, 5), (5, 5), (12, 20), (40, 40)])
def test_the_lower_bound_never_claims_more_than_was_seen(successes, n):
    lower = calibration.wilson_lower(successes, n)
    assert 0.0 <= lower <= successes / n


def test_more_results_at_the_same_rate_mean_more_confidence():
    assert calibration.wilson_lower(4, 5) < calibration.wilson_lower(16, 20) < calibration.wilson_lower(80, 100)


@pytest.mark.parametrize("results, band", [
    (["success"] * 4, "unknown"),  # too few results to say anything
    (["success"] * 15, "high"),
    (["success"] * 3 + ["unknown"] * 3, "low"),
    (["success"] * 8 + ["corrected"] * 2, "low"),  # corrected twice lately
    (["success"] * 8 + ["unknown"] * 2, "medium"),
])
def test_the_band_follows_what_was_recorded(results, band):
    assert calibration.band_for(results, min_samples=5) == band


def _record(results, *, model="fake-model", kind="engineering", claim="tests_pass", project=None, version=None):
    from hermes_cli import __version__

    store = runtime.store()
    for n, result in enumerate(results):
        store.record_outcome(evidence_key=f"seed:{model}:{kind}:{claim}:{n}", session_id="earlier", turn_id=f"t{n}",
                             task_id=None, project_key=project or runtime.project_key(), task_kind=kind,
                             claim_class=claim, actor_kind="self", actor_id=model, provider="custom",
                             hermes_version=version or __version__, result=result, at=time.time() - 3600 + n)


class Session:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.work = tmp_path / "work"
        self.work.mkdir(parents=True)
        monkeypatch.chdir(self.work)
        self.home = Path(os.environ["HERMES_HOME"])
        self.script: list = []
        self.server = FakeLLMServer(lambda _r: self.script.pop(0) if self.script else Text("好。"),
                                    aux=lambda _r: Text('{"verdict":"ok"}'))

    def __enter__(self) -> "Session":
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB

        self.server.start()
        write_hermes_home(self.home, self.server.base_url, extra_config=_ON)
        discover_plugins(force=True)
        self.db = SessionDB(db_path=self.home / "state.db")
        return self

    def __exit__(self, *_exc) -> None:
        self.server.stop()

    def turn(self, message: str, *, model: str = "fake-model") -> str:
        from run_agent import AIAgent

        agent = AIAgent(provider="custom", base_url=self.server.base_url, api_key="sk-fake", model=model,
                        session_db=self.db, session_id="s1", quiet_mode=True, platform="cli",
                        enabled_toolsets=["file", "cognition_wm"], skip_context_files=True, skip_memory=True)
        agent.run_conversation(message, conversation_history=self.db.get_messages_as_conversation("s1"))
        body = self.server.main_requests()[-1]
        return next(str(m.get("content")) for m in reversed(body["messages"]) if m.get("role") == "user")

    def verifier_calls(self) -> int:
        return sum(_VERIFIER in str(body.get("messages")) for body in self.server.aux_requests())


def _task(kind: str = "engineering") -> ToolCall:
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {
        "action": "create", "title": "修測試", "goal": "修好不穩的測試", "kind": kind}}]})


def test_a_poor_record_for_this_kind_of_work_adds_a_check(tmp_path, monkeypatch):
    with Session(tmp_path, monkeypatch) as session:
        _record(["success"] * 2 + ["unknown"] * 3 + ["corrected"] * 2)
        session.script += [_task(), Text("好，開始。")]
        session.turn("幫我修不穩的測試")
        before = session.verifier_calls()
        row = session.turn("繼續")

    assert "Track record" in row and "tests pass" in row and "2 verified of 7" in row
    assert session.verifier_calls() == before + 1


def test_another_models_record_is_not_this_ones(tmp_path, monkeypatch):
    with Session(tmp_path, monkeypatch) as session:
        _record(["corrected"] * 6, model="old-model")
        _record(["corrected"] * 6, version="0.0.1")  # this model, an earlier Hermes
        session.script += [_task(), Text("好，開始。")]
        session.turn("幫我修不穩的測試")
        before = session.verifier_calls()
        row = session.turn("繼續")

    assert "Track record" not in row and session.verifier_calls() == before


def test_a_good_record_or_too_few_results_change_nothing(tmp_path, monkeypatch):
    for results in (["success"] * 15, ["unknown"] * 4):
        with Session(tmp_path / str(len(results)), monkeypatch) as session:
            _record(results)
            session.script += [_task(), Text("好，開始。")]
            session.turn("幫我修不穩的測試")
            before = session.verifier_calls()
            row = session.turn("繼續")
        assert "Track record" not in row and session.verifier_calls() == before


def test_chat_outside_any_task_is_never_checked_harder(tmp_path, monkeypatch):
    with Session(tmp_path, monkeypatch) as session:
        _record(["corrected"] * 8, kind="none")  # claims made outside any task went badly
        row = session.turn("今天天氣如何？")

    assert "Track record" not in row and session.verifier_calls() == 0


def test_the_track_record_command_keeps_models_apart_and_says_when_it_is_too_early(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _record(["success"] * 3, model="model-a")
    _record(["success"] * 12 + ["corrected"] * 3, model="model-b")
    parser = argparse.ArgumentParser(prog="hermes cognition")
    cli.register_cli(parser)

    assert cli.cognition_command(parser.parse_args(["track-record"])) == 0
    out = capsys.readouterr().out
    a = next(line for line in out.splitlines() if "model-a" in line)
    b = next(line for line in out.splitlines() if "model-b" in line)
    assert "too few results" in a and "3 verified of 3" in a
    assert "12 verified of 15" in b and "too few" not in b

    assert cli.cognition_command(parser.parse_args(["track-record", "--reset"])) == 0
    assert runtime.store().outcomes() == []


def test_tasks_marked_done_without_evidence_ids_alone_never_add_a_check(tmp_path, monkeypatch):
    """Found with the real model: it marks tasks done without citing evidence ids far more often than
    its answers are wrong. That bookkeeping gap is shown in the track record but does not make every
    turn of the kind of work pay for the full answer check."""
    with Session(tmp_path, monkeypatch) as session:
        _record(["unknown"] * 8, claim="task")
        session.script += [_task(), Text("好，開始。")]
        session.turn("幫我修不穩的測試")
        before = session.verifier_calls()
        row = session.turn("繼續")

    assert "Track record" not in row and session.verifier_calls() == before
