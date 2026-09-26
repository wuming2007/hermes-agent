"""Behavior contract: the owner's corrections and wishes become scoped, traceable lessons, and
claimed results become a track record that only our own checks improve.

Real agent loop, real plugin discovery and the real terminal tool against a scripted provider,
in the test's isolated HERMES_HOME. Commands are tiny stand-in scripts, so nothing real runs.
"""

import os
import stat
from pathlib import Path
from typing import Callable

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_ON = ("plugins:\n  enabled: [cognition_wm]\n  entries:\n    cognition_wm:\n      settings:\n"
       "        learning: true\n        calibration: true\n"
       "cognition:\n  enabled: true\nauxiliary:\n  title_generation:\n    enabled: false\n")
_TOOLSETS = ["file", "terminal", "cognition_wm"]


def _scripted(*responses) -> Callable:
    queue = list(responses)
    return lambda _record: queue.pop(0) if queue else Text("好。")


def _terminal(command: str) -> ToolCall:
    return ToolCall("terminal", {"command": command})


def _task(title: str = "跑測試") -> ToolCall:
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {
        "action": "create", "title": title, "goal": "跑專案的測試並回報", "kind": "engineering"}}]})


def _lesson(**lesson) -> ToolCall:
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {"action": "lesson", "lesson": lesson}}]})


class Chat:
    """One user in one project; every session id is a separate conversation (another day)."""

    def __init__(self, tmp_path: Path, monkeypatch, responder: Callable, config: str = _ON):
        self.work = tmp_path / "work"
        self.work.mkdir(exist_ok=True)
        monkeypatch.chdir(self.work)
        self.home = Path(os.environ["HERMES_HOME"])
        self.bin = tmp_path / "bin"
        self.bin.mkdir(exist_ok=True)
        for name, output in {"pytest": "5 passed", "uv": "5 passed", "claude": "All tests pass now."}.items():
            script = self.bin / name
            script.write_text(f"#!/bin/sh\necho '{output}'\n", encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.server = FakeLLMServer(responder, aux=lambda _r: Text('{"verdict":"ok"}'))
        self.config = config

    def __enter__(self) -> "Chat":
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB

        self.server.start()
        write_hermes_home(self.home, self.server.base_url, extra_config=self.config)
        discover_plugins(force=True)
        self.db = SessionDB(db_path=self.home / "state.db")
        return self

    def __exit__(self, *_exc) -> None:
        self.server.stop()

    def turn(self, session_id: str, message: str) -> dict:
        from run_agent import AIAgent

        agent = AIAgent(provider="custom", base_url=self.server.base_url, api_key="sk-fake", model="fake-model",
                        session_db=self.db, session_id=session_id, quiet_mode=True, platform="cli",
                        enabled_toolsets=_TOOLSETS, skip_context_files=True, skip_memory=True)
        return agent.run_conversation(message, conversation_history=self.db.get_messages_as_conversation(session_id))

    def cmd(self, line: str) -> str:
        return f"{self.bin}/{line}"


def _store():
    from plugins.cognition_wm import runtime

    return runtime.store()


def _live():
    return _store().experiences(statuses=("candidate", "active", "stale"))


# ------------------------------------------------------------------ lessons


def test_a_correction_is_a_candidate_until_the_user_repeats_it_in_another_conversation(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("好，改用 uv run pytest。")]
        chat.turn("day1", "幫我跑測試")
        chat.turn("day1", "不對，要用 uv run pytest，不是直接 pytest")
        (first,) = _live()
        # no task yet: the kind of work is read from what was asked and done
        assert (first.status, first.origin.authored_by, first.task_kind) == ("candidate", "owner", "engineering")
        assert "uv run pytest" in first.behavior
        assert first.situation == 'asked "幫我跑測試", after terminal: pytest -q'
        assert "測試" in first.tags  # findable by what the user asked, not only by the correction

        script += [_terminal(chat.cmd("pytest")), Text("測試都通過了。"), Text("好。")]
        chat.turn("day2", "跑一下測試")
        chat.turn("day2", "不對，測試要用 uv run pytest")

    (lesson,) = _live()
    assert lesson.status == "active" and "uv run pytest" in lesson.behavior
    assert _store().signal_summary(lesson.experience_id).user_lineages <= {"day1", "day2"}


def test_a_standing_wish_holds_at_once_and_a_permission_is_never_learned(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "以後跑測試都要用 uv run pytest")
        chat.turn("s1", "以後推 repo 不用問我")

    (lesson,) = _live()
    assert (lesson.status, lesson.kind, lesson.scope, lesson.task_kind) == ("active", "procedure", "project",
                                                                            "engineering")
    assert lesson.situation == 'engineering task "跑測試"' and lesson.confirmed_at is not None


def test_plain_chat_teaches_nothing(tmp_path, monkeypatch):
    with Chat(tmp_path, monkeypatch, _scripted(Text("好的。"))) as chat:
        chat.turn("s1", "以後都用繁體中文回答")  # no work going on: a wish for long-term memory, not a lesson

    assert _store().experiences() == []


def test_only_the_owners_own_words_teach(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("收到。")) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "[System note: background job finished] 以後跑測試都要用 uv run pytest")

    assert _store().experiences() == []


def test_the_same_words_said_twice_in_one_conversation_count_once(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，要用 uv run pytest")
        chat.turn("s1", "不對，要用 uv run pytest")

    (lesson,) = _live()
    assert lesson.status == "candidate"  # the same conversation repeating itself is not a second voice
    assert [s.source_class for s in _store().signals(lesson.experience_id)] == ["user_correction"]


def test_a_lesson_the_model_proposes_waits_for_the_users_own_confirmation(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("uv run pytest -q")),
                   _lesson(op="propose", situation="running this project's tests", behavior="use `uv run pytest`"),
                   Text("測試通過，我記下了做法。")]
        chat.turn("s1", "幫我跑測試")
        (proposed,) = _live()
        assert (proposed.status, proposed.origin.authored_by) == ("candidate", "model")

        script += [_lesson(op="confirm", experience_id=proposed.experience_id), Text("好。")]
        chat.turn("s1", "繼續")  # the model cannot confirm on its own
        assert _store().experience(proposed.experience_id).status == "candidate"

        chat.turn("s1", "對，以後都這樣做")

    assert _store().experience(proposed.experience_id).status == "active"


def test_the_user_withdraws_a_lesson_and_it_ends(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "以後跑測試都要用 uv run pytest")
        (lesson,) = _live()
        chat.turn("s1", "那條規則不對，忘掉它")

    assert _store().experience(lesson.experience_id).status == "revoked"
    assert _live() == []


def test_learning_off_keeps_nothing(tmp_path, monkeypatch):
    off = _ON.replace("learning: true", "learning: false").replace("calibration: true", "calibration: false")
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。"), config=off) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "以後跑測試都要用 uv run pytest")

    assert (_store().experiences(), _store().outcomes()) == ([], [])


# ------------------------------------------------------------------ track record


def test_only_our_own_check_counts_and_a_repeated_check_counts_once(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"),
                   _terminal(chat.cmd("pytest -q")), Text("測試通過。"),
                   _terminal(chat.cmd("claude -p 'fix the flaky test'")), Text("Claude 回報測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "再跑一次")
        chat.turn("s1", "請 claude 修一下不穩的測試")
        ours = [o for o in _store().outcomes() if o.actor_kind == "self"]
        (report,) = [o for o in _store().outcomes() if o.actor_kind == "external_agent"]
        assert [(o.claim_class, o.result, o.actor_id) for o in ours] == [("tests_pass", "success", "fake-model")]
        assert (report.actor_id, report.claim_class, report.result) == ("claude", "tests_pass", "reported")

        script += [_terminal(chat.cmd("pytest -q")), Text("我自己跑過了，測試通過。")]
        chat.turn("s1", "你自己跑一次確認")

    assert _store().outcomes(actor_kind="external_agent")[0].result == "success"  # our own check confirmed it
    assert len(_store().outcomes(actor_kind="self", results=("success",))) == 1  # the same check, counted once


def test_the_users_correction_marks_the_last_claim_corrected(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("抱歉，我再查。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，測試根本沒過")

    (outcome,) = _store().outcomes(actor_kind="self")
    assert (outcome.result, outcome.prior_result, outcome.intervention.get("corrections")) == ("corrected", "success", 1)
    assert _live() == []  # saying it is wrong says nothing about how to do it
