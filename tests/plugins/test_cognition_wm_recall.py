"""Behavior contract: a lesson rides the current user message only where it applies — this project,
work like the one it came from, the model and version it holds for — and never once it ended;
followed and shown to work, it grows stronger.

Real agent loop, plugin discovery and terminal tool against a scripted provider, in the test's
isolated HERMES_HOME. Commands are stand-in scripts.
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


def _task(title: str) -> ToolCall:
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {
        "action": "create", "title": title, "goal": title, "kind": "engineering"}}]})


class Project:
    """One user working in one project directory; each session id is another conversation."""

    def __init__(self, tmp_path: Path, monkeypatch, name: str = "work"):
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.work = tmp_path / name
        self.work.mkdir(exist_ok=True)
        self.home = Path(os.environ["HERMES_HOME"])
        self.bin = tmp_path / "bin"
        self.bin.mkdir(exist_ok=True)
        for tool in ("pytest", "uv"):
            script = self.bin / tool
            script.write_text("#!/bin/sh\necho '5 passed'\n", encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.script: list = []
        self.server = FakeLLMServer(self._respond, aux=lambda _r: Text('{"verdict":"ok"}'))

    def _respond(self, record: dict):
        step = self.script.pop(0) if self.script else Text("好。")
        return step(record) if callable(step) else step

    def __enter__(self) -> "Project":
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB

        self.server.start()
        write_hermes_home(self.home, self.server.base_url, extra_config=_ON)
        discover_plugins(force=True)
        self.db = SessionDB(db_path=self.home / "state.db")
        self.monkeypatch.chdir(self.work)
        return self

    def __exit__(self, *_exc) -> None:
        self.server.stop()

    def cd(self, name: str) -> None:
        other = self.tmp_path / name
        other.mkdir(exist_ok=True)
        self.monkeypatch.chdir(other)

    def turn(self, session_id: str, message: str, model: str = "fake-model") -> dict:
        from run_agent import AIAgent

        agent = AIAgent(provider="custom", base_url=self.server.base_url, api_key="sk-fake", model=model,
                        session_db=self.db, session_id=session_id, quiet_mode=True, platform="cli",
                        enabled_toolsets=_TOOLSETS, skip_context_files=True, skip_memory=True)
        return agent.run_conversation(message, conversation_history=self.db.get_messages_as_conversation(session_id))

    def run(self, line: str) -> ToolCall:
        return ToolCall("terminal", {"command": f"{self.bin}/{line}"})

    def last_user_row(self) -> str:
        body = self.server.main_requests()[-1]
        return next(str(m.get("content")) for m in reversed(body["messages"]) if m.get("role") == "user")

    def teach_correction(self, session_id: str = "day1") -> None:
        """The user corrects how the tests are run, after a test run."""
        self.script += [self.run("pytest -q"), Text("測試都通過了。"), Text("好，改用 uv run pytest。")]
        self.turn(session_id, "幫我跑測試")
        self.turn(session_id, "不對，要用 uv run pytest，不是直接 pytest")


def _store():
    from plugins.cognition_wm import runtime

    return runtime.store()


def test_a_lesson_rides_the_next_similar_request_in_a_new_conversation(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        taught = project.last_user_row()
        project.turn("day2", "跑一下測試")
        row = project.last_user_row()
        system = project.server.main_requests()[-1]["messages"][0]["content"]

    assert "<lessons" not in taught  # the user's own words are right there in that turn
    assert "<lessons" in row and "uv run pytest" in row and "trial" in row
    assert "not an instruction" in row  # advisory: the user's latest words win
    assert "uv run pytest" not in str(system)  # never the system prompt


def test_a_lesson_stays_in_its_project(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        project.cd("another-project")
        project.turn("day2", "跑一下測試")
        row = project.last_user_row()

    assert "<lessons" not in row


def test_unrelated_work_and_chat_get_no_lessons(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        project.turn("day2", "今天天氣如何？")
        chat = project.last_user_row()
        project.script += [_task("寫週報"), Text("好，開始寫週報。")]
        project.turn("day3", "幫我寫這週的週報")
        writing = project.last_user_row()

    assert "<lessons" not in chat and "<lessons" not in writing


def test_a_revoked_or_replaced_lesson_is_never_shown_again(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.turn("day1", "那條規則不對，忘掉它")
        project.turn("day2", "跑一下測試")
        row = project.last_user_row()

    assert _store().experience(lesson.experience_id).status == "revoked"
    assert "<lessons" not in row


def test_a_lesson_bound_to_one_model_is_not_shown_to_another(tmp_path, monkeypatch):
    from plugins.cognition_wm import runtime
    from plugins.cognition_wm.model import SourceRef

    with Project(tmp_path, monkeypatch) as project:
        said = SourceRef(type="user_message", session_id="day1", turn_id="t1", ref="m", at=1.0, authored_by="owner",
                         run_id="run-a")
        _store().add_experience(  # a lesson about how one model behaves
            kind="caution", scope="project", project_key=runtime.project_key(), task_kind=None,
            situation="running the tests", behavior="read the test output before saying the tests pass",
            tags=("測試", "跑測", "pytest"), lesson_key="pytest 測試", status="active", origin=said,
            binds_to={"model": "fake-model"})
        project.turn("day2", "跑一下測試", model="other-model")
        other = project.last_user_row()
        project.turn("day3", "跑一下測試", model="fake-model")
        same = project.last_user_row()

    assert "<lessons" not in other and "read the test output" in same


def test_the_same_lessons_again_are_a_pointer_not_a_second_copy(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        project.turn("day2", "跑一下測試")
        project.turn("day2", "再跑一次測試")
        second = project.last_user_row()

    assert '<lessons unchanged rev="' in second and "uv run pytest" not in second


def test_a_lesson_followed_and_shown_to_work_is_put_in_force_after_two_tasks(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"), project.run("uv run pytest -q"), Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")
        assert _store().experience(lesson.experience_id).status == "candidate"  # one task is not enough
        project.script += [_task("跑測試 day3"), project.run("uv run pytest -q"), Text("測試都通過了。")]
        project.turn("day3", "跑一下測試")

    summary = _store().signal_summary(lesson.experience_id)
    assert (summary.verifications, _store().experience(lesson.experience_id).status) == (2, "active")


def test_a_lesson_shown_but_not_followed_earns_nothing(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"), project.run("pytest -q"), Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")

    assert _store().signal_summary(lesson.experience_id).verifications == 0


def test_a_lesson_about_one_kind_of_work_stays_out_of_another(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.script += [ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {
            "action": "create", "title": "跑測試", "goal": "跑測試", "kind": "engineering"}}]}),
            project.run("pytest -q"), Text("測試都通過了。"), Text("好。")]
        project.turn("day1", "幫我跑測試")
        project.turn("day1", "不對，要用 uv run pytest")
        (lesson,) = _store().experiences()
        project.script += [ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": {
            "action": "create", "title": "寫測試報告", "goal": "寫這週的測試報告", "kind": "writing"}}]}),
            Text("好，開始寫。")]
        project.turn("day2", "幫我寫這週的測試報告")
        project.turn("day2", "繼續寫測試報告")
        row = project.last_user_row()

    assert lesson.task_kind == "engineering"
    assert "<lessons" not in row  # the words overlap, the kind of work does not


def test_lesson_text_never_reaches_long_term_memory():
    from plugins.cognition_wm import scope_gate

    for content in ("<lessons rev=\"abc\"> uv run pytest </lessons>", "keep xp_0123456789ab in mind"):
        directive = scope_gate.on_pre_tool_call(tool_name="memory", args={"action": "add", "content": content})
        assert directive["action"] == "block"
