"""Behavior contract: a lesson rides the current user message only where it applies — this project,
work like the one it came from, the model and version it holds for — and never once it ended;
followed and shown to work, it grows stronger.

Real agent loop, plugin discovery and terminal tool against a scripted provider, in the test's
isolated HERMES_HOME. Commands are stand-in scripts.
"""

import json
import os
import stat
from pathlib import Path

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

    def __init__(self, tmp_path: Path, monkeypatch, name: str = "work", config: str = _ON):
        self.tmp_path, self.monkeypatch, self.config = tmp_path, monkeypatch, config
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
        write_hermes_home(self.home, self.server.base_url, extra_config=self.config)
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


def test_before_any_task_a_request_for_other_work_gets_no_lesson(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        project.turn("day2", "幫我寫週報，順便列出測試相關的進度")  # shares 測試, but it is writing
        row = project.last_user_row()

    assert "<lessons" not in row


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


# ------------------------------------------------------------------ review round (2026-09-27)


def _owner_lesson(behavior: str, *, status: str, tags: tuple, exceptions: str = ""):
    from plugins.cognition_wm import runtime
    from plugins.cognition_wm.lessons import topic_terms
    from plugins.cognition_wm.model import SourceRef

    said = SourceRef(type="user_message", session_id="earlier", turn_id="t1", ref="m", at=1.0, authored_by="owner",
                     run_id="run-a")
    return _store().add_experience(
        kind="procedure", scope="project", project_key=runtime.project_key(), task_kind=None,
        situation="running the tests", behavior=behavior, exceptions=exceptions, tags=tags,
        lesson_key=" ".join(sorted(topic_terms(behavior))), status=status, origin=said)


def test_only_lessons_shown_whole_count_as_shown(tmp_path, monkeypatch):
    """A lesson that does not fit is left out whole, never cut off before its exception, and only
    lessons actually shown count as seen: the user's "yes" cannot confirm one they never saw."""
    tight = _ON.replace("        calibration: true\n", "        calibration: true\n        lessons_max_chars: 220\n")
    with Project(tmp_path, monkeypatch, config=tight) as project:
        short = _owner_lesson("use `uv run pytest`", status="active", tags=("測試", "pytest"))
        long = _owner_lesson("run the whole suite with `uv run pytest -q` and read its summary line before saying "
                             "anything about the tests", status="candidate", tags=("測試", "suite"),
                             exceptions="except docs-only changes, where a spot check is enough")
        project.turn("day2", "跑一下測試")
        row = project.last_user_row()
        project.turn("day2", "對，以後都這樣做")

    block = row[row.index("<lessons"):]
    assert "…" not in block and short.experience_id in block and long.experience_id not in block
    assert _store().experience(long.experience_id).status == "candidate"


def test_a_command_only_echoed_is_not_following_a_lesson(tmp_path, monkeypatch):
    """Printing `uv run pytest` is not running it: no credit, and no promotion across conversations."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        for day in ("day2", "day3"):
            project.script += [_task(f"跑測試 {day}"), ToolCall("terminal", {"command": "echo 'uv run pytest'"}),
                               Text("測試都通過了。")]
            project.turn(day, "跑一下測試")

    assert _store().signal_summary(lesson.experience_id).verifications == 0
    assert _store().experience(lesson.experience_id).status == "candidate"


def test_running_the_command_a_lesson_replaces_is_not_following_it(tmp_path, monkeypatch):
    with Project(tmp_path, monkeypatch) as project:
        project.script += [project.run("pytest -q"), Text("測試都通過了。"), Text("好。")]
        project.turn("day1", "幫我跑測試")
        project.turn("day1", "不對，要用 `uv run pytest` 而不是 `pytest`")
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"), project.run("pytest -q"), Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")

    assert _store().signal_summary(lesson.experience_id).verifications == 0


def test_running_a_lessons_command_earns_credit_only_with_a_result_that_held(tmp_path, monkeypatch):
    """Exit status 0 and no failure are not enough: the run must back a result of ours that held."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"), project.run("uv run pytest -q"), Text("輸出在上面。")]
        project.turn("day2", "跑一下測試")
        quiet = _store().signal_summary(lesson.experience_id).verifications
        project.script += [_task("跑測試 day3"), project.run("uv run pytest -q"), Text("測試都通過了。")]
        project.turn("day3", "跑一下測試")

    assert (quiet, _store().signal_summary(lesson.experience_id).verifications) == (0, 1)


def test_a_kept_lesson_is_not_copied_into_memory_or_a_skill_even_without_its_markers(tmp_path, monkeypatch):
    """A copy in long-term memory or a skill would outlive the lesson's revocation. A write that
    repeats a kept lesson word for word is refused; a reworded copy is not caught (see the docs)."""
    from plugins.cognition_wm import scope_gate

    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        copy = f"User preference: {lesson.behavior}"
        memory = scope_gate.on_pre_tool_call(tool_name="memory", args={"action": "add", "content": copy})
        skill = scope_gate.on_pre_tool_call(tool_name="skill_manage", args={
            "action": "create", "name": "run-tests", "content": f"# Running tests\n\n{copy}\n"})
        unrelated = scope_gate.on_pre_tool_call(tool_name="memory", args={"action": "add", "content": "User lives in Taipei."})

    assert (memory or {}).get("action") == (skill or {}).get("action") == "block"
    assert unrelated is None


def test_a_withdrawn_lesson_is_not_copied_into_memory_word_for_word_either(tmp_path, monkeypatch):
    """A revoked lesson stays in the store as history; copied word for word into memory it would come back."""
    from plugins.cognition_wm import scope_gate

    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.turn("day1", "那條規則不對，忘掉它")
        copy = scope_gate.on_pre_tool_call(tool_name="memory", args={"action": "add",
                                                                     "content": f"User preference: {lesson.behavior}"})

    assert _store().experience(lesson.experience_id).status == "revoked"
    assert (copy or {}).get("action") == "block"


def test_a_negated_run_is_neither_a_passing_run_nor_following_a_lesson(tmp_path, monkeypatch):
    """`! uv run pytest` exits 0 because the tests failed."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        (project.bin / "uv").write_text("#!/bin/sh\necho '1 failed'\nexit 1\n", encoding="utf-8")
        project.script += [_task("跑測試 day2"), ToolCall("terminal", {"command": f"! {project.bin}/uv run pytest -q"}),
                           Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")

    assert _store().signal_summary(lesson.experience_id).verifications == 0
    assert [o.result for o in _store().outcomes(session_id="day2", actor_kind="self") if o.claim_class == "tests_pass"] \
        == ["unknown"]


def test_a_lessons_command_only_printed_beside_another_run_is_not_following_it(tmp_path, monkeypatch):
    """The line's own test run backs the result; the lesson's command only appears in what it printed."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"),
                           ToolCall("terminal", {"command": f"{project.bin}/pytest -q && echo 'uv run pytest next time'"}),
                           Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")

    assert _store().signal_summary(lesson.experience_id).verifications == 0


def test_a_lessons_run_negated_behind_time_is_not_following_it(tmp_path, monkeypatch):
    """`time ! uv run pytest -q` exits 0 because the tests failed; `time uv run pytest -q` that passes is
    following the lesson."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        (project.bin / "uv").write_text("#!/bin/sh\necho '1 failed'\nexit 1\n", encoding="utf-8")
        project.script += [_task("跑測試 day2"),
                           ToolCall("terminal", {"command": f"bash -c 'time ! {project.bin}/uv run pytest -q'"}),
                           Text("測試都通過了。")]
        project.turn("day2", "跑一下測試")
        negated = (_store().signal_summary(lesson.experience_id).verifications,
                   [o.result for o in _store().outcomes(session_id="day2", actor_kind="self")
                    if o.claim_class == "tests_pass"])
        (project.bin / "uv").write_text("#!/bin/sh\necho '5 passed'\n", encoding="utf-8")
        project.script += [_task("跑測試 day3"),
                           ToolCall("terminal", {"command": f"bash -c 'time {project.bin}/uv run pytest -q'"}),
                           Text("測試都通過了。")]
        project.turn("day3", "跑一下測試")

    assert negated == (0, ["unknown"])
    assert _store().signal_summary(lesson.experience_id).verifications == 1


def test_a_file_only_printed_beside_a_lessons_run_is_no_result_to_credit_it_with(tmp_path, monkeypatch):
    """The lesson's command tested test_a.py; test_b.py was only printed, so "both pass" did not hold."""
    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        project.script += [_task("跑測試 day2"),
                           ToolCall("terminal", {"command": f"{project.bin}/uv run pytest tests/test_a.py && echo test_b.py"}),
                           Text("test_a.py and test_b.py tests pass.")]
        project.turn("day2", "跑一下測試")
        printed = _store().signal_summary(lesson.experience_id).verifications
        project.script += [_task("跑測試 day3"), ToolCall("terminal", {
            "command": f"{project.bin}/uv run pytest tests/test_a.py && {project.bin}/uv run pytest tests/test_b.py"}),
                           Text("test_a.py and test_b.py tests pass.")]
        project.turn("day3", "跑一下測試")

    assert (printed, _store().signal_summary(lesson.experience_id).verifications) == (0, 1)


def test_a_test_command_only_printed_beside_a_lessons_run_is_no_result_to_credit_it_with(tmp_path, monkeypatch):
    """`echo pytest \\;` prints "pytest ;", so "the tests pass" did not hold, and the lint lesson run
    beside it has no result of ours to rest on."""
    with Project(tmp_path, monkeypatch) as project:
        ruff = project.bin / "ruff"
        ruff.write_text("#!/bin/sh\necho 'All checks passed!'\n", encoding="utf-8")
        ruff.chmod(ruff.stat().st_mode | stat.S_IEXEC)
        project.script += [project.run("ruff check ."), Text("lint 檢查通過了。"), Text("好，改用 uv run ruff check。")]
        project.turn("day1", "幫我跑 lint 檢查")
        project.turn("day1", "不對，要用 uv run ruff check，不是直接 ruff")
        (lesson,) = _store().experiences()
        project.script += [_task("跑 lint 檢查 day2"),
                           ToolCall("terminal", {"command": f"{project.bin}/uv run ruff check . && echo pytest \\;"}),
                           Text("測試都通過了。")]
        project.turn("day2", "幫我跑 lint 檢查")
        shown = "<lessons" in project.last_user_row()

    assert shown
    assert [o.result for o in _store().outcomes(session_id="day2", actor_kind="self")
            if o.claim_class == "tests_pass"] == ["unknown"]
    assert _store().signal_summary(lesson.experience_id).verifications == 0


def test_a_background_jobs_command_is_no_result_to_credit_a_lesson_with(tmp_path, monkeypatch):
    """The lesson's command tested test_a.py in the background; test_b.py is only in the job's command,
    beside what it printed, so "both pass" did not hold. Running both earns the credit."""
    def wait_on_the_job(record) -> ToolCall:
        started = next(json.loads(str(m.get("content"))) for m in reversed(record["body"]["messages"])
                       if m.get("role") == "tool")
        return ToolCall("tool_call", {"calls": [{"name": "process_manage", "arguments": {
            "action": "wait", "session_id": started["session_id"], "timeout": 30}}]})

    with Project(tmp_path, monkeypatch) as project:
        project.teach_correction()
        (lesson,) = _store().experiences()
        uv = f"{project.bin}/uv run pytest"
        project.script += [_task("跑測試 day2"),
                           ToolCall("terminal", {"command": f"{uv} tests/test_a.py && : tests/test_b.py", "background": True}),
                           wait_on_the_job, Text("test_a.py and test_b.py tests pass.")]
        project.turn("day2", "跑一下測試")
        named = _store().signal_summary(lesson.experience_id).verifications
        project.script += [_task("跑測試 day3"),
                           ToolCall("terminal", {"command": f"{uv} tests/test_a.py tests/test_b.py", "background": True}),
                           wait_on_the_job, Text("test_a.py and test_b.py tests pass.")]
        project.turn("day3", "跑一下測試")

    assert (named, _store().signal_summary(lesson.experience_id).verifications) == (0, 1)
