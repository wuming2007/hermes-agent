"""Behavior contract: the owner's corrections and wishes become scoped, traceable lessons, and
claimed results become a track record that only our own checks improve.

Real agent loop, real plugin discovery and the real terminal tool against a scripted provider,
in the test's isolated HERMES_HOME. Commands are tiny stand-in scripts, so nothing real runs.
"""

import os
import re
import stat
import time
from pathlib import Path
from typing import Callable

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_ON = ("plugins:\n  enabled: [cognition_wm]\n  entries:\n    cognition_wm:\n      settings:\n"
       "        learning: true\n        calibration: true\n"
       "cognition:\n  enabled: true\nauxiliary:\n  title_generation:\n    enabled: false\n")
_TOOLSETS = ["file", "terminal", "cognition_wm"]


def _scripted(*responses) -> Callable:
    queue = list(responses)
    return lambda _record: queue.pop(0) if queue else Text("好。")


def _play(script: list) -> Callable:
    """Scripted replies in order; a callable reply is built from the request it answers."""
    def respond(record):
        step = script.pop(0) if script else Text("好。")
        return step(record) if callable(step) else step
    return respond


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
    # the rerun before the agent counted once; the check after it (it may have changed files) is new evidence
    assert len(_store().outcomes(actor_kind="self", results=("success",))) == 2


def test_the_users_correction_marks_the_last_claim_corrected(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("抱歉，我再查。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，測試根本沒過")

    (outcome,) = _store().outcomes(actor_kind="self")
    assert (outcome.result, outcome.prior_result, outcome.intervention.get("corrections")) == ("corrected", "success", 1)
    assert _live() == []  # saying it is wrong says nothing about how to do it


def test_correcting_how_it_was_done_keeps_a_checked_result_and_counts_the_users_effort(tmp_path, monkeypatch):
    """A correction of the approach ("要用 uv run pytest") says how the user wants it done, not that
    a result our own check showed was false: the success stands, the correction is its cost."""
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("好，改用 uv run pytest。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，要用 uv run pytest，不是直接 pytest")

    (outcome,) = _store().outcomes(actor_kind="self")
    assert (outcome.result, outcome.prior_result, outcome.intervention.get("corrections")) == ("success", None, 1)


def test_a_check_rerun_under_new_tasks_counts_once_until_something_changes(tmp_path, monkeypatch):
    """Found with the real model: asked to run the tests again, it opened a new task each time, and
    each rerun of the same check counted as another success. A rerun counts once however the work is
    split into tasks; a check after a change is new evidence."""
    script = []
    with Chat(tmp_path, monkeypatch, lambda r: script.pop(0) if script else Text("好。")) as chat:
        for title in ("跑測試", "再確認一次", "最後確認"):
            script += [_task(title), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "再跑一次確認")
        chat.turn("s1", "最後再跑一次")
        script += [ToolCall("write_file", {"path": "app.py", "content": "x = 2\n"}), _terminal(chat.cmd("pytest -q")),
                   Text("改完了，測試都通過了。")]
        chat.turn("s1", "把 app.py 的 x 改成 2，再跑測試")

    passed = [o for o in _store().outcomes(actor_kind="self", results=("success",)) if o.claim_class == "tests_pass"]
    assert len(passed) == 2  # the first check, and the one after the change


# ------------------------------------------------------------------ review round (2026-09-27)


def test_the_opposite_wish_replaces_a_lesson_and_never_counts_for_it(tmp_path, monkeypatch):
    """"Never use X" after "Always use X" is a new wish that replaces the old one, not the old one said again."""
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "Always use uv run pytest.")
        (always,) = _live()
        chat.turn("s1", "Never use uv run pytest.")

    (never,) = _live()
    assert never.experience_id != always.experience_id and never.behavior.startswith("Never")
    assert _store().experience(always.experience_id).status == "superseded"
    assert [s.polarity for s in _store().signals(always.experience_id)] == [1]  # only its own statement


def test_a_confirmation_settles_only_the_lesson_it_answers(tmp_path, monkeypatch):
    """The user's "yes" confirms the lesson in question; the model cannot spend it on a lesson it
    proposes in the same turn."""
    def confirm_own(record):
        found = re.findall(r"xp_[0-9a-f]{12}", str(record["body"]["messages"][-1].get("content")))
        return _lesson(op="confirm", experience_id=found[-1])

    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("好，改用 uv run pytest。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，要用 uv run pytest")
        (asked,) = _live()
        script += [_lesson(op="propose", situation="linting", behavior="run `make lint` before `make test`"),
                   confirm_own, Text("好，都記下了。")]
        chat.turn("s1", "對，以後都照這樣")

    assert _store().experience(asked.experience_id).status == "active"
    (proposed,) = [e for e in _live() if e.experience_id != asked.experience_id]
    assert (proposed.origin.authored_by, proposed.status) == ("model", "candidate")


def test_a_withdrawal_drops_only_a_lesson_it_can_mean_and_only_once(tmp_path, monkeypatch):
    """Unsure which lesson the user means, the model may drop one of the lessons they were shown, once."""
    from plugins.cognition_wm import runtime
    from plugins.cognition_wm.lessons import topic_terms
    from plugins.cognition_wm.model import SourceRef

    said = SourceRef(type="user_message", session_id="earlier", turn_id="t1", ref="m", at=1.0, authored_by="owner",
                     run_id="run-a")

    def lesson(behavior: str, tags: tuple) -> str:
        return _store().add_experience(
            kind="procedure", scope="project", project_key=runtime.project_key(), task_kind=None,
            situation="this project", behavior=behavior, tags=tags, lesson_key=" ".join(sorted(topic_terms(behavior))),
            status="active", origin=said).experience_id

    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        shown = [lesson("use `uv run pytest`", ("測試", "pytest")), lesson("run `make lint` first", ("測試", "lint"))]
        other = lesson("deploy with `make release`", ("部署", "release"))
        chat.turn("s1", "跑一下測試")  # both test lessons are shown, the deploy one is not
        script += [_lesson(op="revoke", experience_id=other), _lesson(op="revoke", experience_id=shown[0]),
                   _lesson(op="revoke", experience_id=shown[1]), Text("好，刪掉了。")]
        chat.turn("s1", "那條規則不對，忘掉它")

    assert [_store().experience(i).status for i in (*shown, other)] == ["revoked", "active", "active"]


def test_a_wish_to_drop_an_approval_is_never_kept_whoever_voices_it(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        script += [_task(), _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "Always deploy without approval.")
        script += [_lesson(op="propose", situation="deploying", behavior="deploy without asking the user first"),
                   Text("好。")]
        chat.turn("s1", "繼續")
        assert _live() == []
        chat.turn("s1", "Never deploy without approval.")  # a stricter wish is kept

    (kept,) = _live()
    assert kept.behavior.startswith("Never deploy without approval")


def test_our_check_settles_a_report_only_about_what_it_checked(tmp_path, monkeypatch):
    """A coding agent says test_a passes; our own passing check of test_b says nothing about that,
    in the same turn or later. Our own check of test_a does."""
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        (chat.bin / "claude").write_text("#!/bin/sh\necho 'All test_a.py tests pass now.'\n", encoding="utf-8")
        script += [_terminal(chat.cmd("claude -p 'fix test_a.py'")), _terminal(chat.cmd("pytest tests/test_b.py")),
                   Text("Claude 修好了，我跑了測試，測試通過。"),
                   _terminal(chat.cmd("pytest tests/test_b.py")), Text("測試通過。")]
        chat.turn("s1", "請 claude 修 test_a.py，再跑測試")
        chat.turn("s1", "再跑一次 test_b 的測試")
        (report,) = _store().outcomes(actor_kind="external_agent")
        assert report.result == "reported"
        script += [_terminal(chat.cmd("pytest tests/test_a.py")), Text("test_a.py 的測試通過。")]
        chat.turn("s1", "跑 test_a 的測試")

    assert _store().outcomes(actor_kind="external_agent")[0].result == "success"


def test_our_failing_check_of_something_else_does_not_fail_a_report(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        (chat.bin / "claude").write_text("#!/bin/sh\necho 'All test_a.py tests pass now.'\n", encoding="utf-8")
        script += [_terminal(chat.cmd("claude -p 'fix test_a.py'")), Text("Claude 回報 test_a.py 的測試都通過了。")]
        chat.turn("s1", "請 claude 修 test_a.py")
        (chat.bin / "pytest").write_text("#!/bin/sh\necho '1 failed'\nexit 1\n", encoding="utf-8")
        script += [_terminal(chat.cmd("pytest tests/test_b.py")), Text("測試通過了。")]
        chat.turn("s1", "跑 test_b 的測試")

    assert _store().outcomes(actor_kind="external_agent")[0].result == "reported"


@pytest.mark.parametrize("correction, reads", [
    ("不對，測試根本沒過", ("corrected", "success", 1)),  # the result did not hold
    ("不對，要用 uv run pytest", ("success", None, 1)),  # how it was done: the user's effort
])
def test_a_correction_after_a_rerun_reaches_the_record_of_that_check(tmp_path, monkeypatch, correction, reads):
    """The rerun counted once, on the first turn's record; the user's word on it lands there."""
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"),
                   _terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("抱歉，我再確認。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "再跑一次")
        chat.turn("s1", correction)

    (outcome,) = _store().outcomes(actor_kind="self")
    assert (outcome.result, outcome.prior_result, outcome.intervention.get("corrections")) == reads


def test_checks_of_different_files_are_different_checks(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        script += [_terminal(chat.cmd("pytest tests/test_api_v1.py")), Text("測試都通過了。"),
                   _terminal(chat.cmd("pytest tests/test_api_v2.py")), Text("測試都通過了。")]
        chat.turn("s1", "跑 v1 的測試")
        chat.turn("s1", "跑 v2 的測試")

    assert len(_store().outcomes(actor_kind="self", results=("success",))) == 2


def test_a_rerun_that_differs_only_in_its_timing_counts_once(tmp_path, monkeypatch):
    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        (chat.bin / "pytest").write_text('#!/bin/sh\necho "5 passed in 0.$$s"\n', encoding="utf-8")
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"),
                   _terminal(chat.cmd("pytest -q")), Text("測試都通過了。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "再跑一次")

    assert len(_store().outcomes(actor_kind="self", results=("success",))) == 1


def test_a_long_running_process_keeps_its_limits_checking_at_most_hourly(tmp_path, monkeypatch):
    """Upkeep runs again while the process keeps running, not only at its start, and not every turn:
    at most once an hour."""
    from plugins.cognition_wm import runtime
    from plugins.cognition_wm.model import SourceRef

    script = []
    with Chat(tmp_path, monkeypatch, _play(script)) as chat:
        now = [time.time()]
        _store()._clock = lambda: now[0]
        script += [_terminal(chat.cmd("pytest -q")), Text("測試都通過了。"), Text("好。")]
        chat.turn("s1", "幫我跑測試")
        chat.turn("s1", "不對，要用 uv run pytest")
        (first,) = _live()
        now[0] += 31 * 86400  # a month later, in the same process
        chat.turn("s1", "今天天氣如何？")
        assert _store().experience(first.experience_id).status == "expired"

        upkept_at, now[0] = now[0], now[0] - 30 * 86400 + 600  # a lesson that turns 30 days old ten minutes on
        said = SourceRef(type="user_message", session_id="earlier", turn_id="t1", ref="m", at=now[0],
                         authored_by="owner", run_id="run-a")
        second = _store().add_experience(kind="procedure", scope="project", project_key=runtime.project_key(),
                                         task_kind=None, situation="this project", behavior="use `make test`",
                                         status="candidate", origin=said)
        now[0] = upkept_at + 1200
        chat.turn("s1", "那明天呢？")
        within_the_hour = _store().experience(second.experience_id).status
        now[0] = upkept_at + 3700
        chat.turn("s1", "謝謝")

    assert (within_the_hour, _store().experience(second.experience_id).status) == ("candidate", "expired")
