"""Acceptance scenarios for cognition v2 phase 2: the same scripted model and user, phase 1 against
the full v2 (phase 1 plus learning and calibration).

Everything is real except the model: AIAgent (the CLI core), SessionDB on disk, the terminal tool,
compaction, plugin discovery and the scope gate, in the test's isolated HERMES_HOME. Commands are
stand-in scripts that log each run. The scripted model follows a lesson when the request carries
one, so a scenario shows that a lesson reaching the request changes what actually runs; whether a
real model follows it is the real-model evaluation's question, not this file's.
"""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from typing import Callable

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_BASE = ("compression:\n  protect_last_n: 4\nauxiliary:\n  title_generation:\n    enabled: false\n"
         "cognition:\n  enabled: true\n  evidence_loop:\n    enabled: true\n    max_extra_tool_calls: 2\n")
_PHASE1 = _BASE + "plugins:\n  enabled: [cognition_wm]\n"
_V2 = _PHASE1 + ("  entries:\n    cognition_wm:\n      settings:\n        learning: true\n"
                 "        calibration: true\n")
_TOOLSETS = ["file", "terminal", "cognition_wm"]
_VERIFIER = "You are a strict consistency verifier."
_SUMMARY = "## Goal\nKeep helping with the user's task.\n## Progress\n### Done\n- Ran the tests.\n## Next Steps\n- Continue.\n"


def _aux(body: dict) -> Text:
    return Text('{"verdict":"ok"}') if _VERIFIER in str(body.get("messages")) else Text(_SUMMARY)


def _user_rows(body: dict) -> list[str]:
    return [str(m.get("content")) for m in body["messages"] if m.get("role") == "user"]


def _after_tool(body: dict) -> bool:
    return body["messages"][-1].get("role") == "tool"


def _task_state(**arguments) -> ToolCall:
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": arguments}]})


class World:
    """One user, their projects and conversations under one Hermes home; ``v2`` adds learning and
    calibration to phase 1."""

    def __init__(self, tmp_path: Path, monkeypatch, *, v2: bool, respond: Callable[["World", dict], object]):
        self.v2, self.monkeypatch = v2, monkeypatch
        self.root = tmp_path / ("v2" if v2 else "phase1")
        self.home, self.bin = self.root / "home", self.root / "bin"
        self.bin.mkdir(parents=True)
        self.log = self.root / "runs.log"
        outputs = {"pytest": "5 passed", "uv": "5 passed", "claude": "All tests pass now.", "deploy.sh": "deployed"}
        for name, output in outputs.items():
            script = self.bin / name
            script.write_text(f'#!/bin/sh\necho "$(basename "$0") $*" >> "{self.log}"\necho "{output}"\n',
                              encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("HERMES_HOME", str(self.home))
        self.server = FakeLLMServer(lambda record: respond(self, record["body"]),
                                    aux=lambda record: _aux(record["body"]))
        self.model = "fake-model"
        self._turn_start = 0

    def __enter__(self) -> "World":
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB

        self.server.start()
        write_hermes_home(self.home, self.server.base_url, extra_config=_V2 if self.v2 else _PHASE1)
        discover_plugins(force=True)
        self.db = SessionDB(db_path=self.home / "state.db")
        self.cd("work")
        return self

    def __exit__(self, *_exc) -> None:
        self.server.stop()

    def cd(self, project: str) -> None:
        (self.root / project).mkdir(exist_ok=True)
        self.monkeypatch.chdir(self.root / project)

    def agent(self, session_id: str):
        from run_agent import AIAgent

        return AIAgent(provider="custom", base_url=self.server.base_url, api_key="sk-fake", model=self.model,
                       session_db=self.db, session_id=session_id, quiet_mode=True, platform="cli",
                       enabled_toolsets=_TOOLSETS, skip_context_files=True, skip_memory=True)

    def turn(self, session_id: str, message: str) -> dict:
        history = self.db.get_messages_as_conversation(session_id)
        self._turn_start = len(self.server.main_requests())
        return self.agent(session_id).run_conversation(message, conversation_history=history)

    def turn_row(self) -> str:
        """The user row the latest turn opened with, as sent (a check round comes later)."""
        return _user_rows(self.server.main_requests()[self._turn_start])[-1]

    def turn_sent(self) -> str:
        """Every user row the latest turn sent: what the model had in view, history included."""
        return "\n".join(_user_rows(self.server.main_requests()[self._turn_start]))

    def compress(self, session_id: str) -> None:
        from tui_gateway.server import _compress_session_history

        session = {"agent": self.agent(session_id), "history": self.db.get_messages_as_conversation(session_id),
                   "history_lock": threading.Lock(), "history_version": 1, "session_key": session_id}
        _compress_session_history(session, "")

    def runs(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def cmd(self, line: str) -> ToolCall:
        return ToolCall("terminal", {"command": f"{self.bin}/{line}"})

    def verifier_calls(self) -> int:
        return sum(_VERIFIER in str(body.get("messages")) for body in self.server.aux_requests())


def _store():
    from plugins.cognition_wm import runtime

    return runtime.store()


def _test_runner(world: World, body: dict):
    """Runs the tests when asked — the way the request's lessons say, if any — and reports."""
    asked = _user_rows(body)[-1]
    if "[Evidence check" in asked:
        return Text("以上是我看到的結果。")
    if _after_tool(body):
        return Text("測試都通過了。")
    if "測試" in asked.split("\n", 1)[0]:
        return world.cmd("uv run pytest -q" if "uv run pytest" in asked else "pytest -q")
    return Text("好，記下了。")


# ------------------------------------------------------------------ L1 a correction carries over


def test_l1_after_a_correction_a_new_conversation_on_a_similar_task_does_it_right(tmp_path, monkeypatch):
    day2 = {}
    for v2 in (False, True):
        with World(tmp_path, monkeypatch, v2=v2, respond=_test_runner) as world:
            world.turn("day1", "幫我跑測試")
            world.turn("day1", "不對，要用 uv run pytest，不是直接 pytest")
            before = len(world.runs())
            world.turn("day2", "跑一下測試")
            day2[v2] = world.runs()[before:]

    assert day2[False] == ["pytest -q"]  # phase 1 forgets: the user has to correct it again
    assert day2[True] == ["uv run pytest -q"]


# ------------------------------------------------------------------ L2 not misapplied


def test_l2_another_project_or_another_kind_of_work_does_not_get_the_lesson(tmp_path, monkeypatch):
    def respond(world: World, body: dict):
        asked = _user_rows(body)[-1]
        if "週報" in asked.split("\n", 1)[0] and not _after_tool(body):
            return _task_state(action="create", title="寫週報", goal="寫這週的週報", kind="writing")
        return _test_runner(world, body)

    with World(tmp_path, monkeypatch, v2=True, respond=respond) as world:
        world.turn("day1", "幫我跑測試")
        world.turn("day1", "不對，要用 uv run pytest，不是直接 pytest")
        world.cd("another-project")
        before = len(world.runs())
        world.turn("day2", "跑一下測試")
        elsewhere = world.runs()[before:]
        world.cd("work")
        world.turn("day3", "幫我寫週報，順便列出測試相關的進度")
        writing = world.turn_row()

    assert elsewhere == ["pytest -q"]
    assert "<lessons" not in writing


# ------------------------------------------------------------------ L3 a withdrawn lesson is gone


def test_l3_a_revoked_lesson_is_not_used_again(tmp_path, monkeypatch):
    with World(tmp_path, monkeypatch, v2=True, respond=_test_runner) as world:
        world.turn("day1", "幫我跑測試")
        world.turn("day1", "不對，要用 uv run pytest，不是直接 pytest")
        world.turn("day2", "跑一下測試")
        world.turn("day2", "那條規則不對，忘掉它")
        before = len(world.runs())
        world.turn("day3", "跑一下測試")
        day3 = world.runs()[before:]

    assert day3 == ["pytest -q"]
    assert _store().experiences(statuses=("candidate", "active", "stale")) == []


# ------------------------------------------------------------------ L4 reports do not inflate


def test_l4_a_coding_agents_own_word_never_raises_our_record_or_its_lessons(tmp_path, monkeypatch):
    def relay(world: World, body: dict):
        if _after_tool(body):
            return Text("Claude 回報測試都通過了。")
        return world.cmd("claude -p 'fix the flaky test'")

    with World(tmp_path, monkeypatch, v2=True, respond=relay) as world:
        for n in range(5):
            world.turn("s1", f"請 claude 修第 {n} 個不穩的測試")

    ours = _store().outcomes(actor_kind="self")
    theirs = _store().outcomes(actor_kind="external_agent")
    assert [o.result for o in ours if o.result == "success"] == []
    assert [(o.actor_id, o.result) for o in theirs] == [("claude", "reported")] * 5
    assert _store().experiences() == []


# ------------------------------------------------------------------ L5 + L8 the record is per model


def _ops_worker(world: World, body: dict):
    """Claims a deploy it never ran, keeps the claim when asked to check, and opens a task when asked."""
    asked = _user_rows(body)[-1]
    first = asked.split("\n", 1)[0]
    if "[Evidence check" in asked:
        return Text("已部署上線。")
    if first.startswith("開始") and not _after_tool(body):
        kind = "writing" if "文件" in first else "ops"
        return _task_state(action="create", title=first, goal=first, kind=kind)
    if _after_tool(body):
        return Text("好，任務建立了。")
    if first.startswith("不對"):
        return Text("抱歉，我再確認。")
    return Text("已部署上線。")


def test_l5_l8_a_poor_record_makes_that_model_check_that_kind_of_work_and_nothing_else(tmp_path, monkeypatch):
    with World(tmp_path, monkeypatch, v2=True, respond=_ops_worker) as world:
        world.model = "model-a"
        world.turn("a", "開始部署服務")
        for _ in range(5):
            world.turn("a", "部署一下")
            world.turn("a", "不對，根本沒部署")
        before = world.verifier_calls()
        world.turn("a", "再部署一次")
        poor = (world.turn_sent(), world.verifier_calls() - before)

        world.turn("a-docs", "開始寫部署文件")
        before = world.verifier_calls()
        world.turn("a-docs", "寫下一段")
        other_work = (world.turn_sent(), world.verifier_calls() - before)

        world.model = "model-b"
        world.turn("b", "開始部署服務")
        before = world.verifier_calls()
        world.turn("b", "部署一下")
        other_model = (world.turn_sent(), world.verifier_calls() - before)

    assert "Track record" in poor[0] and '"deployed" 0 verified of 5' in poor[0] and poor[1] == 1
    assert "Track record" not in other_work[0] and other_work[1] == 0
    assert "Track record" not in other_model[0] and other_model[1] == 0


# ------------------------------------------------------------------ L6 limits and stops still win


def test_l6_a_lesson_never_overrides_a_limit_or_a_stop(tmp_path, monkeypatch):
    def worker(world: World, body: dict):
        asked = _user_rows(body)[-1]
        first = asked.split("\n", 1)[0]
        if _after_tool(body):
            return Text("好。")
        if first.startswith("開始"):
            return _task_state(action="create", title="改 parser", goal="改 parser 並部署", kind="engineering")
        if first.startswith("取消"):
            return _task_state(action="set_status", status="cancelled", reason=first)
        if "部署" in first and any("<lessons" in row and "deploy.sh" in row for row in _user_rows(body)):
            return world.cmd("deploy.sh")  # follows the lesson it was shown, whatever the user just said
        return Text("好。")

    with World(tmp_path, monkeypatch, v2=True, respond=worker) as world:
        world.turn("s1", "開始改 parser")
        world.turn("s1", "以後改完都要跑 deploy.sh 部署")
        world.turn("s1", "這次不要部署，只要改程式")
        limited = world.turn_row()
        world.turn("s1", "取消這個任務")
        world.turn("s1", "部署一下")
        stopped = world.turn_row()

    assert "deploy.sh" in limited and "<lessons" in stopped  # the lesson was there both times
    assert world.runs() == []  # and neither the limit nor the stop gave way to it


# ------------------------------------------------------------------ L7 plain chat is untouched


def test_l7_plain_chat_is_byte_identical_and_leaves_no_record(tmp_path, monkeypatch):
    seen = {}
    for v2 in (False, True):
        with World(tmp_path, monkeypatch, v2=v2, respond=lambda _w, _b: Text("今天天氣不錯。")) as world:
            world.turn("s1", "今天天氣如何？")
            world.turn("s1", "那明天呢？")
            sent = json.dumps([(b["messages"], b.get("tools")) for b in world.server.main_requests()],
                              ensure_ascii=False, sort_keys=True)
            seen[v2] = (sent.replace(str(world.root), "<arm>"), len(world.server.aux_requests()))
            if v2:
                assert (_store().experiences(), _store().outcomes()) == ([], [])

    assert seen[True] == seen[False]


# ------------------------------------------------------------------ L9 one check counts once


def test_l9_the_same_check_counts_once_through_a_retry_and_compaction(tmp_path, monkeypatch):
    with World(tmp_path, monkeypatch, v2=True, respond=_test_runner) as world:
        world.turn("s1", "幫我跑測試")
        world.turn("s1", "再跑一次測試")
        world.compress("s1")
        world.turn("s1", "最後再跑一次測試確認")

    assert len(world.runs()) == 3
    assert [o.result for o in _store().outcomes(actor_kind="self")] == ["success"]
