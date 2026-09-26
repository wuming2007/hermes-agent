"""Acceptance scenarios for cognition v2 phase 1: the same scripted model, OFF against ON.

OFF is today's behavior (cognition routing on, no working memory, no evidence loop). ON enables
the cognition_wm plugin and the evidence loop. Everything is real except the model: AIAgent,
SessionDB on disk, tools, compaction (the TUI /compress path) and plugin discovery, in the test's
isolated HERMES_HOME. Each scenario asserts where the arms must differ, and S0 that they must not.
"""

from __future__ import annotations

import os
import re
import stat
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

SID = "sess-v2"
# Small protected tail so a few turns are enough for /compress (as in tests/e2e/core/compaction).
_COMMON = "compression:\n  protect_last_n: 4\nauxiliary:\n  title_generation:\n    enabled: false\n"
_OFF = _COMMON + "cognition:\n  enabled: true\n"
_ON = (_COMMON + "plugins:\n  enabled: [cognition_wm]\ncognition:\n  enabled: true\n"
       "  evidence_loop:\n    enabled: true\n    max_extra_tool_calls: 2\n")
_TOOLSETS = ["file", "terminal", "cognition_wm", "delegation"]
_SUMMARY = "## Goal\nKeep helping with the user's task.\n## Progress\n### Done\n- Planned the work.\n## Next Steps\n- Continue.\n"


def _task_state(**arguments):
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": arguments}]})


class Arm:
    """One scenario arm: turns of a real agent against a scripted responder."""

    def __init__(self, on: bool, tmp_path: Path, monkeypatch, responder: Callable, aux: Callable | None = None):
        self.on, self.work = on, tmp_path / ("on" if on else "off")
        self.work.mkdir()
        self.home = tmp_path / ("home-on" if on else "home-off")
        monkeypatch.setenv("HERMES_HOME", str(self.home))
        monkeypatch.chdir(self.work)
        self.server = FakeLLMServer(responder, aux=aux or (lambda _r: Text(_SUMMARY)))
        self._monkeypatch = monkeypatch

    def __enter__(self) -> "Arm":
        from hermes_cli.plugins import discover_plugins
        from hermes_state import SessionDB

        self.server.start()
        write_hermes_home(self.home, self.server.base_url, extra_config=_ON if self.on else _OFF)
        discover_plugins(force=True)
        self.db = SessionDB(db_path=self.home / "state.db")
        return self

    def __exit__(self, *_exc) -> None:
        self.server.stop()

    def agent(self):
        from run_agent import AIAgent

        return AIAgent(provider="custom", base_url=self.server.base_url, api_key="sk-fake", model="fake-model",
                       session_db=self.db, session_id=SID, quiet_mode=True, platform="cli",
                       enabled_toolsets=_TOOLSETS, skip_context_files=True, skip_memory=True)

    def history(self) -> list:
        return self.db.get_messages_as_conversation(SID)

    def turn(self, message: str) -> dict:
        return self.agent().run_conversation(message, conversation_history=self.history())

    def compress(self) -> int:
        """The TUI /compress path (the newest two exchanges stay verbatim)."""
        from tui_gateway.server import _compress_session_history

        session = {"agent": self.agent(), "history": self.history(), "history_lock": threading.Lock(),
                   "history_version": 1, "session_key": SID}
        removed, _usage = _compress_session_history(session, "")
        return removed

    def new_run(self) -> None:
        """A resumed chat in a new process: grants from the earlier run become historical."""
        import plugins.cognition_wm.provenance as provenance

        self._monkeypatch.setattr(provenance, "RUN_ID", f"later-run-{self.on}")


def _last_user(body: dict) -> str:
    return next(str(m.get("content")) for m in reversed(body["messages"]) if m.get("role") == "user")


def _turn_tool_results(result: dict) -> list[str]:
    messages = result["messages"]
    start = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    return [str(m["content"]) for m in messages[start:] if m.get("role") == "tool"]


def _json(text: Any) -> dict:
    from utils import safe_json_loads

    parsed = safe_json_loads(str(text))
    return parsed if isinstance(parsed, dict) else {}


def _scripted(*responses) -> Callable:
    queue = list(responses)
    return lambda _record: queue.pop(0) if queue else Text("ok")


# ------------------------------------------------------------------ S0 chat costs nothing


def test_s0_plain_chat_is_untouched(tmp_path, monkeypatch):
    runs = {}
    for on in (False, True):
        with Arm(on, tmp_path, monkeypatch, _scripted(Text("今天天氣不錯，適合出去走走。"))) as arm:
            result = arm.turn("今天適合出門嗎？")
            runs[on] = (result["final_response"], [_last_user(b) for b in arm.server.main_requests()])

    assert runs[True] == runs[False]
    assert len(runs[True][1]) == 1 and "<working-memory" not in runs[True][1][0]


# ------------------------------------------------------------------ S1 compaction and resume


def test_s1_task_state_survives_compaction_and_a_new_run(tmp_path, monkeypatch):
    plan = _task_state(action="create", title="資料庫遷移規劃", goal="規劃 schema 遷移", kind="engineering", items=[
        {"kind": "decision", "text": "用 alembic 管理遷移"},
        {"kind": "deliverable", "text": "遷移腳本"},
        {"kind": "authorization", "text": "可以改 migrations 目錄", "state": "granted", "action_class": "code_change"},
        {"kind": "authorization", "text": "先不要動正式資料庫", "state": "forbidden", "action_class": "data_migration"},
    ])
    first_after = {}
    for on in (False, True):
        steps = [Text(f"步驟{n}：" + "說明內容" * 120) for n in range(1, 5)]
        script = _scripted(plan, Text("好，已記下。"), *steps, Text("繼續。"))
        with Arm(on, tmp_path, monkeypatch, script) as arm:
            arm.turn("幫我規劃資料庫遷移，決定用 alembic，可以改 migrations 目錄，但先不要動正式資料庫。")
            for n in range(1, 5):
                arm.turn(f"說明第 {n} 步")
            assert arm.compress() > 0
            arm.new_run()
            before = len(arm.server.main_requests())
            arm.turn("繼續")
            first_after[on] = _last_user(arm.server.main_requests()[before])

    assert "<working-memory" not in first_after[False]
    on = first_after[True]
    assert '<working-memory rev="' in on and "用 alembic 管理遷移" in on and "[pending] 遷移腳本" in on
    assert "code_change=historical" in on and "data_migration=forbidden" in on


# ------------------------------------------------------------------ S2 cancel against a stale summary


def test_s2_a_stopped_task_is_not_revived_by_a_stale_summary(tmp_path, monkeypatch):
    stale = ("## Goal\nRefactor parser.py.\n## Progress\n### Done\n- Planned the refactor.\n"
             "## Next Steps\n- Apply the refactor patch to parser.py right away.\n")
    outcome = {}
    for on in (False, True):
        script = _scripted(
            _task_state(action="create", title="重構 parser", goal="重構 parser.py", kind="engineering"), Text("開始。"),
            _task_state(action="set_status", status="paused", reason="使用者說先不要做了"), Text("好，先停。"),
            *[Text(f"回覆{n}：" + "說明內容" * 120) for n in range(3)],
            ToolCall("patch", {"path": "parser.py", "old_string": "old", "new_string": "new"}), Text("已套用。"))
        with Arm(on, tmp_path, monkeypatch, script, aux=lambda _r: Text(stale)) as arm:
            (arm.work / "parser.py").write_text("old\n", encoding="utf-8")
            for message in ("幫我重構 parser.py", "先不要做了", "還在嗎", "謝謝", "嗯"):
                arm.turn(message)
            assert arm.compress() > 0  # the pause turn is now only the stale summary
            before = len(arm.server.main_requests())
            result = arm.turn("好")
            outcome[on] = ((arm.work / "parser.py").read_text(encoding="utf-8"), _turn_tool_results(result),
                           _last_user(arm.server.main_requests()[before]))

    assert outcome[False][0] == "new\n"
    text, (patched,), snapshot_row = outcome[True]
    assert text == "old\n" and "BLOCKED" in patched
    assert "PAUSED" in snapshot_row


# ------------------------------------------------------------------ S3 prompt only


def test_s3_a_prompt_only_request_launches_nothing(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "claude-ran"
    fake = bin_dir / "claude"
    fake.write_text(f"#!/bin/sh\ntouch '{marker}'\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    touched = {}
    for on in (False, True):
        marker.unlink(missing_ok=True)
        script = _scripted(ToolCall("terminal", {"command": f"{fake} -p 'refactor the parser'"}),
                           ToolCall("write_file", {"path": "parser.py", "content": "x\n"}), Text("這是 prompt：……"))
        with Arm(on, tmp_path, monkeypatch, script) as arm:
            arm.turn("只要幫我寫給 Claude 的 prompt，不要改 repo。")
            touched[on] = (marker.exists(), (arm.work / "parser.py").exists())

    assert touched == {False: (True, True), True: (False, False)}


# ------------------------------------------------------------------ S4 unfinished work claimed as done


def test_s4_timeouts_and_running_jobs_are_not_reported_as_done(tmp_path, monkeypatch):
    claim = "全部完成，已部署上線。"
    outcome = {}
    for on in (False, True):
        script = _scripted(
            ToolCall("terminal", {"command": "sleep 5", "timeout": 1}),
            ToolCall("terminal", {"command": "sleep 2", "background": True}),
            Text(claim),
            ToolCall("read_file", {"path": "notes.txt"}),
            Text(claim),
        )
        with Arm(on, tmp_path, monkeypatch, script) as arm:
            (arm.work / "notes.txt").write_text("deploy notes", encoding="utf-8")
            result = arm.turn("幫我把服務部署上線")
            stored = [m for m in arm.history() if m.get("role") == "assistant"][-1]["content"]
            nudges = max(sum("[Evidence check" in str(m.get("content")) for m in b["messages"])
                         for b in arm.server.main_requests())
            outcome[on] = (result["final_response"], stored, nudges)

    assert outcome[False] == (claim, claim, 0)
    final, stored, nudges = outcome[True]
    assert nudges == 1 and stored == final and final.startswith(claim)
    assert "證據狀態" in final and ("逾時" in final or "仍在執行" in final)


# ------------------------------------------------------------------ S5 new evidence replaces a fact


def test_s5_new_evidence_replaces_an_old_fact_and_keeps_it(tmp_path, monkeypatch):
    def responder_for(arm_state: dict) -> Callable:
        def respond(record: dict) -> Any:
            messages = record["body"]["messages"]
            step = arm_state["step"] = arm_state.get("step", 0) + 1
            results = [m for m in messages if m.get("role") == "tool"]
            if step == 1:
                return ToolCall("read_file", {"path": "config.yaml"})
            if step == 2:
                return _task_state(action="create", title="查 port", goal="確認服務 port", kind="ops", items=[
                    {"kind": "fact", "text": "port 是 8080", "evidence_ids": [f"e_{messages[-2]['tool_calls'][0]['id']}"]}])
            if step == 3:
                arm_state["fact"] = (_json(results[-1]["content"]).get("items") or [None])[0]
                return Text("port 是 8080。")
            if step == 4:
                return ToolCall("read_file", {"path": "config.yaml"})
            if step == 5:
                call = messages[-2]["tool_calls"][0]["id"]
                return _task_state(action="add", items=[{"kind": "fact", "text": "port 是 9090",
                                                         "supersedes": arm_state.get("fact"), "evidence_ids": [f"e_{call}"]}])
            return Text("現在是 9090。")
        return respond

    snapshots = {}
    for on in (False, True):
        state: dict = {}
        with Arm(on, tmp_path, monkeypatch, responder_for(state)) as arm:
            (arm.work / "config.yaml").write_text("port: 8080\n", encoding="utf-8")
            arm.turn("查一下 config 的 port")
            (arm.work / "config.yaml").write_text("port: 9090\n", encoding="utf-8")
            arm.turn("設定檔改過了，再查一次")
            before = len(arm.server.main_requests())
            arm.turn("所以現在 port 是多少？")
            snapshots[on] = _last_user(arm.server.main_requests()[before])
            if on:
                from plugins.cognition_wm.runtime import store

                (task,) = store().tasks(project_key=f"cwd:{os.path.realpath(arm.work)}")
                facts = {i.text: i for i in store().items(task.task_id, kinds=("fact",))}

    assert "<working-memory" not in snapshots[False]
    assert "port 是 9090 (e_" in snapshots[True] and "port 是 8080" not in snapshots[True]
    assert facts["port 是 8080"].state == "superseded" and facts["port 是 8080"].evidence_ids
    assert facts["port 是 9090"].supersedes == facts["port 是 8080"].item_id


# ------------------------------------------------------------------ S6 a subagent's report is not a check


def test_s6_a_subagents_report_is_not_our_verification(tmp_path, monkeypatch):
    """Top-level delegations run async; the result comes back as a notice the next turn."""
    from tools.process_registry_notifications import format_process_notification

    notice = format_process_notification({
        "type": "async_delegation", "delegation_id": "deleg_1", "goal": "Fix the parser bug and run the tests",
        "status": "completed", "summary": "Fixed the parser and all tests pass.", "api_calls": 4})
    claim = "已驗證所有測試通過。"
    outcome = {}
    for on in (False, True):
        script = _scripted(Text("已請子代理處理，完成後回報。"), Text(claim), Text(claim))
        with Arm(on, tmp_path, monkeypatch, script) as arm:
            arm.turn("請子代理修 parser 並跑測試")
            outcome[on] = arm.turn(notice)["final_response"]

    assert outcome[False] == claim
    assert outcome[True].startswith(claim) and "只有子代理的回報" in outcome[True]
