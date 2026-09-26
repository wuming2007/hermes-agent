"""Behavior contract: working memory rides the current user row, never the system prompt, and keeps
task state, authority and evidence straight across turns and runs.

Real agent loop, real plugin discovery (``plugins.enabled: [cognition_wm]``) against a scripted
provider, in the test's isolated HERMES_HOME.
"""

import json
import os
from pathlib import Path

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home

_CONFIG = "plugins:\n  enabled: [cognition_wm]\ncognition:\n  enabled: true\n"


def _task_state(**arguments):
    return ToolCall("tool_call", {"calls": [{"name": "task_state", "arguments": arguments}]})


def _agent(server, db, session_id="sess-wm"):
    from run_agent import AIAgent

    return AIAgent(
        provider="custom", base_url=server.base_url, api_key="sk-fake", model="fake-model", session_db=db,
        session_id=session_id, quiet_mode=True, platform="cli", enabled_toolsets=["file", "cognition_wm"],
        skip_context_files=True, skip_memory=True,
    )


def _setup(tmp_path, monkeypatch, server):
    from hermes_cli.plugins import discover_plugins
    from hermes_state import SessionDB

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    write_hermes_home(Path(os.environ["HERMES_HOME"]), server.base_url, extra_config=_CONFIG)
    discover_plugins(force=True)
    return work, SessionDB(db_path=Path(os.environ["HERMES_HOME"]) / "state.db")


def _user_rows(body):
    return [str(m.get("content")) for m in body["messages"] if m.get("role") == "user"]


def test_no_task_and_no_intent_injects_nothing(tmp_path, monkeypatch):
    with FakeLLMServer([Text("hi there")]) as server:
        _work, db = _setup(tmp_path, monkeypatch, server)
        _agent(server, db).run_conversation("hello", conversation_history=[])
        (body,) = server.main_requests()

    assert "<working-memory" not in _user_rows(body)[-1]


def test_state_rides_the_user_row_and_the_prompt_stays_byte_stable(tmp_path, monkeypatch):
    script = [
        _task_state(action="create", title="週報草稿", goal="寫週報草稿給主管", kind="writing",
                    items=[{"kind": "constraint", "text": "只要草稿，不要寄出"}, {"kind": "deliverable", "text": "週報草稿"}]),
        Text("好的，已記下任務。"),
        Text("草稿如下。"),
        Text("好。"),
    ]
    with FakeLLMServer(script) as server:
        _work, db = _setup(tmp_path, monkeypatch, server)
        _agent(server, db).run_conversation("幫我寫週報給主管，只要草稿，不要寄出。", conversation_history=[])
        history = db.get_messages_as_conversation("sess-wm")
        _agent(server, db).run_conversation("開始寫", conversation_history=history)
        history = db.get_messages_as_conversation("sess-wm")
        _agent(server, db).run_conversation("謝謝", conversation_history=history)
        bodies = server.main_requests()

    assert "Pending" in _user_rows(bodies[0])[-1]  # the restriction waits to be recorded
    turn2, turn3 = _user_rows(bodies[2])[-1], _user_rows(bodies[3])[-1]
    assert "<working-memory" in turn2 and "週報草稿" in turn2 and "只要草稿，不要寄出" in turn2
    assert "unchanged" in turn3 and "週報草稿" not in turn3
    assert "週報草稿" in _user_rows(bodies[3])[-2]  # the earlier full snapshot is replayed as sent
    assert len({json.dumps(b["messages"][0], ensure_ascii=False) for b in bodies}) == 1
    assert len({json.dumps(b.get("tools"), sort_keys=True) for b in bodies}) == 1


def test_a_grant_from_an_earlier_run_reads_as_historical(tmp_path, monkeypatch):
    import plugins.cognition_wm.provenance as provenance

    script = [
        _task_state(action="create", title="上線新版", goal="部署新版到正式環境", kind="ops",
                    items=[{"kind": "authorization", "text": "可以重啟服務", "state": "granted",
                            "action_class": "deploy_restart"}]),
        Text("已記下。"),
        Text("好，繼續。"),
    ]
    with FakeLLMServer(script) as server:
        _work, db = _setup(tmp_path, monkeypatch, server)
        _agent(server, db).run_conversation("幫我上線新版，可以重啟服務。", conversation_history=[])
        monkeypatch.setattr(provenance, "RUN_ID", "a-later-run")
        history = db.get_messages_as_conversation("sess-wm")
        _agent(server, db).run_conversation("繼續上線", conversation_history=history)
        bodies = server.main_requests()

    assert "deploy_restart=historical" in _user_rows(bodies[-1])[-1]


def test_tool_results_become_evidence_of_the_focus_task(tmp_path, monkeypatch):
    from plugins.cognition_wm.store import WorkingMemoryStore

    script = [
        _task_state(action="create", title="整理筆記", goal="讀筆記並整理", kind="writing"),
        ToolCall("read_file", {"path": "notes.txt"}),
        Text("讀完了。"),
    ]
    with FakeLLMServer(script) as server:
        work, db = _setup(tmp_path, monkeypatch, server)
        (work / "notes.txt").write_text("alpha", encoding="utf-8")
        result = _agent(server, db).run_conversation("幫我整理筆記", conversation_history=[])

    read_call = next(tc["id"] for m in result["messages"] for tc in m.get("tool_calls") or []
                     if tc["function"]["name"] == "read_file")
    store = WorkingMemoryStore.open()
    (task,) = store.tasks(project_key=f"cwd:{os.path.realpath(work)}")
    assert f"e_{read_call}" in store.evidence_ids(task.task_id)


def test_every_tool_result_is_its_own_evidence_even_without_a_call_id(tmp_path, monkeypatch):
    from plugins.cognition_wm import evidence_link, runtime

    monkeypatch.chdir(tmp_path)
    for path in ("a.txt", "b.txt"):
        evidence_link.on_post_tool_call(tool_name="read_file", args={"path": path}, result='{"content": "x"}',
                                        session_id="s-none", tool_call_id="", turn_id="t1", status="ok")

    assert len(runtime.store().evidence_ids()) == 2


def test_a_runtime_notice_cannot_create_a_task(tmp_path, monkeypatch):
    notice = "[IMPORTANT: Background process p1 completed normally (exit code 0).\nCommand: ./job.sh\nOutput:\nok]"
    script = [_task_state(action="create", title="x", goal="y", kind="other"), Text("ok")]
    with FakeLLMServer(script) as server:
        _work, db = _setup(tmp_path, monkeypatch, server)
        result = _agent(server, db).run_conversation(notice, conversation_history=[])

    (tool_row,) = [m for m in result["messages"] if m.get("role") == "tool"]
    assert "owner" in str(tool_row["content"])


def test_the_snapshot_stays_within_its_configured_size(tmp_path, monkeypatch):
    items = [{"kind": "fact", "text": f"fact number {n} " + "x" * 80} for n in range(12)]
    script = [_task_state(action="create", title="長任務", goal="很多事實", kind="research", items=items),
              Text("記下了。"), Text("好。")]
    with FakeLLMServer(script) as server:
        work, db = _setup(tmp_path, monkeypatch, server)
        home = Path(os.environ["HERMES_HOME"])
        write_hermes_home(home, server.base_url, extra_config=_CONFIG.replace(
            "  enabled: [cognition_wm]\n",
            "  enabled: [cognition_wm]\n  entries:\n    cognition_wm:\n      settings:\n        snapshot_max_chars: 400\n"))
        _agent(server, db).run_conversation("研究這些事實", conversation_history=[])
        history = db.get_messages_as_conversation("sess-wm")
        _agent(server, db).run_conversation("繼續整理", conversation_history=history)
        last = _user_rows(server.main_requests()[-1])[-1]

    snapshot = last[last.index("<working-memory"):]
    body = snapshot.split("\n", 2)[2].rsplit("\n</working-memory>", 1)[0]
    assert "長任務" in body and len(body) <= 400


def test_the_snapshot_dates_what_it_carries_from_before(tmp_path, monkeypatch):
    """A fact or a finished step shows when it was recorded, so it is cited as of then, not as now."""
    import re

    import plugins.cognition_wm.provenance as provenance
    from plugins.cognition_wm import runtime, snapshot

    monkeypatch.chdir(tmp_path)
    turn = runtime.TurnState(session_id="s1", turn_id="t1", owner=True, project_key=runtime.project_key(),
                             lineage_root="s1", platform="cli")
    source = provenance.turn_source(owner=True, session_id="s1", turn_id="t1")
    wm = runtime.store()
    task = wm.create_task(project_key=turn.project_key, lineage_root="s1", origin_session="s1", title="上線",
                          goal="上線新版", kind="ops", source=source)
    wm.set_focus(task.task_id, source=source)
    wm.add_item(task.task_id, kind="fact", text="正式環境跑 v2.3", source=source)

    body = snapshot.render_body(turn, runtime.settings())

    assert re.search(r"正式環境跑 v2\.3 \(as of \d{4}-\d{2}-\d{2} \d{2}:\d{2}\)", body)
