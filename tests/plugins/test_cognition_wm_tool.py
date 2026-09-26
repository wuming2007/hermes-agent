"""Behavior contract: the task_state tool only reaches this conversation's own tasks, and a record
settles the user's stop or limit only when it records that stop or limit.

Real working-memory store in the test's isolated HERMES_HOME; the tool is called as the agent
dispatches it (``handle(args, session_id=...)``).
"""

import json

import pytest


def _turn(monkeypatch, cwd, session_id, *, intent=None):
    from plugins.cognition_wm import runtime

    monkeypatch.chdir(cwd)
    state = runtime.TurnState(session_id=session_id, turn_id="t1", owner=True, project_key=runtime.project_key(),
                              lineage_root=session_id, platform="cli", intent=intent)
    runtime.remember_turn(state)
    return state


def _call(session_id, **args):
    from plugins.cognition_wm import tool

    return json.loads(tool.handle(args, session_id=session_id))


def _task_elsewhere(monkeypatch, cwd, session_id):
    """A task with one item in another conversation (another project or another lineage)."""
    _turn(monkeypatch, cwd, session_id)
    created = _call(session_id, action="create", title="別的任務", goal="別的任務", kind="engineering",
                    items=[{"kind": "deliverable", "text": "報告"}])
    return created["task"]["task_id"], created["items"][0]


@pytest.mark.parametrize("same_project", [False, True], ids=["other-project", "same-project-other-conversation"])
def test_explicit_ids_cannot_reach_a_task_outside_this_conversation(tmp_path, monkeypatch, same_project):
    from plugins.cognition_wm import runtime

    here, there = tmp_path / "a", tmp_path / ("a" if same_project else "b")
    here.mkdir(exist_ok=True)
    there.mkdir(exist_ok=True)
    task_id, item_id = _task_elsewhere(monkeypatch, there, "s-there")
    before = [(i.kind, i.state) for i in runtime.store().items(task_id)]
    _turn(monkeypatch, here, "s-here")

    attempts = [
        {"action": "set_status", "task_id": task_id, "status": "cancelled"},
        {"action": "add", "task_id": task_id,
         "items": [{"kind": "authorization", "text": "可以推", "state": "granted", "action_class": "repo_write"}]},
        {"action": "update_item", "item_id": item_id, "state": "done"},
    ]
    replies = [_call("s-here", **args) for args in attempts]

    assert all("error" in reply for reply in replies)
    wm = runtime.store()
    assert wm.task(task_id).status == "active"
    assert [(i.kind, i.state) for i in wm.items(task_id)] == before


@pytest.mark.parametrize(
    "intent_kind, record, settles",
    [
        pytest.param("cancel", {"action": "set_status", "status": "cancelled"}, True, id="cancel-recorded"),
        pytest.param("cancel", {"action": "add", "items": [{"kind": "progress", "text": "讀完了"}]}, False,
                     id="cancel-then-progress"),
        pytest.param("pause", {"action": "set_status", "status": "paused"}, True, id="pause-recorded"),
        pytest.param("restrict", {"action": "add", "items": [
            {"kind": "authorization", "text": "不要改 repo", "state": "forbidden", "action_class": "repo_write"}]},
                     True, id="limit-recorded-as-a-rule"),
        pytest.param("restrict", {"action": "add", "items": [{"kind": "constraint", "text": "只要計畫"}]}, False,
                     id="limit-as-free-text"),
    ],
)
def test_a_record_settles_the_users_stop_or_limit_only_when_it_records_it(tmp_path, monkeypatch, intent_kind,
                                                                         record, settles):
    from plugins.cognition_wm import runtime
    from plugins.cognition_wm.intents import ControlIntent

    _turn(monkeypatch, tmp_path, "s1")
    _call("s1", action="create", title="重構 parser", goal="重構 parser", kind="engineering")
    _turn(monkeypatch, tmp_path, "s1", intent=ControlIntent(intent_kind, "（使用者的話）"))

    assert "ok" in _call("s1", **record)
    assert (runtime.current_turn("s1").intent is None) is settles
