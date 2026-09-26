"""Behavior contract: the working-memory store keeps task state honest across turns, runs and processes.

Real SQLite under the test's isolated HERMES_HOME; expectations are hand-written.
"""

import multiprocessing
import os
from pathlib import Path

import pytest

from plugins.cognition_wm.model import SourceRef
from plugins.cognition_wm.store import (
    InvalidTransition,
    LimitReached,
    Limits,
    ProvenanceError,
    RevisionConflict,
    WorkingMemoryStore,
)


def _src(authored_by="owner", turn="t1", run="run-a"):
    return SourceRef(type="user_message" if authored_by == "owner" else "assistant", session_id="s1", turn_id=turn,
                     ref="m1", at=1000.0, authored_by=authored_by, run_id=run)


def _store(**limits):
    return WorkingMemoryStore.open(limits=Limits(**limits))


def _task(store, project="cwd:/work", lineage="s1", source=None):
    return store.create_task(project_key=project, lineage_root=lineage, origin_session="s1", title="ship the report",
                             goal="write and send the weekly report", kind="writing", source=source or _src())


def test_each_profile_keeps_its_own_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _task(WorkingMemoryStore.open())
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "b"))
    store_b = WorkingMemoryStore.open()

    assert store_b.tasks(project_key="cwd:/work") == []
    assert store_b.path == tmp_path / "b" / "plugin-data" / "cognition_wm" / "working_memory.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    assert len(WorkingMemoryStore.open().tasks(project_key="cwd:/work")) == 1


def test_only_the_owner_grants_authority():
    store = _store()
    task = _task(store)
    grant = {"action_class": "coding_agent_launch"}

    with pytest.raises(ProvenanceError):
        store.add_item(task.task_id, kind="authorization", text="may run claude", state="granted", attrs=grant,
                       source=_src("model"))
    item = store.add_item(task.task_id, kind="authorization", text="may run claude", state="granted", attrs=grant,
                          source=_src())
    assert item.state == "granted"
    with pytest.raises(ProvenanceError):
        _task(store, source=_src("model"))


def test_a_superseded_fact_keeps_its_history():
    store = _store()
    task = _task(store)
    old = store.add_item(task.task_id, kind="fact", text="report due Friday", source=_src())
    new = store.add_item(task.task_id, kind="fact", text="report due Thursday", source=_src(), supersedes=old.item_id)

    items = {i.item_id: i for i in store.items(task.task_id)}
    assert (items[old.item_id].state, items[new.item_id].state) == ("superseded", "current")
    assert items[new.item_id].supersedes == old.item_id


def test_writes_are_compare_and_set_on_the_task_revision():
    store = _store()
    task = _task(store)

    paused = store.set_status(task.task_id, "paused", source=_src(), expected_revision=task.revision)
    assert paused.revision == task.revision + 1
    with pytest.raises(RevisionConflict):
        store.set_status(task.task_id, "cancelled", source=_src(), expected_revision=task.revision)


def test_a_cancelled_task_comes_back_only_on_a_later_owner_turn():
    store = _store()
    task = store.set_status(_task(store).task_id, "cancelled", source=_src(turn="t1"), expected_revision=1)

    with pytest.raises(InvalidTransition):
        store.set_status(task.task_id, "active", source=_src(turn="t1"), expected_revision=task.revision)
    with pytest.raises(ProvenanceError):
        store.set_status(task.task_id, "active", source=_src("model", turn="t2"), expected_revision=task.revision)
    assert store.set_status(task.task_id, "active", source=_src(turn="t2"), expected_revision=task.revision).status == "active"


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_only_the_owner_pauses_or_cancels(status):
    store = _store()
    task = _task(store)

    with pytest.raises(ProvenanceError):
        store.set_status(task.task_id, status, source=_src("model"), expected_revision=task.revision)
    assert store.set_status(task.task_id, "done", source=_src("model"), expected_revision=task.revision).status == "done"


def test_one_focus_per_lineage():
    store = _store()
    first, second, elsewhere = _task(store), _task(store), _task(store, lineage="s9")
    for task in (first, second, elsewhere):
        store.set_focus(task.task_id, source=_src())

    focus = {t.task_id: t.focus for t in store.tasks(project_key="cwd:/work")}
    assert focus == {first.task_id: False, second.task_id: True, elsewhere.task_id: True}


def test_open_task_and_item_caps():
    store = _store(open_tasks_per_project=2, items_per_task=2)
    first, _second = _task(store), _task(store)
    with pytest.raises(LimitReached):
        _task(store)
    store.set_status(first.task_id, "done", source=_src(), expected_revision=first.revision)
    third = _task(store)
    store.add_item(third.task_id, kind="progress", text="outline", source=_src())
    store.add_item(third.task_id, kind="progress", text="draft", source=_src())
    with pytest.raises(LimitReached):
        store.add_item(third.task_id, kind="progress", text="polish", source=_src())


def test_evidence_of_ended_tasks_goes_first():
    now = [90_000.0]  # e_old is over a day old; the rest are not
    store = WorkingMemoryStore.open(limits=Limits(evidence_rows=3, evidence_days=1), clock=lambda: now[0])
    live, ended = _task(store), _task(store)
    for i, task in enumerate((live, ended, live, ended)):
        store.record_evidence(evidence_id=f"e_{i}", task_id=task.task_id, session_id="s1", turn_id="t1",
                              tool_name="terminal", kind="test", status="ok", provenance="self", recorded_at=4000.0 + i)
    store.record_evidence(evidence_id="e_old", task_id=live.task_id, session_id="s0", turn_id="t0",
                          tool_name="read_file", kind="read", status="ok", provenance="self", recorded_at=100.0)
    store.set_status(ended.task_id, "done", source=_src(), expected_revision=ended.revision)

    store.prune_evidence()

    assert sorted(store.evidence_ids()) == ["e_0", "e_2", "e_3"]


def test_a_grant_from_an_earlier_run_or_long_ago_is_historical():
    now = [1000.0]
    store = WorkingMemoryStore.open(clock=lambda: now[0])
    task = _task(store)
    store.add_item(task.task_id, kind="authorization", text="may deploy", state="granted",
                   attrs={"action_class": "deploy_restart"}, source=_src(run="run-a"))
    store.add_item(task.task_id, kind="authorization", text="never message the team", state="forbidden",
                   attrs={"action_class": "external_message"}, source=_src(run="run-a"))

    assert store.effective_authorizations(task.task_id, run_id="run-a") == {
        "deploy_restart": "granted", "external_message": "forbidden"}
    assert store.effective_authorizations(task.task_id, run_id="run-b")["deploy_restart"] == "historical"
    now[0] += 13 * 3600
    assert store.effective_authorizations(task.task_id, run_id="run-a") == {
        "deploy_restart": "historical", "external_message": "forbidden"}


def _append_items(home: str, task_id: str, worker: int, count: int) -> None:
    os.environ["HERMES_HOME"] = home
    store = WorkingMemoryStore.open()
    for n in range(count):
        while True:
            revision = store.task(task_id).revision
            try:
                store.add_item(task_id, kind="progress", text=f"w{worker}-{n}", source=_src(),
                               expected_revision=revision)
                break
            except RevisionConflict:
                continue


def test_two_processes_writing_one_task_lose_nothing():
    store = _store()
    task = _task(store)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_items, args=(os.environ["HERMES_HOME"], task.task_id, w, 15)) for w in (1, 2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)

    assert [p.exitcode for p in procs] == [0, 0]
    assert len(store.items(task.task_id)) == 30
    assert store.task(task.task_id).revision == task.revision + 30
    assert Path(store.path).exists()


def test_an_earlier_task_is_attached_only_within_its_project():
    store = _store()
    here = _task(store, lineage="s-old")
    elsewhere = _task(store, project="cwd:/other", lineage="s-old")

    attached = store.attach(here.task_id, project_key="cwd:/work", lineage_root="s-new", source=_src(turn="t9"))
    assert (attached.lineage_root, attached.focus) == ("s-new", True)
    with pytest.raises(InvalidTransition):
        store.attach(elsewhere.task_id, project_key="cwd:/work", lineage_root="s-new", source=_src(turn="t9"))
    with pytest.raises(ProvenanceError):
        store.attach(here.task_id, project_key="cwd:/work", lineage_root="s-new", source=_src("model"))


def test_revoking_one_rule_leaves_the_others_of_its_class():
    store = _store()
    task = _task(store)
    store.add_item(task.task_id, kind="authorization", text="may deploy", state="granted",
                   attrs={"action_class": "deploy_restart"}, source=_src())
    ask = store.add_item(task.task_id, kind="authorization", text="ask before deploying", state="ask_first",
                         attrs={"action_class": "deploy_restart"}, source=_src())
    assert store.effective_authorizations(task.task_id, run_id="run-a") == {"deploy_restart": "ask_first"}

    store.set_item_state(ask.item_id, "revoked", source=_src())

    assert store.effective_authorizations(task.task_id, run_id="run-a") == {"deploy_restart": "granted"}


@pytest.mark.parametrize(
    "stop, then",
    [
        pytest.param("cancelled", "done", id="cancelled-to-done"),
        pytest.param("cancelled", "superseded", id="cancelled-to-superseded"),
        pytest.param("cancelled", "paused", id="cancelled-to-paused"),
        pytest.param("paused", "done", id="paused-to-done"),
        pytest.param("paused", "active", id="paused-to-active"),
    ],
)
def test_a_task_stopped_in_this_turn_stays_stopped_for_the_rest_of_the_turn(stop, then):
    store = _store()
    task = _task(store)
    store.set_status(task.task_id, stop, source=_src(turn="t2"), expected_revision=None)

    with pytest.raises(InvalidTransition):
        store.set_status(task.task_id, then, source=_src(turn="t2"), expected_revision=None)

    assert store.task(task.task_id).status == stop


def test_a_stop_can_be_made_stronger_in_the_same_turn():
    store = _store()
    task = _task(store)
    store.set_status(task.task_id, "paused", source=_src(turn="t2"), expected_revision=None)

    assert store.set_status(task.task_id, "cancelled", source=_src(turn="t2"), expected_revision=None).status \
        == "cancelled"


def test_permissions_given_before_a_stop_must_be_confirmed_again_after_the_task_is_revived():
    store = _store()
    task = _task(store)
    store.add_item(task.task_id, kind="authorization", text="可以推", state="granted",
                   attrs={"action_class": "repo_write"}, source=_src(turn="t1"))
    store.set_status(task.task_id, "cancelled", source=_src(turn="t2"), expected_revision=None)
    store.set_status(task.task_id, "active", source=_src(turn="t3"), expected_revision=None)

    assert store.effective_authorizations(task.task_id, run_id="run-a")["repo_write"] == "historical"

    store.add_item(task.task_id, kind="authorization", text="可以部署", state="granted",
                   attrs={"action_class": "deploy_restart"}, source=_src(turn="t3"))
    assert store.effective_authorizations(task.task_id, run_id="run-a")["deploy_restart"] == "granted"
