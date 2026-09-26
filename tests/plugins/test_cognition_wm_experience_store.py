"""Behavior contract: experiences and the track record keep who said what, count each piece of
evidence once, and never bring back what ended — across upgrades, profiles and processes.

Real SQLite under the test's isolated HERMES_HOME; expectations are hand-written.
"""

import multiprocessing
import os

import pytest

from plugins.cognition_wm.experience_store import ExperienceStore, LearningLimits
from plugins.cognition_wm.model import RECALLED_STATUSES, SourceRef
from plugins.cognition_wm.store import InvalidTransition, LimitReached, ProvenanceError, WorkingMemoryStore

DAY = 86400.0


def _src(authored_by="owner", session="s1", turn="t1"):
    kind = {"owner": "user_message", "model": "assistant", "tool": "tool_result", "system": "system"}[authored_by]
    return SourceRef(type=kind, session_id=session, turn_id=turn, ref="m1", at=1000.0, authored_by=authored_by,
                     run_id="run-a")


def _xp(store, *, project="git:/work", status="candidate", origin=None, scope="project", **extra):
    fields = dict(kind="procedure", scope=scope, project_key=None if scope == "global" else project,
                  task_kind="engineering", situation="running the tests", behavior="use `uv run pytest`",
                  tags=("測試", "pytest"), lesson_key="pytest run uv", status=status, origin=origin or _src())
    return store.add_experience(**{**fields, **extra})


def _outcome(store, key, *, result="unknown", actor_kind="self", **extra):
    fields = dict(evidence_key=key, session_id="s1", turn_id="t1", task_id="t_1", project_key="git:/work",
                  task_kind="engineering", claim_class="tests_pass", actor_kind=actor_kind, actor_id="model-a",
                  provider="p", hermes_version="1.0", result=result)
    return store.record_outcome(**{**fields, **extra})


class _Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_a_phase_one_store_upgrades_and_keeps_everything():
    old = WorkingMemoryStore.open()
    task = old.create_task(project_key="git:/work", lineage_root="s1", origin_session="s1", title="report",
                           goal="write the report", kind="writing", source=_src())
    old.add_item(task.task_id, kind="decision", text="use markdown", source=_src())

    store = ExperienceStore.open()
    exp = _xp(store)

    assert [t.task_id for t in store.tasks(project_key="git:/work")] == [task.task_id]
    assert [i.text for i in store.items(task.task_id)] == ["use markdown"]
    assert store.get_meta("schema_version") == "2"
    # the phase-1 code opening the upgraded file neither breaks nor downgrades it
    assert WorkingMemoryStore.open().get_meta("schema_version") == "2"
    assert ExperienceStore.open().experience(exp.experience_id) == exp


def test_only_the_users_words_or_our_own_checks_put_an_experience_in_force():
    store = ExperienceStore.open()

    with pytest.raises(ProvenanceError):
        _xp(store, status="active", origin=_src("model"))
    with pytest.raises(ProvenanceError):
        _xp(store, scope="global", origin=_src("model"))
    guessed = _xp(store, origin=_src("model"))
    with pytest.raises(ProvenanceError):
        store.set_experience_status(guessed.experience_id, "active", source=_src("model"))
    assert store.set_experience_status(guessed.experience_id, "active", source=_src("tool")).status == "active"
    assert _xp(store, status="active").confirmed_at is not None  # the user said so
    assert _xp(store, scope="global").project_key is None


def test_only_the_user_revokes_and_what_ended_never_comes_back():
    store = ExperienceStore.open()
    exp = _xp(store, status="active")

    for who in ("model", "tool", "system"):
        with pytest.raises(ProvenanceError):
            store.set_experience_status(exp.experience_id, "revoked", source=_src(who))
    revoked = store.set_experience_status(exp.experience_id, "revoked", source=_src())
    assert (revoked.status, revoked.ended_at is not None) == ("revoked", True)
    for status in ("active", "candidate", "stale"):
        with pytest.raises(InvalidTransition):
            store.set_experience_status(exp.experience_id, status, source=_src())
    assert store.experiences(project_key="git:/work", statuses=RECALLED_STATUSES) == []


def test_a_replacement_ends_the_old_experience_and_keeps_it_as_history():
    store = ExperienceStore.open()
    old = _xp(store, status="active", behavior="use `pytest -x`")
    new = _xp(store, status="active", behavior="use `uv run pytest`", supersedes=old.experience_id)

    assert store.experience(old.experience_id).status == "superseded"
    assert new.supersedes == old.experience_id
    assert [e.experience_id for e in store.experiences(project_key="git:/work", statuses=RECALLED_STATUSES)] == [
        new.experience_id]
    with pytest.raises(InvalidTransition):
        _xp(store, supersedes=old.experience_id)  # an ended experience cannot be replaced again


def test_each_piece_of_evidence_counts_once():
    store = ExperienceStore.open()
    exp = _xp(store)

    assert store.add_signal(exp.experience_id, source_class="tool_verified", polarity=1, evidence_key="task:t_1",
                            source=_src("tool"))
    # a retry, a summary or a replay of the same task's result
    assert not store.add_signal(exp.experience_id, source_class="tool_verified", polarity=1, evidence_key="task:t_1",
                                source=_src("tool", turn="t2"))
    assert store.add_signal(exp.experience_id, source_class="tool_verified", polarity=1, evidence_key="task:t_2",
                            source=_src("tool"))
    assert store.signal_summary(exp.experience_id).verifications == 2
    assert _outcome(store, "result:t_1:abc") and not _outcome(store, "result:t_1:abc", turn_id="t2")
    assert len(store.outcomes()) == 1


def test_the_users_voice_and_our_checks_cannot_be_claimed_by_anyone_else():
    store = ExperienceStore.open()
    exp = _xp(store)

    for source_class in ("user_directive", "user_correction", "user_confirmation"):
        with pytest.raises(ProvenanceError):
            store.add_signal(exp.experience_id, source_class=source_class, polarity=1, evidence_key=f"k-{source_class}",
                             source=_src("model"))
    with pytest.raises(ProvenanceError):
        store.add_signal(exp.experience_id, source_class="tool_verified", polarity=1, evidence_key="k-tool",
                         source=_src("model"))
    store.add_signal(exp.experience_id, source_class="subagent_report", polarity=1, evidence_key="k-sub",
                     source=_src("tool"))
    summary = store.signal_summary(exp.experience_id)
    assert (summary.verifications, summary.confirmations) == (0, 0)  # a report is not a check


def test_the_user_speaking_in_two_conversations_is_counted_per_conversation():
    store = ExperienceStore.open()
    exp = _xp(store)

    for key, lineage in (("msg:a:1", "conv-a"), ("msg:a:2", "conv-a"), ("msg:b:1", "conv-b")):
        store.add_signal(exp.experience_id, source_class="user_correction", polarity=1, evidence_key=key,
                         source=_src(), lineage=lineage)
    store.add_signal(exp.experience_id, source_class="user_correction", polarity=-1, evidence_key="msg:c:1",
                     source=_src(), lineage="conv-c")

    summary = store.signal_summary(exp.experience_id)
    assert summary.user_lineages == {"conv-a", "conv-b"}
    assert (summary.confirmations, summary.corrections) == (3, 1)


def test_projects_keep_their_experiences_apart_and_everywhere_ones_are_shared():
    store = ExperienceStore.open()
    here = _xp(store, project="git:/work")
    _xp(store, project="git:/other")
    everywhere = _xp(store, scope="global")

    assert [e.experience_id for e in store.experiences(project_key="git:/work")] == [
        here.experience_id, everywhere.experience_id]
    assert [e.experience_id for e in store.experiences(project_key="git:/work", include_global=False)] == [
        here.experience_id]


def test_a_relabelled_outcome_keeps_its_first_reading():
    store = ExperienceStore.open()
    _outcome(store, "claim:s1:t1:tests_pass")
    (outcome,) = store.outcomes()

    assert store.relabel_outcomes([outcome.outcome_id], "corrected", source=_src(turn="t2"), correction=True) == 1
    assert store.relabel_outcomes([outcome.outcome_id], "success", source=_src(turn="t3")) == 1
    (after,) = store.outcomes()
    assert (after.result, after.prior_result, after.intervention["corrections"]) == ("success", "unknown", 1)
    assert after.relabelled_by.turn_id == "t3"


def test_unsupported_experiences_lapse_and_supported_ones_stay():
    clock = _Clock()
    store = ExperienceStore.open(clock=clock)
    candidate = _xp(store)
    kept = _xp(store, status="active", behavior="use `make test`")
    lapsing = _xp(store, status="active", behavior="use `tox`")

    clock.now += 60 * DAY
    store.add_signal(kept.experience_id, source_class="tool_verified", polarity=1, evidence_key="task:t_9",
                     source=_src("tool"))
    clock.now += 40 * DAY
    store.maintain(expire_after_days=30, stale_after_days=90)

    assert store.experience(candidate.experience_id).status == "expired"
    assert store.experience(kept.experience_id).status == "active"
    assert store.experience(lapsing.experience_id).status == "stale"
    assert store.set_experience_status(lapsing.experience_id, "active", source=_src()).status == "active"


def test_old_outcomes_give_way_to_new_ones():
    clock = _Clock()
    store = ExperienceStore.open(clock=clock, learning_limits=LearningLimits(outcome_rows=3, outcome_days=10))
    _outcome(store, "ancient")
    clock.now += 11 * DAY
    for n in range(4):
        clock.now += 1
        _outcome(store, f"k{n}")

    store.maintain(expire_after_days=30, stale_after_days=90)

    assert [o.evidence_key for o in store.outcomes()] == ["k1", "k2", "k3"]


def test_a_project_keeps_a_bounded_number_of_live_experiences():
    store = ExperienceStore.open(learning_limits=LearningLimits(experiences_per_project=2))
    first = _xp(store)
    _xp(store)

    with pytest.raises(LimitReached):
        _xp(store)
    store.set_experience_status(first.experience_id, "revoked", source=_src())
    assert _xp(store).status == "candidate"  # an ended experience frees its place


def test_each_profile_keeps_its_own_experiences(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _xp(ExperienceStore.open())
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "b"))
    assert ExperienceStore.open().experiences() == []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    assert len(ExperienceStore.open().experiences()) == 1


def _record_outcomes(home: str, worker: int) -> None:
    os.environ["HERMES_HOME"] = home
    store = ExperienceStore.open()
    for n in range(15):
        _outcome(store, f"w{worker}-{n}")
    for n in range(5):
        _outcome(store, f"shared-{n}")  # the same results seen by both processes


def test_two_processes_recording_outcomes_lose_nothing_and_count_nothing_twice():
    store = ExperienceStore.open()
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_record_outcomes, args=(os.environ["HERMES_HOME"], w)) for w in (1, 2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)

    assert [p.exitcode for p in procs] == [0, 0]
    assert len(store.outcomes()) == 35
