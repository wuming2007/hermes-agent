"""Experiences and the track record (cognition v2, phase 2), kept in the working-memory database.

Three more tables in the same SQLite file, one short transaction per write, as the working memory:

- ``experiences``: how the user wants a kind of work done, or what worked, with where it applies,
  its exceptions, where it came from, when and under which model and Hermes version it was learned,
  and whether it still holds (candidate → active → stale; superseded, revoked or expired end it).
- ``experience_signals``: every piece of support or doubt, keyed by what it rests on (one tool
  call, one user message, one task), so a summary, a retry or a replay never counts twice.
- ``outcomes``: how each claimed result turned out, per task kind and per executor, model, provider
  and Hermes version — never pooled across them.

Authority follows the working memory's rules: only the owner states, confirms, revokes or widens an
experience to every project; only our own tool results verify; the model's or a subagent's word
makes a candidate and nothing more. Only new tables are added: a phase-1 store keeps working.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from plugins.cognition_wm.model import (
    ACTOR_KINDS,
    BEHAVIOR_MAX,
    ENDED_EXPERIENCE_STATUSES,
    EXCEPTIONS_MAX,
    EXPERIENCE_KINDS,
    OUTCOME_RESULTS,
    OWNER_SOURCE_CLASSES,
    SITUATION_MAX,
    SOURCE_CLASSES,
    Experience,
    Outcome,
    Signal,
    SourceRef,
    clip,
)
from plugins.cognition_wm.store import (
    InvalidTransition,
    LimitReached,
    Limits,
    ProvenanceError,
    RevisionConflict,
    WorkingMemoryError,
    WorkingMemoryStore,
)

SCHEMA_VERSION = "2"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences(
  experience_id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL, project_key TEXT, task_kind TEXT,
  situation TEXT NOT NULL, behavior TEXT NOT NULL, exceptions TEXT NOT NULL DEFAULT '',
  tags TEXT NOT NULL DEFAULT '[]', lesson_key TEXT NOT NULL, status TEXT NOT NULL, supersedes TEXT,
  origin TEXT NOT NULL, binds_to TEXT NOT NULL DEFAULT '{}', learned_under TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL, updated_at REAL NOT NULL, confirmed_at REAL, ended_at REAL, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS experience_signals(
  signal_id INTEGER PRIMARY KEY AUTOINCREMENT, experience_id TEXT NOT NULL, source_class TEXT NOT NULL,
  polarity INTEGER NOT NULL, evidence_key TEXT NOT NULL, source TEXT NOT NULL, lineage TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '', at REAL NOT NULL, UNIQUE(experience_id, evidence_key));
CREATE TABLE IF NOT EXISTS outcomes(
  outcome_id TEXT PRIMARY KEY, evidence_key TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
  task_id TEXT, project_key TEXT NOT NULL, task_kind TEXT NOT NULL, claim_class TEXT NOT NULL,
  actor_kind TEXT NOT NULL, actor_id TEXT NOT NULL, provider TEXT NOT NULL, hermes_version TEXT NOT NULL,
  result TEXT NOT NULL, prior_result TEXT, reason TEXT NOT NULL DEFAULT '', evidence_ids TEXT NOT NULL DEFAULT '[]',
  intervention TEXT NOT NULL DEFAULT '{}', relabelled_by TEXT, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS experiences_by_project ON experiences(project_key, status);
CREATE INDEX IF NOT EXISTS experiences_by_scope ON experiences(scope, status);
CREATE INDEX IF NOT EXISTS signals_by_experience ON experience_signals(experience_id);
CREATE INDEX IF NOT EXISTS outcomes_by_cell ON outcomes(project_key, task_kind, actor_kind, actor_id, hermes_version);
CREATE INDEX IF NOT EXISTS outcomes_by_turn ON outcomes(session_id, turn_id);
CREATE INDEX IF NOT EXISTS outcomes_by_task ON outcomes(task_id);
"""
# status -> the statuses it may move to; who may move it is checked on top
_TRANSITIONS: dict[str, frozenset] = {
    "candidate": frozenset({"active", "superseded", "revoked", "expired"}),
    "active": frozenset({"stale", "superseded", "revoked"}),
    "stale": frozenset({"active", "superseded", "revoked"}),
    "superseded": frozenset(), "revoked": frozenset(), "expired": frozenset(),
}
_LIVE = ("candidate", "active", "stale")
# Who may put an experience into force: the owner's own words, or our own verified tool results.
_ACTIVATORS = frozenset({"owner", "tool"})


@dataclass(frozen=True)
class LearningLimits:
    experiences_per_project: int = 200
    global_experiences: int = 100
    outcome_days: float = 180
    outcome_rows: int = 20_000


@dataclass(frozen=True)
class SignalSummary:
    """What an experience rests on, counted once per piece of evidence."""

    verifications: int = 0  # our own tool results that followed it and worked (one per task)
    corrections: int = 0  # the user's words against it
    confirmations: int = 0  # the user's words for it (stated, repeated or confirmed)
    user_lineages: frozenset = frozenset()  # conversations in which the user spoke for it
    last_support_at: Optional[float] = None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class ExperienceStore(WorkingMemoryStore):
    """The working-memory store, plus experiences, their signals and the track record."""

    def __init__(self, *args: Any, learning_limits: LearningLimits = LearningLimits(), **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._learning_limits = learning_limits

    @classmethod
    def open(cls, *, limits: Limits = Limits(), learning_limits: LearningLimits = LearningLimits(),
             clock=time.time) -> "ExperienceStore":
        store = super().open(limits=limits, clock=clock)
        store._learning_limits = learning_limits
        with store._lock:  # idempotent DDL; executescript commits on its own, so not inside _write()
            store._conn.executescript(_SCHEMA)
        with store._write() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?) ON CONFLICT(key) DO UPDATE"
                         " SET value=excluded.value WHERE CAST(meta.value AS INTEGER) < CAST(excluded.value AS INTEGER)",
                         (SCHEMA_VERSION,))
        return store

    # ------------------------------------------------------------------ experiences

    @staticmethod
    def _to_experience(row) -> Experience:
        return Experience(
            experience_id=row["experience_id"], kind=row["kind"], scope=row["scope"], project_key=row["project_key"],
            task_kind=row["task_kind"], situation=row["situation"], behavior=row["behavior"],
            exceptions=row["exceptions"], tags=tuple(json.loads(row["tags"])), lesson_key=row["lesson_key"],
            status=row["status"], supersedes=row["supersedes"], origin=SourceRef.from_json(row["origin"]),
            binds_to=json.loads(row["binds_to"]), learned_under=json.loads(row["learned_under"]),
            created_at=row["created_at"], updated_at=row["updated_at"], confirmed_at=row["confirmed_at"],
            ended_at=row["ended_at"], revision=row["revision"],
        )

    def experience(self, experience_id: str) -> Optional[Experience]:
        rows = self._read("SELECT * FROM experiences WHERE experience_id=?", (experience_id,))
        return self._to_experience(rows[0]) if rows else None

    def experiences(self, *, project_key: Optional[str] = None, statuses: Optional[Iterable[str]] = None,
                    include_global: bool = True) -> list[Experience]:
        """Experiences of a project (and, unless told otherwise, the ones that hold everywhere)."""
        sql, params = "SELECT * FROM experiences WHERE 1=1", []
        if project_key is not None:
            sql += " AND (project_key=?" + (" OR scope='global')" if include_global else ")")
            params.append(project_key)
        wanted = tuple(statuses) if statuses is not None else None
        if wanted is not None:
            sql += f" AND status IN ({','.join('?' * len(wanted))})" if wanted else " AND 0"
            params += list(wanted)
        return [self._to_experience(r) for r in self._read(sql + " ORDER BY created_at, rowid", params)]

    def add_experience(self, *, kind: str, scope: str, project_key: Optional[str], task_kind: Optional[str],
                       situation: str, behavior: str, exceptions: str = "", tags: Iterable[str] = (),
                       lesson_key: str = "", status: str = "candidate", origin: SourceRef,
                       binds_to: Optional[dict] = None, learned_under: Optional[dict] = None,
                       supersedes: Optional[str] = None) -> Experience:
        """Record an experience; ``supersedes`` ends the one it replaces (kept as history)."""
        if kind not in EXPERIENCE_KINDS:
            raise WorkingMemoryError(f"unknown experience kind {kind!r}")
        if scope not in ("project", "global") or (scope == "project" and not project_key):
            raise WorkingMemoryError("an experience holds for one project (with its key) or for every project")
        if status not in ("candidate", "active"):
            raise InvalidTransition(f"a new experience is a candidate or active, not {status!r}")
        if status == "active" and origin.authored_by not in _ACTIVATORS:
            raise ProvenanceError("only the user's own words or our own verified results put an experience in force")
        if scope == "global" and not origin.is_owner:
            raise ProvenanceError("only the user widens an experience to every project")
        if not str(behavior or "").strip():
            raise WorkingMemoryError("an experience says what to do")
        experience_id, now = _new_id("xp"), self._clock()
        with self._write() as conn:
            where, params = (("scope='global'", ()) if scope == "global" else ("project_key=?", (project_key,)))
            (count,) = conn.execute(
                f"SELECT COUNT(*) FROM experiences WHERE {where} AND status IN ({','.join('?' * len(_LIVE))})",
                (*params, *_LIVE)).fetchone()
            cap = (self._learning_limits.global_experiences if scope == "global"
                   else self._learning_limits.experiences_per_project)
            if count >= cap:
                raise LimitReached(f"{count} experiences kept for this scope; revoke or forget some first")
            if supersedes is not None:
                old = conn.execute("SELECT status FROM experiences WHERE experience_id=?", (supersedes,)).fetchone()
                if old is None or old["status"] in ENDED_EXPERIENCE_STATUSES:
                    raise InvalidTransition(f"experience {supersedes} is not in force")
                conn.execute("UPDATE experiences SET status='superseded', ended_at=?, updated_at=?,"
                             " revision=revision+1 WHERE experience_id=?", (now, now, supersedes))
            conn.execute(
                "INSERT INTO experiences(experience_id, kind, scope, project_key, task_kind, situation, behavior,"
                " exceptions, tags, lesson_key, status, supersedes, origin, binds_to, learned_under, created_at,"
                " updated_at, confirmed_at, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (experience_id, kind, scope, None if scope == "global" else project_key, task_kind,
                 clip(situation, SITUATION_MAX), clip(behavior, BEHAVIOR_MAX), clip(exceptions, EXCEPTIONS_MAX),
                 _dumps(list(dict.fromkeys(tags))), lesson_key, status, supersedes, origin.to_json(),
                 _dumps(binds_to or {}), _dumps(learned_under or {}), now, now,
                 now if status == "active" and origin.is_owner else None))
        return self.experience(experience_id)

    def set_experience_status(self, experience_id: str, status: str, *, source: SourceRef,
                              expected_revision: Optional[int] = None) -> Experience:
        with self._write() as conn:
            row = conn.execute("SELECT * FROM experiences WHERE experience_id=?", (experience_id,)).fetchone()
            if row is None:
                raise WorkingMemoryError(f"unknown experience {experience_id}")
            current = self._to_experience(row)
            if status == current.status:
                return current
            if status not in _TRANSITIONS.get(current.status, frozenset()):
                raise InvalidTransition(f"an experience that is {current.status} cannot become {status}")
            if status == "revoked" and not source.is_owner:
                raise ProvenanceError("revoking an experience is the user's call")
            if status == "active" and source.authored_by not in _ACTIVATORS:
                raise ProvenanceError("only the user's own words or our own verified results put an experience in force")
            if expected_revision is not None and current.revision != expected_revision:
                raise RevisionConflict(f"experience {experience_id} is at revision {current.revision}")
            now = self._clock()
            conn.execute(
                "UPDATE experiences SET status=?, updated_at=?, revision=revision+1,"
                " confirmed_at=CASE WHEN ? THEN ? ELSE confirmed_at END,"
                " ended_at=CASE WHEN ? THEN ? ELSE ended_at END WHERE experience_id=?",
                (status, now, status == "active" and source.is_owner, now, status in ENDED_EXPERIENCE_STATUSES, now,
                 experience_id))
        return self.experience(experience_id)

    def forget_experience(self, experience_id: str) -> bool:
        """Delete an experience and its signals for good (the user's explicit request only)."""
        with self._write() as conn:
            conn.execute("DELETE FROM experience_signals WHERE experience_id=?", (experience_id,))
            return conn.execute("DELETE FROM experiences WHERE experience_id=?", (experience_id,)).rowcount == 1

    def forget_experiences(self, *, project_key: Optional[str] = None) -> int:
        """Delete every experience (of one project, or all) and their signals for good."""
        where, params = ("project_key=?", (project_key,)) if project_key is not None else ("1=1", ())
        with self._write() as conn:
            conn.execute("DELETE FROM experience_signals WHERE experience_id IN"
                         f" (SELECT experience_id FROM experiences WHERE {where})", params)
            return conn.execute(f"DELETE FROM experiences WHERE {where}", params).rowcount

    # ------------------------------------------------------------------ signals

    def add_signal(self, experience_id: str, *, source_class: str, polarity: int, evidence_key: str,
                   source: SourceRef, lineage: str = "", detail: str = "") -> bool:
        """Record support (+1) or doubt (-1); False when that evidence already counted."""
        if source_class not in SOURCE_CLASSES:
            raise WorkingMemoryError(f"unknown source class {source_class!r}")
        if source_class in OWNER_SOURCE_CLASSES and not source.is_owner:
            raise ProvenanceError(f"{source_class} comes only from the user's own words")
        if source_class == "tool_verified" and source.authored_by != "tool":
            raise ProvenanceError("tool_verified comes only from our own tool results")
        if polarity not in (1, -1) or not evidence_key:
            raise WorkingMemoryError("a signal is +1 or -1 and names what it rests on")
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM experiences WHERE experience_id=?", (experience_id,)).fetchone() is None:
                raise WorkingMemoryError(f"unknown experience {experience_id}")
            inserted = conn.execute(
                "INSERT OR IGNORE INTO experience_signals(experience_id, source_class, polarity, evidence_key, source,"
                " lineage, detail, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (experience_id, source_class, polarity, evidence_key, source.to_json(), lineage,
                 clip(detail, 200), self._clock())).rowcount
        return inserted == 1

    def signals(self, experience_id: str) -> list[Signal]:
        rows = self._read("SELECT * FROM experience_signals WHERE experience_id=? ORDER BY signal_id", (experience_id,))
        return [Signal(experience_id=r["experience_id"], source_class=r["source_class"], polarity=r["polarity"],
                       evidence_key=r["evidence_key"], source=SourceRef.from_json(r["source"]), lineage=r["lineage"],
                       detail=r["detail"], at=r["at"]) for r in rows]

    def signal_summary(self, experience_id: str) -> SignalSummary:
        found = self.signals(experience_id)
        exp = self.experience(experience_id)
        support = [s.at for s in found if s.polarity > 0]
        if exp is not None:
            support += [exp.created_at, *([exp.confirmed_at] if exp.confirmed_at else [])]
        return SignalSummary(
            verifications=sum(s.polarity > 0 and s.source_class == "tool_verified" for s in found),
            corrections=sum(s.polarity < 0 and s.source_class in OWNER_SOURCE_CLASSES for s in found),
            confirmations=sum(s.polarity > 0 and s.source_class in OWNER_SOURCE_CLASSES for s in found),
            user_lineages=frozenset(s.lineage for s in found
                                    if s.polarity > 0 and s.source_class in OWNER_SOURCE_CLASSES and s.lineage),
            last_support_at=max(support) if support else None,
        )

    # ------------------------------------------------------------------ outcomes

    @staticmethod
    def _to_outcome(row) -> Outcome:
        return Outcome(
            outcome_id=row["outcome_id"], evidence_key=row["evidence_key"], session_id=row["session_id"],
            turn_id=row["turn_id"], task_id=row["task_id"], project_key=row["project_key"],
            task_kind=row["task_kind"], claim_class=row["claim_class"], actor_kind=row["actor_kind"],
            actor_id=row["actor_id"], provider=row["provider"], hermes_version=row["hermes_version"],
            result=row["result"], prior_result=row["prior_result"], reason=row["reason"],
            evidence_ids=tuple(json.loads(row["evidence_ids"])), intervention=json.loads(row["intervention"]),
            relabelled_by=SourceRef.from_json(row["relabelled_by"]) if row["relabelled_by"] else None, at=row["at"],
        )

    def record_outcome(self, *, evidence_key: str, session_id: str, turn_id: str, task_id: Optional[str],
                       project_key: str, task_kind: str, claim_class: str, actor_kind: str, actor_id: str,
                       provider: str, hermes_version: str, result: str, reason: str = "",
                       evidence_ids: Iterable[str] = (), intervention: Optional[dict] = None,
                       at: Optional[float] = None) -> bool:
        """Record one claimed result; False when the same evidence was already recorded."""
        if result not in OUTCOME_RESULTS or actor_kind not in ACTOR_KINDS:
            raise WorkingMemoryError(f"unknown outcome {result!r} for {actor_kind!r}")
        with self._write() as conn:
            return conn.execute(
                "INSERT OR IGNORE INTO outcomes(outcome_id, evidence_key, session_id, turn_id, task_id, project_key,"
                " task_kind, claim_class, actor_kind, actor_id, provider, hermes_version, result, reason, evidence_ids,"
                " intervention, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_new_id("oc"), evidence_key, session_id, turn_id, task_id, project_key, task_kind, claim_class,
                 actor_kind, actor_id, provider, hermes_version, result, reason, _dumps(list(evidence_ids)),
                 _dumps(intervention or {}), self._clock() if at is None else at)).rowcount == 1

    def outcomes(self, *, project_key: Optional[str] = None, task_kind: Optional[str] = None,
                 claim_class: Optional[str] = None, actor_kind: Optional[str] = None, actor_id: Optional[str] = None,
                 provider: Optional[str] = None, hermes_version: Optional[str] = None,
                 session_id: Optional[str] = None, turn_id: Optional[str] = None, task_id: Optional[str] = None,
                 results: Optional[Iterable[str]] = None) -> list[Outcome]:
        """Outcomes matching every filter given, oldest first."""
        filters = {"project_key": project_key, "task_kind": task_kind, "claim_class": claim_class,
                   "actor_kind": actor_kind, "actor_id": actor_id, "provider": provider,
                   "hermes_version": hermes_version, "session_id": session_id, "turn_id": turn_id, "task_id": task_id}
        clauses = [f"{column}=?" for column, value in filters.items() if value is not None]
        params: list[Any] = [value for value in filters.values() if value is not None]
        if results is not None:
            wanted = tuple(results)
            clauses.append(f"result IN ({','.join('?' * len(wanted))})" if wanted else "0")
            params += list(wanted)
        where = " AND ".join(clauses) or "1=1"
        return [self._to_outcome(r) for r in self._read(f"SELECT * FROM outcomes WHERE {where} ORDER BY at, rowid",
                                                        params)]

    def relabel_outcomes(self, outcome_ids: Iterable[str], result: str, *, source: Optional[SourceRef],
                         correction: bool = False) -> int:
        """Change what outcomes read (a later correction, confirmation or check), keeping the first
        reading in ``prior_result``; a correction also counts toward the intervention."""
        if result not in OUTCOME_RESULTS:
            raise WorkingMemoryError(f"unknown outcome {result!r}")
        changed = 0
        with self._write() as conn:
            for outcome_id in dict.fromkeys(outcome_ids):
                row = conn.execute("SELECT * FROM outcomes WHERE outcome_id=?", (outcome_id,)).fetchone()
                if row is None or row["result"] == result:
                    continue
                intervention = json.loads(row["intervention"])
                if correction:
                    intervention["corrections"] = int(intervention.get("corrections") or 0) + 1
                conn.execute("UPDATE outcomes SET result=?, prior_result=COALESCE(prior_result, ?), intervention=?,"
                             " relabelled_by=? WHERE outcome_id=?",
                             (result, row["result"], _dumps(intervention), source.to_json() if source else None,
                              outcome_id))
                changed += 1
        return changed

    def forget_outcomes(self, *, project_key: Optional[str] = None) -> int:
        where, params = ("project_key=?", (project_key,)) if project_key is not None else ("1=1", ())
        with self._write() as conn:
            return conn.execute(f"DELETE FROM outcomes WHERE {where}", params).rowcount

    def verified_evidence_ids(self, task_id: str) -> set[str]:
        """Evidence of a task that our own tools produced and that ended well."""
        rows = self._read("SELECT evidence_id FROM evidence WHERE task_id=? AND status='ok' AND provenance='self'",
                          (task_id,))
        return {row["evidence_id"] for row in rows}

    def owner_turns(self, task_id: str) -> int:
        """How many of the user's own turns went into a task (a cost of the work to them)."""
        turns = set()
        for event in self.events(task_id):
            source = json.loads(event["source"])
            if source.get("authored_by") == "owner":
                turns.add((source.get("session_id"), source.get("turn_id")))
        return len(turns)

    # ------------------------------------------------------------------ upkeep

    def maintain(self, *, expire_after_days: float, stale_after_days: float) -> dict[str, int]:
        """Let unsupported experiences lapse (a candidate expires, an active one goes stale) and drop
        outcomes past their retention, then the oldest over the cap."""
        now = self._clock()
        lapsed = {"expired": 0, "stale": 0}
        for exp in self.experiences(statuses=("candidate", "active")):
            last = self.signal_summary(exp.experience_id).last_support_at or exp.created_at
            limit, status = ((expire_after_days, "expired") if exp.status == "candidate"
                             else (stale_after_days, "stale"))
            if now - last > limit * 86400:
                with self._write() as conn:
                    conn.execute("UPDATE experiences SET status=?, updated_at=?, revision=revision+1,"
                                 " ended_at=CASE WHEN ?='expired' THEN ? ELSE ended_at END"
                                 " WHERE experience_id=? AND status=?",
                                 (status, now, status, now, exp.experience_id, exp.status))
                lapsed[status] += 1
        cutoff = now - self._learning_limits.outcome_days * 86400
        with self._write() as conn:
            removed = conn.execute("DELETE FROM outcomes WHERE at < ?", (cutoff,)).rowcount
            (count,) = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()
            excess = max(0, count - self._learning_limits.outcome_rows)
            if excess:
                conn.execute("DELETE FROM outcomes WHERE outcome_id IN"
                             " (SELECT outcome_id FROM outcomes ORDER BY at LIMIT ?)", (excess,))
        return {**lapsed, "outcomes_dropped": removed + excess}
