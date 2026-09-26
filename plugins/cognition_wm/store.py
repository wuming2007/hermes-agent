"""Working-memory store: SQLite in the profile's plugin-data dir, one short transaction per write.

Every change to a task bumps its revision and appends an event. Writers pass the revision they
read (compare-and-set), so two processes working on one task never overwrite each other: the
loser re-reads and retries. A replaced item stays as history, authority comes only from the
owner, and a grant from an earlier run reads as historical.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from plugins.cognition_wm.model import (
    ACTION_CLASSES,
    ENDED_STATUSES,
    GOAL_MAX,
    ITEM_STATES,
    ITEM_TEXT_MAX,
    OWNER_ONLY_STATUSES,
    SUPERSEDED_STATE,
    TASK_KINDS,
    TASK_STATUSES,
    TITLE_MAX,
    SourceRef,
    Task,
    TaskItem,
    clip,
)

PLUGIN_NAME = "cognition_wm"
DB_FILE = "working_memory.db"
SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(
  task_id TEXT PRIMARY KEY, project_key TEXT NOT NULL, lineage_root TEXT NOT NULL,
  origin_session TEXT NOT NULL, title TEXT NOT NULL, goal TEXT NOT NULL, kind TEXT NOT NULL,
  status TEXT NOT NULL, status_reason TEXT NOT NULL DEFAULT '', status_source TEXT NOT NULL,
  focus INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS task_items(
  item_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
  state TEXT NOT NULL, attrs TEXT NOT NULL DEFAULT '{}', source TEXT NOT NULL,
  evidence_ids TEXT NOT NULL DEFAULT '[]', supersedes TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS task_events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, at REAL NOT NULL,
  op TEXT NOT NULL, payload TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence(
  evidence_id TEXT PRIMARY KEY, task_id TEXT, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
  tool_call_id TEXT, tool_name TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  target TEXT NOT NULL DEFAULT '', scope TEXT NOT NULL DEFAULT '{}', excerpt TEXT NOT NULL DEFAULT '',
  provenance TEXT NOT NULL, recorded_at REAL NOT NULL, duration_ms INTEGER);
CREATE INDEX IF NOT EXISTS tasks_by_project ON tasks(project_key, lineage_root, status);
CREATE INDEX IF NOT EXISTS items_by_task ON task_items(task_id, kind);
CREATE INDEX IF NOT EXISTS evidence_by_turn ON evidence(session_id, turn_id);
CREATE INDEX IF NOT EXISTS evidence_by_task ON evidence(task_id);
"""
_ENDED = tuple(sorted(ENDED_STATUSES))
_TARGET_MAX, _EXCERPT_MAX = 200, 400


class WorkingMemoryError(Exception):
    """A write the store refused; the caller reports it to the model instead of crashing."""


class RevisionConflict(WorkingMemoryError):
    """The task changed since it was read: re-read and try again."""


class ProvenanceError(WorkingMemoryError):
    """The change needs the owner's own turn."""


class LimitReached(WorkingMemoryError):
    """A size cap would be exceeded."""


class InvalidTransition(WorkingMemoryError):
    """The state change is not allowed from where the task or item stands."""


@dataclass(frozen=True)
class Limits:
    open_tasks_per_project: int = 50
    items_per_task: int = 200
    evidence_days: float = 30
    evidence_rows: int = 10_000
    historical_after_hours: float = 12


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class WorkingMemoryStore:
    def __init__(self, conn: sqlite3.Connection, path: Path, *, limits: Limits = Limits(),
                 clock: Callable[[], float] = time.time):
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # explicit BEGIN IMMEDIATE / COMMIT below
        self._lock = threading.Lock()
        self._limits = limits
        self._clock = clock
        self.path = path

    @classmethod
    def open(cls, *, limits: Limits = Limits(), clock: Callable[[], float] = time.time) -> "WorkingMemoryStore":
        """Open the active profile's store (resolved per call, so profile switches are followed)."""
        from plugins.plugin_storage import plugin_data_dir, plugin_db

        store = cls(plugin_db(PLUGIN_NAME, DB_FILE), plugin_data_dir(PLUGIN_NAME) / DB_FILE, limits=limits, clock=clock)
        with store._lock:  # idempotent DDL; executescript commits on its own, so not inside _write()
            store._conn.executescript(_SCHEMA)
        with store._write() as conn:
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        return store

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _read(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    # ------------------------------------------------------------------ tasks

    @staticmethod
    def _to_task(row: sqlite3.Row) -> Task:
        return Task(
            task_id=row["task_id"], project_key=row["project_key"], lineage_root=row["lineage_root"],
            origin_session=row["origin_session"], title=row["title"], goal=row["goal"], kind=row["kind"],
            status=row["status"], status_reason=row["status_reason"],
            status_source=SourceRef.from_json(row["status_source"]), focus=bool(row["focus"]),
            revision=row["revision"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def _task_in(self, conn: sqlite3.Connection, task_id: str) -> Task:
        row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise WorkingMemoryError(f"unknown task {task_id}")
        return self._to_task(row)

    def _bump(self, conn: sqlite3.Connection, task: Task, expected_revision: Optional[int], op: str,
              payload: dict, source: SourceRef) -> None:
        if expected_revision is not None and task.revision != expected_revision:
            raise RevisionConflict(f"task {task.task_id} is at revision {task.revision}, not {expected_revision}")
        now = self._clock()
        conn.execute("UPDATE tasks SET revision=revision+1, updated_at=? WHERE task_id=?", (now, task.task_id))
        conn.execute("INSERT INTO task_events(task_id, at, op, payload, source) VALUES (?, ?, ?, ?, ?)",
                     (task.task_id, now, op, json.dumps(payload, ensure_ascii=False), source.to_json()))

    def task(self, task_id: str) -> Optional[Task]:
        rows = self._read("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        return self._to_task(rows[0]) if rows else None

    def tasks(self, *, project_key: str, lineage_root: Optional[str] = None, open_only: bool = True) -> list[Task]:
        sql, params = "SELECT * FROM tasks WHERE project_key=?", [project_key]
        if lineage_root is not None:
            sql, params = sql + " AND lineage_root=?", [*params, lineage_root]
        if open_only:
            sql, params = sql + f" AND status NOT IN ({','.join('?' * len(_ENDED))})", [*params, *_ENDED]
        return [self._to_task(r) for r in self._read(sql + " ORDER BY created_at, rowid", params)]

    def create_task(self, *, project_key: str, lineage_root: str, origin_session: str, title: str, goal: str,
                    kind: str, source: SourceRef) -> Task:
        if not source.is_owner:
            raise ProvenanceError("a task is created on the owner's turn")
        task_id, now = _new_id("t"), self._clock()
        with self._write() as conn:
            (open_count,) = conn.execute(
                f"SELECT COUNT(*) FROM tasks WHERE project_key=? AND status NOT IN ({','.join('?' * len(_ENDED))})",
                (project_key, *_ENDED)).fetchone()
            if open_count >= self._limits.open_tasks_per_project:
                raise LimitReached(f"{open_count} open tasks in this project; finish or cancel one first")
            conn.execute(
                "INSERT INTO tasks(task_id, project_key, lineage_root, origin_session, title, goal, kind, status,"
                " status_source, revision, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, 1, ?, ?)",
                (task_id, project_key, lineage_root, origin_session, clip(title, TITLE_MAX), clip(goal, GOAL_MAX),
                 kind if kind in TASK_KINDS else "other", source.to_json(), now, now))
            conn.execute("INSERT INTO task_events(task_id, at, op, payload, source) VALUES (?, ?, 'create', '{}', ?)",
                         (task_id, now, source.to_json()))
        return self.task(task_id)

    def set_status(self, task_id: str, status: str, *, source: SourceRef, expected_revision: Optional[int],
                   reason: str = "") -> Task:
        if status not in TASK_STATUSES:
            raise InvalidTransition(f"unknown status {status!r}")
        with self._write() as conn:
            task = self._task_in(conn, task_id)
            stopped = task.status in ("paused", "cancelled")
            if not source.is_owner and (status in OWNER_ONLY_STATUSES or stopped):
                raise ProvenanceError(f"moving a task to {status!r} or out of {task.status!r} is the owner's call")
            same_turn = (task.status_source.session_id, task.status_source.turn_id) == (source.session_id, source.turn_id)
            if stopped and status == "active" and same_turn:
                raise InvalidTransition(f"a task {task.status} in this turn comes back only on a later turn")
            self._bump(conn, task, expected_revision, "set_status", {"status": status, "reason": reason}, source)
            conn.execute(
                "UPDATE tasks SET status=?, status_reason=?, status_source=?,"
                " focus=CASE WHEN ? THEN 0 ELSE focus END WHERE task_id=?",
                (status, clip(reason, ITEM_TEXT_MAX), source.to_json(), status in ENDED_STATUSES, task_id))
        return self.task(task_id)

    def set_focus(self, task_id: str, *, source: SourceRef, expected_revision: Optional[int] = None) -> Task:
        """Make ``task_id`` the focus of its lineage; the lineage's previous focus loses it."""
        with self._write() as conn:
            task = self._task_in(conn, task_id)
            if task.ended:
                raise InvalidTransition(f"task {task_id} has ended")
            others = conn.execute("SELECT * FROM tasks WHERE lineage_root=? AND focus=1 AND task_id<>?",
                                  (task.lineage_root, task_id)).fetchall()
            for other in others:
                self._bump(conn, self._to_task(other), None, "unfocus", {"focus": task_id}, source)
            conn.execute("UPDATE tasks SET focus=0 WHERE lineage_root=? AND task_id<>?", (task.lineage_root, task_id))
            self._bump(conn, task, expected_revision, "focus", {}, source)
            conn.execute("UPDATE tasks SET focus=1 WHERE task_id=?", (task_id,))
        return self.task(task_id)

    # ------------------------------------------------------------------ items

    @staticmethod
    def _to_item(row: sqlite3.Row) -> TaskItem:
        return TaskItem(
            item_id=row["item_id"], task_id=row["task_id"], kind=row["kind"], text=row["text"], state=row["state"],
            source=SourceRef.from_json(row["source"]), attrs=json.loads(row["attrs"]),
            evidence_ids=tuple(json.loads(row["evidence_ids"])), supersedes=row["supersedes"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def items(self, task_id: str, kinds: Optional[Iterable[str]] = None) -> list[TaskItem]:
        rows = self._read("SELECT * FROM task_items WHERE task_id=? ORDER BY created_at, rowid", (task_id,))
        wanted = set(kinds) if kinds is not None else None
        return [self._to_item(r) for r in rows if wanted is None or r["kind"] in wanted]

    def add_item(self, task_id: str, *, kind: str, text: str, source: SourceRef, state: Optional[str] = None,
                 attrs: Optional[dict] = None, evidence_ids: Iterable[str] = (), supersedes: Optional[str] = None,
                 expected_revision: Optional[int] = None) -> TaskItem:
        states = ITEM_STATES.get(kind)
        if states is None:
            raise WorkingMemoryError(f"unknown item kind {kind!r}")
        state = state or states[0]
        if state not in states:
            raise InvalidTransition(f"{kind} items cannot be {state!r}")
        attrs = dict(attrs or {})
        if kind == "authorization":
            if not source.is_owner:
                raise ProvenanceError("only the owner grants, forbids or limits an action")
            if attrs.get("action_class") not in ACTION_CLASSES:
                raise WorkingMemoryError(f"authorization needs an action_class in {ACTION_CLASSES}")
            attrs.update(run_id=source.run_id, granted_at=self._clock())
        item_id, now = _new_id("i"), self._clock()
        with self._write() as conn:
            task = self._task_in(conn, task_id)
            (count,) = conn.execute("SELECT COUNT(*) FROM task_items WHERE task_id=?", (task_id,)).fetchone()
            if count >= self._limits.items_per_task:
                raise LimitReached(f"task {task_id} already holds {count} items")
            if supersedes is not None:
                old = conn.execute("SELECT kind FROM task_items WHERE item_id=? AND task_id=?",
                                   (supersedes, task_id)).fetchone()
                if old is None:
                    raise WorkingMemoryError(f"unknown item {supersedes} in task {task_id}")
                conn.execute("UPDATE task_items SET state=?, updated_at=? WHERE item_id=?",
                             (SUPERSEDED_STATE[old["kind"]], now, supersedes))
            self._bump(conn, task, expected_revision, "add_item",
                       {"item_id": item_id, "kind": kind, "supersedes": supersedes}, source)
            conn.execute(
                "INSERT INTO task_items(item_id, task_id, kind, text, state, attrs, source, evidence_ids, supersedes,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, task_id, kind, clip(text, ITEM_TEXT_MAX), state, json.dumps(attrs, ensure_ascii=False),
                 source.to_json(), json.dumps(list(evidence_ids)), supersedes, now, now))
        return self._to_item(self._read("SELECT * FROM task_items WHERE item_id=?", (item_id,))[0])

    def set_item_state(self, item_id: str, state: str, *, source: SourceRef, evidence_ids: Iterable[str] = (),
                       expected_revision: Optional[int] = None) -> TaskItem:
        with self._write() as conn:
            row = conn.execute("SELECT * FROM task_items WHERE item_id=?", (item_id,)).fetchone()
            if row is None:
                raise WorkingMemoryError(f"unknown item {item_id}")
            item = self._to_item(row)
            if state not in ITEM_STATES[item.kind]:
                raise InvalidTransition(f"{item.kind} items cannot be {state!r}")
            if item.kind == "authorization" and not source.is_owner:
                raise ProvenanceError("only the owner changes an authorization")
            merged = list(dict.fromkeys([*item.evidence_ids, *evidence_ids]))
            self._bump(conn, self._task_in(conn, item.task_id), expected_revision, "set_item_state",
                       {"item_id": item_id, "state": state}, source)
            conn.execute("UPDATE task_items SET state=?, evidence_ids=?, updated_at=? WHERE item_id=?",
                         (state, json.dumps(merged), self._clock(), item_id))
        return self._to_item(self._read("SELECT * FROM task_items WHERE item_id=?", (item_id,))[0])

    def effective_authorizations(self, task_id: str, *, run_id: str) -> dict[str, str]:
        """action_class -> state, the latest item per class winning. A grant made in an earlier run
        or longer ago than ``historical_after_hours`` is ``historical``: it must be asked again."""
        horizon = self._limits.historical_after_hours * 3600
        now = self._clock()
        result: dict[str, str] = {}
        for item in self.items(task_id, kinds=("authorization",)):
            if item.state == "revoked":
                result.pop(item.attrs.get("action_class"), None)
                continue
            stale = item.attrs.get("run_id") != run_id or now - float(item.attrs.get("granted_at") or 0) > horizon
            result[item.attrs["action_class"]] = "historical" if item.state == "granted" and stale else item.state
        return result

    def events(self, task_id: str) -> list[dict]:
        return [dict(r) for r in self._read("SELECT * FROM task_events WHERE task_id=? ORDER BY event_id", (task_id,))]

    # ------------------------------------------------------------------ evidence

    def record_evidence(self, *, evidence_id: str, session_id: str, turn_id: str, tool_name: str, kind: str,
                        status: str, provenance: str, recorded_at: Optional[float] = None,
                        tool_call_id: Optional[str] = None, task_id: Optional[str] = None, target: str = "",
                        scope: Optional[dict] = None, excerpt: str = "", duration_ms: Optional[int] = None) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evidence(evidence_id, task_id, session_id, turn_id, tool_call_id, tool_name,"
                " kind, status, target, scope, excerpt, provenance, recorded_at, duration_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (evidence_id, task_id, session_id, turn_id, tool_call_id, tool_name, kind, status,
                 clip(target, _TARGET_MAX), json.dumps(scope or {}, ensure_ascii=False), clip(excerpt, _EXCERPT_MAX),
                 provenance, self._clock() if recorded_at is None else recorded_at, duration_ms))

    def link_evidence(self, evidence_id: str, task_id: str) -> None:
        with self._write() as conn:
            conn.execute("UPDATE evidence SET task_id=? WHERE evidence_id=?", (task_id, evidence_id))

    def evidence_ids(self, task_id: Optional[str] = None) -> list[str]:
        if task_id is None:
            return [r["evidence_id"] for r in self._read("SELECT evidence_id FROM evidence ORDER BY recorded_at")]
        return [r["evidence_id"] for r in
                self._read("SELECT evidence_id FROM evidence WHERE task_id=? ORDER BY recorded_at", (task_id,))]

    def prune_evidence(self) -> int:
        """Drop evidence past its retention, then the oldest rows over the cap — ended or unlinked
        tasks' evidence first, so a live task keeps what backs it."""
        cutoff = self._clock() - self._limits.evidence_days * 86400
        with self._write() as conn:
            removed = conn.execute("DELETE FROM evidence WHERE recorded_at < ?", (cutoff,)).rowcount
            (count,) = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()
            excess = max(0, count - self._limits.evidence_rows)
            if excess:
                conn.execute(
                    "DELETE FROM evidence WHERE evidence_id IN (SELECT e.evidence_id FROM evidence e"
                    " LEFT JOIN tasks t ON t.task_id = e.task_id ORDER BY"
                    f" CASE WHEN t.task_id IS NULL OR t.status IN ({','.join('?' * len(_ENDED))}) THEN 0 ELSE 1 END,"
                    " e.recorded_at LIMIT ?)", (*_ENDED, excess))
        return removed + excess
