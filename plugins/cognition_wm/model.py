"""Working-memory data model: tasks, their items, and where every state came from."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SOURCE_TYPES = frozenset({"user_message", "tool_result", "assistant", "subagent", "document", "system"})
AUTHORS = frozenset({"owner", "model", "tool", "system"})
TASK_KINDS = ("engineering", "ops", "research", "writing", "other")
TASK_STATUSES = ("active", "paused", "cancelled", "done", "superseded")
ENDED_STATUSES = frozenset({"done", "cancelled", "superseded"})
# Only the owner moves a task into these: pausing and cancelling are the user's call, and so is
# bringing a task back.
OWNER_ONLY_STATUSES = frozenset({"active", "paused", "cancelled"})

# kind -> allowed states, the first being the default.
ITEM_STATES: dict[str, tuple[str, ...]] = {
    "deliverable": ("pending", "in_progress", "done", "dropped"),
    "progress": ("pending", "in_progress", "done", "dropped"),
    "decision": ("in_force", "reversed"),
    "rejected_option": ("in_force", "reversed"),
    "fact": ("current", "superseded", "retracted"),
    "assumption": ("unverified", "confirmed", "refuted"),
    # "historical" is derived at read time (earlier run or too old), never stored.
    "authorization": ("granted", "forbidden", "ask_first", "revoked"),
    "constraint": ("in_force", "lifted"),
}
# What a replaced item turns into; the replaced row stays as history.
SUPERSEDED_STATE = {
    "deliverable": "dropped", "progress": "dropped", "decision": "reversed", "rejected_option": "reversed",
    "fact": "superseded", "assumption": "refuted", "authorization": "revoked", "constraint": "lifted",
}
ACTION_CLASSES = ("code_change", "repo_write", "coding_agent_launch", "external_message", "deploy_restart",
                  "data_migration", "other")

TITLE_MAX, GOAL_MAX, ITEM_TEXT_MAX = 120, 500, 300


def clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class SourceRef:
    """Where a state came from. ``authored_by="owner"`` is set only by the plugin for a turn whose
    input is the user's own words; tool arguments can never claim it."""

    type: str
    session_id: str
    turn_id: str
    ref: str
    at: float
    authored_by: str
    run_id: str

    def __post_init__(self) -> None:
        if self.type not in SOURCE_TYPES:
            raise ValueError(f"unknown source type {self.type!r}")
        if self.authored_by not in AUTHORS:
            raise ValueError(f"unknown author {self.authored_by!r}")

    @property
    def is_owner(self) -> bool:
        return self.authored_by == "owner"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SourceRef":
        data = json.loads(text)
        return cls(**{k: data[k] for k in ("type", "session_id", "turn_id", "ref", "at", "authored_by", "run_id")})


@dataclass(frozen=True)
class Task:
    task_id: str
    project_key: str
    lineage_root: str
    origin_session: str
    title: str
    goal: str
    kind: str
    status: str
    status_reason: str
    status_source: SourceRef
    focus: bool
    revision: int
    created_at: float
    updated_at: float

    @property
    def ended(self) -> bool:
        return self.status in ENDED_STATUSES


@dataclass(frozen=True)
class TaskItem:
    item_id: str
    task_id: str
    kind: str
    text: str
    state: str
    source: SourceRef
    attrs: dict = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
