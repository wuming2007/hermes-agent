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


# ------------------------------------------------------------------ experiences and outcomes (phase 2)

EXPERIENCE_KINDS = ("preference", "procedure", "fact", "caution", "ask_first")
EXPERIENCE_STATUSES = ("candidate", "active", "stale", "superseded", "revoked", "expired")
ENDED_EXPERIENCE_STATUSES = frozenset({"superseded", "revoked", "expired"})
RECALLED_STATUSES = frozenset({"candidate", "active"})
# Where support for (or doubt about) an experience came from. The user's own words say how they want
# the work done; only our own tool results say it worked; a report or an old record never counts
# by itself.
SOURCE_CLASSES = ("user_directive", "user_correction", "user_confirmation", "tool_verified",
                  "subagent_report", "external_agent_report", "model_claim", "historical_record")
OWNER_SOURCE_CLASSES = frozenset({"user_directive", "user_correction", "user_confirmation"})
OUTCOME_RESULTS = ("success", "failure", "partial", "blocked", "unknown", "reported", "corrected")
ACTOR_KINDS = ("self", "subagent", "external_agent")
SITUATION_MAX, BEHAVIOR_MAX, EXCEPTIONS_MAX = 200, 300, 200


@dataclass(frozen=True)
class Experience:
    experience_id: str
    kind: str
    scope: str  # project | global
    project_key: str | None
    task_kind: str | None
    situation: str
    behavior: str
    exceptions: str
    tags: tuple[str, ...]
    lesson_key: str
    status: str
    supersedes: str | None
    origin: SourceRef
    binds_to: dict
    learned_under: dict
    created_at: float
    updated_at: float
    confirmed_at: float | None
    ended_at: float | None
    revision: int


@dataclass(frozen=True)
class Signal:
    experience_id: str
    source_class: str
    polarity: int  # +1 supports, -1 doubts
    evidence_key: str  # what it rests on; one key counts once
    source: SourceRef
    lineage: str  # the conversation it came from
    detail: str
    at: float


@dataclass(frozen=True)
class Outcome:
    outcome_id: str
    evidence_key: str
    session_id: str
    turn_id: str
    task_id: str | None
    project_key: str
    task_kind: str
    claim_class: str
    actor_kind: str
    actor_id: str
    provider: str
    hermes_version: str
    result: str
    prior_result: str | None  # what it read before the user (or a later check) relabelled it
    reason: str
    evidence_ids: tuple[str, ...]
    intervention: dict
    relabelled_by: SourceRef | None
    at: float
    claim: str = ""  # a report's own words for it (what our later checks must be about to settle it)
