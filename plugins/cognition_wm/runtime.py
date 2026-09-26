"""Per-process plugin state: each profile's store and settings, and what the plugin learned about
each live turn (whose turn it is, which project and conversation it belongs to).

Everything is keyed by the active profile's home, resolved per call: one process may serve
several profiles.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from plugins.cognition_wm.intents import ControlIntent
from plugins.cognition_wm.model import Task
from plugins.cognition_wm.store import Limits, WorkingMemoryStore

_DEFAULTS = {"snapshot_max_chars": 1600, "other_open_tasks_max": 3, "authorization_historical_after_hours": 12,
             "scope_gate": True, "cancelled_under_yolo": "block"}


@dataclass(frozen=True)
class Settings:
    snapshot_max_chars: int = 1600
    other_open_tasks_max: int = 3
    authorization_historical_after_hours: float = 12
    scope_gate: bool = True
    cancelled_under_yolo: str = "block"  # block | approve (D2: block)


@dataclass
class TurnState:
    session_id: str
    turn_id: str
    owner: bool
    project_key: str
    lineage_root: str
    platform: str
    intent: Optional[ControlIntent] = None
    notes: dict = field(default_factory=dict)


_lock = threading.Lock()
_stores: dict[str, WorkingMemoryStore] = {}
_turns: dict[tuple[str, str], TurnState] = {}
_lineage: dict[tuple[str, str], str] = {}


def home_key() -> str:
    from hermes_constants import hermes_home_key

    return hermes_home_key()


def settings() -> Settings:
    from hermes_cli.config import load_config_readonly

    block = ((load_config_readonly() or {}).get("cognition") or {}).get("working_memory") or {}
    merged = {**_DEFAULTS, **(block if isinstance(block, dict) else {})}
    return Settings(**{k: merged[k] for k in _DEFAULTS})


def store() -> WorkingMemoryStore:
    key = home_key()
    with _lock:
        if key not in _stores:
            hours = settings().authorization_historical_after_hours
            _stores[key] = WorkingMemoryStore.open(limits=Limits(historical_after_hours=float(hours)))
        return _stores[key]


def project_key(*, sender_id: str = "", platform: str = "") -> str:
    """A messaging chat is its own project; otherwise the git work tree (or directory) of the cwd."""
    if sender_id:
        return f"chat:{platform}:{sender_id}"
    cwd = Path(os.environ.get("TERMINAL_CWD") or os.getcwd()).resolve()
    top = next((d for d in (cwd, *cwd.parents) if (d / ".git").exists()), None)
    return f"git:{top}" if top is not None else f"cwd:{cwd}"


def lineage_root(session_id: str) -> str:
    """The first session of the conversation this session continues after compaction."""
    key = (home_key(), session_id)
    with _lock:
        if key in _lineage:
            return _lineage[key]
    root = session_id
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    if session_id and db_path.exists():
        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path)
        try:
            chain = db.get_compression_lineage(session_id)
        finally:
            db.close()
        root = chain[0] if chain else session_id
    with _lock:
        _lineage[key] = root
    return root


def remember_turn(state: TurnState) -> None:
    with _lock:
        _turns[(home_key(), state.session_id)] = state


def current_turn(session_id: Optional[str]) -> Optional[TurnState]:
    with _lock:
        return _turns.get((home_key(), session_id or ""))


def clear_intent(session_id: Optional[str]) -> None:
    with _lock:
        key = (home_key(), session_id or "")
        if key in _turns:
            _turns[key] = replace(_turns[key], intent=None)


def focus_task(turn: TurnState) -> Optional[Task]:
    """The conversation's focus task: the one marked, else its most recently touched open task."""
    tasks = [t for t in store().tasks(project_key=turn.project_key, lineage_root=turn.lineage_root, open_only=False)
             if not t.ended or t.focus]
    marked = [t for t in tasks if t.focus]
    if marked:
        return marked[-1]
    live = [t for t in tasks if not t.ended]
    return max(live, key=lambda t: t.updated_at) if live else None
