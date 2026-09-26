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

# Plugin settings (plugins.entries.cognition_wm.settings, declared in plugin.yaml config_schema).
_DEFAULTS = {"snapshot_max_chars": 1600, "other_open_tasks_max": 3, "authorization_historical_after_hours": 12,
             "scope_gate": True, "cancelled_under_yolo": "block"}
_context = None  # the PluginContext register() received; its get_config reads the active profile's file


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
_chains: dict[tuple[str, str], tuple[str, ...]] = {}
_pruned: set[str] = set()


def home_key() -> str:
    from hermes_constants import hermes_home_key

    return hermes_home_key()


def bind(context) -> None:
    global _context
    _context = context


def settings() -> Settings:
    if _context is None:
        return Settings()
    return Settings(**{key: _context.get_config(key, default) for key, default in _DEFAULTS.items()})


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


def _compression_chain(session_id: str) -> tuple[str, ...]:
    """The sessions this one continues through compaction, oldest first, ending with itself. A
    session's ancestors never change, so the chain is read once per process."""
    key = (home_key(), session_id)
    with _lock:
        if key in _chains:
            return _chains[key]
    chain: tuple[str, ...] = (session_id,)
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    if session_id and db_path.exists():
        from hermes_state import SessionDB

        db = SessionDB(db_path=db_path)
        try:
            chain = tuple(db.get_compression_lineage(session_id)) or chain
        finally:
            db.close()
    with _lock:
        _chains[key] = chain
    return chain


def lineage_root(session_id: str) -> str:
    """The first session of the conversation this session continues after compaction."""
    return _compression_chain(session_id)[0]


def prune_once() -> None:
    """Apply evidence retention the first time this process touches a profile's store."""
    key = home_key()
    with _lock:
        if key in _pruned:
            return
        _pruned.add(key)
    store().prune_evidence()


def remember_turn(state: TurnState) -> None:
    with _lock:
        _turns[(home_key(), state.session_id)] = state


def current_turn(session_id: Optional[str]) -> Optional[TurnState]:
    """The live turn of a session. A rotating compaction (``compression.in_place: false``) moves a
    turn to a child session halfway through; the child then carries on its parent's turn."""
    home = home_key()
    with _lock:
        state = _turns.get((home, session_id or ""))
    if state is not None or not session_id:
        return state
    for ancestor in reversed(_compression_chain(session_id)[:-1]):
        with _lock:
            inherited = _turns.get((home, ancestor))
            if inherited is not None:
                return _turns.setdefault((home, session_id), replace(inherited, session_id=session_id))
    return None


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
