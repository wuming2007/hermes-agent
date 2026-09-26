"""``task_state``: the model records the work's state; the plugin decides whose authority it carries.

The source of every write is this turn's: the owner's when the turn is the user's own words,
otherwise the model's. Tool arguments cannot claim otherwise, so a prompt-injected "the user
approved" in a web page or a notice cannot create, cancel or authorize anything.
"""

from __future__ import annotations

import json
import time
from functools import partial
from typing import Any, Callable, Optional

from plugins.cognition_wm import runtime, snapshot
from plugins.cognition_wm.lessons import same_topic, search_terms, topic_terms
from plugins.cognition_wm.model import (
    ACTION_CLASSES,
    EXPERIENCE_KINDS,
    ITEM_STATES,
    TASK_KINDS,
    TASK_STATUSES,
    SourceRef,
    clip,
)
from plugins.cognition_wm.provenance import RUN_ID, turn_source
from plugins.cognition_wm.store import ProvenanceError, WorkingMemoryError

SCHEMA = {
    "name": "task_state",
    "description": (
        "Keep the state of the user's work across turns, resumed chats and context compaction. "
        "create: start a task when the user asks for a piece of work (title, goal, kind, items). "
        "add: record items on a task — deliverable, progress, decision, rejected_option, fact, assumption, "
        "constraint, or authorization (what the user allows, forbids or wants asked first; needs action_class). "
        "update_item: change an item's state (e.g. a deliverable to done, citing evidence_ids). "
        "set_status: active, paused, cancelled, done or superseded — record the user's own decision. "
        "attach: continue an earlier task of this project the user asked to pick up. "
        "show: read the current task state. "
        "lesson: propose a lesson this work taught (kept unconfirmed until the user confirms it or it works "
        "again), or record the user's own confirm or revoke of a lesson by experience_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["create", "add", "update_item", "set_status", "attach", "show", "lesson"]},
            "task_id": {"type": "string", "description": "Defaults to this conversation's focus task."},
            "title": {"type": "string"},
            "goal": {"type": "string"},
            "kind": {"type": "string", "enum": list(TASK_KINDS)},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(ITEM_STATES)},
                        "text": {"type": "string"},
                        "state": {"type": "string"},
                        "action_class": {"type": "string", "enum": list(ACTION_CLASSES)},
                        "supersedes": {"type": "string", "description": "item_id this one replaces"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["kind", "text"],
                },
            },
            "item_id": {"type": "string"},
            "state": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": list(TASK_STATUSES)},
            "reason": {"type": "string"},
            "lesson": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["propose", "confirm", "revoke"]},
                    "experience_id": {"type": "string"},
                    "situation": {"type": "string", "description": "When it applies"},
                    "behavior": {"type": "string", "description": "What to do, with the command in backticks"},
                    "exceptions": {"type": "string"},
                    "kind": {"type": "string", "enum": list(EXPERIENCE_KINDS)},
                },
            },
        },
        "required": ["action"],
    },
}


def _task_json(task) -> dict:
    return {"task_id": task.task_id, "title": task.title, "status": task.status, "revision": task.revision}


def _own_task(task_id: str, turn: runtime.TurnState) -> str:
    """A task this conversation may change: its project and conversation (an earlier one comes in
    through ``attach``), so an id seen elsewhere cannot reach another project's or chat's task."""
    task = runtime.store().task(task_id)
    if task is None or (task.project_key, task.lineage_root) != (turn.project_key, turn.lineage_root):
        raise WorkingMemoryError(f"task {task_id} is not part of this conversation; attach it first")
    return task_id


def _focus_id(args: dict, turn: runtime.TurnState) -> str:
    task_id = str(args.get("task_id") or "")
    if task_id:
        return _own_task(task_id, turn)
    focus = runtime.focus_task(turn)
    if focus is None:
        raise WorkingMemoryError("no task in this conversation yet; create one first")
    return focus.task_id


def _add_items(task_id: str, items: list, source) -> list[str]:
    wm, added = runtime.store(), []
    for raw in items or []:
        attrs = {"action_class": raw["action_class"]} if raw.get("action_class") else {}
        item = wm.add_item(task_id, kind=raw.get("kind", ""), text=raw.get("text", ""), state=raw.get("state"),
                           attrs=attrs, source=source, evidence_ids=raw.get("evidence_ids") or (),
                           supersedes=raw.get("supersedes"))
        added.append(item.item_id)
    return added


def _create(args: dict, turn: runtime.TurnState, source) -> dict:
    wm = runtime.store()
    task = wm.create_task(project_key=turn.project_key, lineage_root=turn.lineage_root,
                          origin_session=turn.session_id, title=args.get("title") or "untitled",
                          goal=args.get("goal") or args.get("title") or "", kind=args.get("kind") or "other",
                          source=source)
    wm.set_focus(task.task_id, source=source)
    items = _add_items(task.task_id, args.get("items"), source)
    return {"task": _task_json(wm.task(task.task_id)), "items": items}


def _add(args: dict, turn: runtime.TurnState, source) -> dict:
    task_id = _focus_id(args, turn)
    items = _add_items(task_id, args.get("items"), source)
    return {"task": _task_json(runtime.store().task(task_id)), "items": items}


def _update_item(args: dict, turn: runtime.TurnState, source) -> dict:
    wm, item_id = runtime.store(), str(args.get("item_id") or "")
    current = wm.item(item_id)
    if current is None:
        raise WorkingMemoryError(f"unknown item {item_id}")
    _own_task(current.task_id, turn)
    item = wm.set_item_state(item_id, str(args.get("state") or ""), source=source,
                             evidence_ids=args.get("evidence_ids") or ())
    return {"item": {"item_id": item.item_id, "kind": item.kind, "state": item.state,
                     "evidence_ids": list(item.evidence_ids)}}


def _set_status(args: dict, turn: runtime.TurnState, source) -> dict:
    wm, task_id = runtime.store(), _focus_id(args, turn)
    current = wm.task(task_id)
    task = wm.set_status(task_id, str(args.get("status") or ""), source=source,
                         expected_revision=current.revision if current else None, reason=str(args.get("reason") or ""))
    return {"task": _task_json(task)}


def _attach(args: dict, turn: runtime.TurnState, source) -> dict:
    task = runtime.store().attach(str(args.get("task_id") or ""), project_key=turn.project_key,
                                  lineage_root=turn.lineage_root, source=source)
    return {"task": _task_json(task)}


def _show(args: dict, turn: runtime.TurnState, source) -> dict:
    return {"state": snapshot.render_body(turn, runtime.settings()) or "no tasks in this conversation"}


def _propose(spec: dict, turn: runtime.TurnState, _source) -> dict:
    """A lesson the model draws from the work: a candidate in its own name, whoever's turn it is."""
    behavior = str(spec.get("behavior") or "").strip()
    wm, focus = runtime.store(), runtime.focus_task(turn)
    topic = topic_terms(behavior)
    for lesson in wm.experiences(project_key=turn.project_key, statuses=("candidate", "active", "stale")):
        if same_topic(topic, frozenset(lesson.lesson_key.split())):
            raise WorkingMemoryError(f"lesson {lesson.experience_id} already covers this; if it should change, "
                                     "ask the user")
    situation = str(spec.get("situation") or (f'{focus.kind} task "{clip(focus.title, 60)}"' if focus else "this project"))
    said_by_model = SourceRef(type="assistant", session_id=turn.session_id, turn_id=turn.turn_id, ref="task_state",
                              at=time.time(), authored_by="model", run_id=RUN_ID)
    from plugins.cognition_wm.learning import environment

    lesson = wm.add_experience(
        kind=spec.get("kind") if spec.get("kind") in EXPERIENCE_KINDS else "procedure", scope="project",
        project_key=turn.project_key, task_kind=focus.kind if focus else None, situation=situation,
        behavior=behavior, exceptions=str(spec.get("exceptions") or ""),
        tags=search_terms(f"{behavior} {situation}")[:24], lesson_key=" ".join(sorted(topic)), status="candidate",
        origin=said_by_model, learned_under=environment(turn.notes.get("model", ""), turn.notes.get("provider", "")))
    return {"experience_id": lesson.experience_id, "status": lesson.status}


def _settle(spec: dict, turn: runtime.TurnState, source, *, intent: str, status: str) -> dict:
    """The user's own confirm or revoke of a lesson, recorded by id: only on a turn where they said it."""
    if not turn.owner or turn.notes.get("lesson_intent") != intent:
        raise ProvenanceError(f"the user did not ask to {'confirm' if status == 'active' else 'revoke'} a lesson "
                              "in this turn; only their own words do")
    wm = runtime.store()
    lesson = wm.experience(str(spec.get("experience_id") or ""))
    if lesson is None or (lesson.scope == "project" and lesson.project_key != turn.project_key):
        raise WorkingMemoryError("no such lesson in this project")
    if status == "active":
        wm.add_signal(lesson.experience_id, source_class="user_confirmation", polarity=1, source=source,
                      evidence_key=f"msg:{turn.lineage_root}:{turn.turn_id}:confirm", lineage=turn.lineage_root)
    lesson = wm.set_experience_status(lesson.experience_id, status, source=source)
    return {"experience_id": lesson.experience_id, "status": lesson.status}


_LESSON_OPS: dict[str, Callable[[dict, runtime.TurnState, Any], dict]] = {
    "propose": _propose,
    "confirm": partial(_settle, intent="confirmation", status="active"),
    "revoke": partial(_settle, intent="revoke", status="revoked"),
}


def _lesson(args: dict, turn: runtime.TurnState, source) -> dict:
    if not runtime.settings().learning:
        raise WorkingMemoryError("learning is off in this profile")
    spec = args.get("lesson") if isinstance(args.get("lesson"), dict) else {}
    op = _LESSON_OPS.get(str(spec.get("op") or "propose"))
    if op is None:
        raise WorkingMemoryError(f"unknown lesson op; use one of {sorted(_LESSON_OPS)}")
    return {"lesson": op(spec, turn, source)}


_ACTIONS: dict[str, Callable[[dict, runtime.TurnState, Any], dict]] = {
    "create": _create, "add": _add, "update_item": _update_item, "set_status": _set_status,
    "attach": _attach, "show": _show, "lesson": _lesson,
}
_LIMITING = frozenset({"forbidden", "ask_first"})


def _records_limit(args: dict, result: dict) -> bool:
    """A forbid or ask-first rule was recorded: the form of a limit the scope gate enforces."""
    added = any(i.get("kind") == "authorization" and i.get("state") in _LIMITING for i in args.get("items") or [])
    item = result.get("item") or {}
    return added or (item.get("kind") == "authorization" and item.get("state") in _LIMITING)


def _sets_status(*statuses: str) -> Callable[[dict, dict], bool]:
    return lambda args, _result: args.get("action") == "set_status" and args.get("status") in statuses


# intent kind -> whether a successful call records it; a note about something else leaves it pending
_SETTLES: dict[str, Callable[[dict, dict], bool]] = {
    "cancel": _sets_status("cancelled"),
    "pause": _sets_status("paused", "cancelled"),
    "resume": _sets_status("active"),
    "restrict": _records_limit,
}


def handle(args: dict, session_id: Optional[str] = None, **_: Any) -> str:
    action = _ACTIONS.get(str((args or {}).get("action") or ""))
    if action is None:
        return json.dumps({"error": f"unknown action; use one of {sorted(_ACTIONS)}"})
    turn = runtime.current_turn(session_id) or runtime.TurnState(
        session_id=session_id or "", turn_id="", owner=False, project_key=runtime.project_key(),
        lineage_root=session_id or "", platform="")
    source = turn_source(owner=turn.owner, session_id=turn.session_id, turn_id=turn.turn_id)
    try:
        result = action(args or {}, turn, source)
    except WorkingMemoryError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    if turn.intent is not None and _SETTLES[turn.intent.kind](args or {}, result):
        runtime.clear_intent(session_id)  # the model recorded the stop or limit the user asked for
    return json.dumps({"ok": True, **result}, ensure_ascii=False)
