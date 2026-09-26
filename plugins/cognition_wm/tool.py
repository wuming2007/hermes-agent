"""``task_state``: the model records the work's state; the plugin decides whose authority it carries.

The source of every write is this turn's: the owner's when the turn is the user's own words,
otherwise the model's. Tool arguments cannot claim otherwise, so a prompt-injected "the user
approved" in a web page or a notice cannot create, cancel or authorize anything.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from plugins.cognition_wm import runtime, snapshot
from plugins.cognition_wm.model import ACTION_CLASSES, ITEM_STATES, TASK_KINDS, TASK_STATUSES
from plugins.cognition_wm.provenance import turn_source
from plugins.cognition_wm.store import WorkingMemoryError

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
        "show: read the current task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "add", "update_item", "set_status", "attach", "show"]},
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
        },
        "required": ["action"],
    },
}


def _task_json(task) -> dict:
    return {"task_id": task.task_id, "title": task.title, "status": task.status, "revision": task.revision}


def _focus_id(args: dict, turn: runtime.TurnState) -> str:
    task_id = str(args.get("task_id") or "")
    if not task_id:
        focus = runtime.focus_task(turn)
        if focus is None:
            raise WorkingMemoryError("no task in this conversation yet; create one first")
        task_id = focus.task_id
    return task_id


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
    item = runtime.store().set_item_state(str(args.get("item_id") or ""), str(args.get("state") or ""),
                                          source=source, evidence_ids=args.get("evidence_ids") or ())
    return {"item": {"item_id": item.item_id, "state": item.state, "evidence_ids": list(item.evidence_ids)}}


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


_ACTIONS: dict[str, Callable[[dict, runtime.TurnState, Any], dict]] = {
    "create": _create, "add": _add, "update_item": _update_item, "set_status": _set_status,
    "attach": _attach, "show": _show,
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
    if action is not _show:
        runtime.clear_intent(session_id)  # the model recorded the state the user asked for
    return json.dumps({"ok": True, **result}, ensure_ascii=False)
