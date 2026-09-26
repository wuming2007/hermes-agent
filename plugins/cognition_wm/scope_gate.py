"""``pre_tool_call`` scope gate: execution tools stay inside what the user asked for.

- The focus task is paused or cancelled, or the user just asked to stop: execution tools are
  blocked (``cancelled_under_yolo: block``, so /yolo does not reopen stopped work).
- The user just limited the work and the model has not recorded it yet: execution tools go to
  the existing human-approval prompt.
- A recorded authorization decides per action class: forbidden blocks; ask_first, or a grant
  from an earlier run, goes to the approval prompt; a grant made in this run says nothing.
- Shell commands and ``execute_code`` scripts are read through ``agent.command_effects``, however
  they are wrapped. A command whose effect cannot be told (``python3 refactor.py``) is stopped like
  any execution while the task is stopped or the user's new limit is unrecorded, and goes to the
  approval prompt while the user forbids any action for the task.
- Read-only tools, ``task_state`` and ``clarify`` are never gated, and working-memory state is
  never written into long-term memory.

Errors never block: the gate says nothing and the normal approval rules apply.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable, Optional

from agent.command_effects import (
    AGENT_LAUNCH,
    DEPLOY,
    FILE_WRITE,
    MIGRATION,
    REPO_CHANGE,
    UNKNOWN,
    code_effects,
    command_effects,
)
from plugins.cognition_wm import provenance, runtime
from plugins.cognition_wm.model import clip

logger = logging.getLogger(__name__)

UNKNOWN_EFFECT = "unknown_effect"  # gate-internal: never recorded as a rule's action class
_CLASS_ORDER = ("coding_agent_launch", "data_migration", "deploy_restart", "repo_write", "code_change",
                "external_message", "other", UNKNOWN_EFFECT)
_TOOL_CLASSES = {
    "write_file": "code_change", "patch": "code_change", "delegate_task": "coding_agent_launch",
    "send_message": "external_message", "kanban_create": "other",
}
_EFFECT_CLASSES = {
    AGENT_LAUNCH: "coding_agent_launch", MIGRATION: "data_migration", DEPLOY: "deploy_restart",
    REPO_CHANGE: "repo_write", FILE_WRITE: "code_change", UNKNOWN: UNKNOWN_EFFECT,
}
_PROCESS_ACTIONS = frozenset({"write", "submit"})
_WM_TRACE = re.compile(r"<working-memory|\bt_[0-9a-f]{12}\b|internal task state kept by Hermes")
_STOPPED = {"paused": "paused", "cancelled": "cancelled"}


def _classes(effects: Iterable[str]) -> set[str]:
    return {_EFFECT_CLASSES[effect] for effect in effects}


def _fixed(cls: str) -> Callable[[dict], set[str]]:
    return lambda _args: {cls}


# tool name -> the action classes a call with these arguments would exercise
_CLASSIFIERS: dict[str, Callable[[dict], set[str]]] = {
    "terminal": lambda args: _classes(command_effects(str(args.get("command") or ""))),
    "execute_code": lambda args: _classes(code_effects(str(args.get("code") or ""))),
    "process_manage": lambda args: {"other"} if args.get("action") in _PROCESS_ACTIONS else set(),
    "cronjob_manage": lambda args: {"other"} if args.get("action") not in ("list", "show", None) else set(),
    **{name: _fixed(cls) for name, cls in _TOOL_CLASSES.items()},
}


def action_classes(tool_name: str, args: Any) -> tuple[str, ...]:
    """The action classes a tool call would exercise (empty: not an execution)."""
    classify = _CLASSIFIERS.get(tool_name)
    found = classify(args if isinstance(args, dict) else {}) if classify else set()
    return tuple(cls for cls in _CLASS_ORDER if cls in found)


def _stopped_task(turn: runtime.TurnState, settings: runtime.Settings, tool_name: str) -> Optional[dict]:
    focus = runtime.focus_task(turn)
    if focus is not None and focus.status in _STOPPED:
        message = (f"BLOCKED: the user {_STOPPED[focus.status]} task {focus.task_id} \"{clip(focus.title, 60)}\" "
                   f"(status {focus.status.upper()}), so '{tool_name}' did not run. Do not continue it unless the "
                   "user asks again in their own words.")
        return _stop(settings, message)
    intent = turn.intent
    if intent is not None and intent.kind in ("cancel", "pause"):
        message = (f"BLOCKED: the user just asked to {'stop' if intent.kind == 'cancel' else 'pause'} "
                   f"(\"{intent.text}\"), so '{tool_name}' did not run. Record it with task_state and answer the user.")
        return _stop(settings, message)
    return None


def _stop(settings: runtime.Settings, message: str) -> dict:
    if settings.cancelled_under_yolo == "approve":
        return {"action": "approve", "message": message, "rule_key": "cognition:stopped"}
    return {"action": "block", "message": message}


def _authorized(turn: runtime.TurnState, classes: tuple[str, ...], tool_name: str) -> Optional[dict]:
    focus = runtime.focus_task(turn)
    if focus is None:
        return None
    grants = runtime.store().effective_authorizations(focus.task_id, run_id=provenance.RUN_ID)
    rules = {i.attrs.get("action_class"): i.text for i in runtime.store().items(focus.task_id, kinds=("authorization",))}
    # A class the user forbade blocks the call, whatever other class it also exercises.
    blocked = next((cls for cls in classes if grants.get(cls) == "forbidden"), None)
    if blocked:
        return {"action": "block", "message": f"BLOCKED: the user forbade {blocked} for this task "
                                              f"(\"{clip(rules.get(blocked, ''), 120)}\"), so '{tool_name}' did not run."}
    asked = next((cls for cls in classes if grants.get(cls) in ("ask_first", "historical")), None)
    if asked:
        why = ("the user asked to be asked first" if grants[asked] == "ask_first"
               else "the permission was given before the task was stopped, or in an earlier session, and must be "
                    "confirmed again")
        return {"action": "approve", "rule_key": f"cognition:{focus.task_id}:{asked}",
                "message": f"{asked} for task {focus.task_id}: {why} (\"{clip(rules.get(asked, ''), 120)}\")."}
    forbidden = next((cls for cls in _CLASS_ORDER if grants.get(cls) == "forbidden"), None)
    if UNKNOWN_EFFECT in classes and forbidden:
        return {"action": "approve", "rule_key": f"cognition:{focus.task_id}:{UNKNOWN_EFFECT}",
                "message": f"what '{tool_name}' would change cannot be told from the call, and the user forbade "
                           f"{forbidden} for task {focus.task_id} (\"{clip(rules.get(forbidden, ''), 120)}\")."}
    return None


def _limited(turn: runtime.TurnState, tool_name: str) -> Optional[dict]:
    if turn.intent is None or turn.intent.kind != "restrict":
        return None
    return {"action": "approve", "rule_key": "cognition:limit",
            "message": f"the user limited this work (\"{turn.intent.text}\"); running '{tool_name}' needs their "
                       "confirmation. Record the limit with task_state as an authorization item (forbidden or "
                       "ask_first, with its action_class)."}


def decide(tool_name: str, args: Any, turn: runtime.TurnState, settings: runtime.Settings) -> Optional[dict]:
    """The gate's directive for one tool call, or None to leave it to the normal rules. A block
    from any rule wins over a request for approval."""
    classes = action_classes(tool_name, args)
    if not classes or not settings.scope_gate:
        return None
    found = [d for d in (_stopped_task(turn, settings, tool_name), _authorized(turn, classes, tool_name),
                         _limited(turn, tool_name)) if d is not None]
    return next((d for d in found if d["action"] == "block"), found[0] if found else None)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> Optional[dict]:
    try:
        if tool_name == "memory" and _WM_TRACE.search(str((args or {}).get("content") or "")):
            return {"action": "block", "message": "BLOCKED: working-memory state is kept by Hermes and must not be "
                                                  "stored in long-term memory."}
        turn = runtime.current_turn(session_id)
        return decide(tool_name, args, turn, runtime.settings()) if turn is not None else None
    except Exception as exc:
        logger.warning("cognition_wm scope gate failed (no opinion): %s", exc)
        return None
