"""``pre_tool_call`` scope gate: execution tools stay inside what the user asked for.

- The focus task is paused or cancelled, or the user just asked to stop: a call that acts (a write,
  repo-changing git, a coding agent, a deploy, a migration, a message, a click or keystroke in a
  browser or on the desktop, a Home Assistant service, board work, paid generation) is blocked
  (``cancelled_under_yolo: block``, so /yolo does not reopen stopped work). A call whose effect
  cannot be told (``python3 refactor.py``, page script, an unknown plugin tool, an MCP tool not
  declared read-only) goes to the approval prompt, and is blocked while approvals are bypassed.
  Reading stays open: pages, screenshots, status, searches, listing or stopping jobs and subagents.
- The user just limited the work and the model has not recorded it yet: execution tools go to
  the existing human-approval prompt.
- A recorded authorization decides per action class: forbidden blocks whatever else the call
  does; ask_first, or a grant from an earlier run or from before the task was stopped, goes to the
  approval prompt; a grant made in this run says nothing. While any class is forbidden, a call
  whose effect cannot be told is asked about.
- Shell commands and ``execute_code`` scripts are read through ``agent.command_effects``, however
  they are wrapped.
- Read-only tools, ``task_state``, ``clarify`` and Hermes's own UI tools are never gated.
- A memory or skill write that carries working-memory markers, or repeats a kept lesson word for
  word, is refused: a copy there would outlive the lesson's revocation. This is a check on what is
  written, not an isolation of where it came from: the background memory review replays earlier
  turns as they were sent (lessons included, for prompt-cache parity), and a reworded copy passes.

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
from agent.tool_result_classification import NO_EFFECT_TOOL_NAMES
from plugins.cognition_wm import provenance, runtime
from plugins.cognition_wm.lessons import normalized
from plugins.cognition_wm.model import clip

logger = logging.getLogger(__name__)

UNKNOWN_EFFECT = "unknown_effect"  # gate-internal: never recorded as a rule's action class
_CLASS_ORDER = ("coding_agent_launch", "data_migration", "deploy_restart", "repo_write", "code_change",
                "external_message", "other", UNKNOWN_EFFECT)
_TOOL_CLASSES = {
    "write_file": "code_change", "patch": "code_change",
    **{name: "external_message" for name in (
        "send_message", "yb_send_dm", "yb_send_sticker", "feishu_drive_add_comment", "feishu_drive_reply_comment")},
    # acting on a page, an account, the house, the board, or paid generation
    **{name: "other" for name in (
        "browser_click", "browser_type", "browser_press", "browser_exec", "browser_cdp", "browser_dialog",
        "browser_vault_enter_code", "browser_vault_fill", "browser_vault_save_login", "browser_vault_unlock",
        "ha_call_service", "kanban_create", "kanban_comment", "kanban_complete", "kanban_block", "kanban_unblock",
        "kanban_link", "kanban_attach", "kanban_attach_url", "kanban_request_review", "kanban_request_changes",
        "kanban_heartbeat", "image_generate", "video_generate", "xai_video_edit", "xai_video_extend",
        "text_to_speech")},
}
# Tools that only read, or only arrange Hermes's own UI and bookkeeping: never gated.
_UNGATED = NO_EFFECT_TOOL_NAMES | frozenset({
    "clarify", "todo_list", "todo", "task_state", "memory", "skill_manage", "skill_view", "skills_list",
    "tool_search", "tool_describe", "session_search", "mixture_of_agents", "read_terminal", "read_window_below",
    "video_analyze", "x_search", "ha_get_state", "ha_list_entities", "ha_list_services", "kanban_show",
    "kanban_list", "kanban_attachments", "feishu_doc_read", "feishu_drive_list_comments",
    "feishu_drive_list_comment_replies", "yb_query_group_info", "yb_query_group_members", "yb_search_sticker",
    "browser_navigate", "browser_back", "browser_scroll", "browser_snapshot", "browser_vision",
    "browser_get_images", "browser_vault_list", "desktop_preview", "annotate_preview", "apply_layout",
    "focus_pane", "gui_tour", "show_tip", "react_to_message", "close_terminal",
})
# tool -> (the class of its acting actions, the actions that act); its other actions only read
_ACTING_ACTIONS: dict[str, tuple[str, frozenset]] = {
    "computer_use": ("other", frozenset({"click", "double_click", "right_click", "middle_click", "drag", "type",
                                         "key", "set_value"})),
    "drive_preview": ("other", frozenset({"click", "type", "press"})),
    "discord": ("external_message", frozenset({"create_thread"})),
    "discord_admin": ("other", frozenset({"pin_message", "unpin_message", "delete_message", "add_role",
                                          "remove_role"})),
    "manage_connections": ("other", frozenset({"connect", "reconnect", "install", "enable", "authorize"})),
    "manage_catalog": ("other", frozenset({"install"})),
    "desktop_project": ("other", frozenset({"create"})),
    "process_manage": ("other", frozenset({"write", "submit"})),
}
_EFFECT_CLASSES = {
    AGENT_LAUNCH: "coding_agent_launch", MIGRATION: "data_migration", DEPLOY: "deploy_restart",
    REPO_CHANGE: "repo_write", FILE_WRITE: "code_change", UNKNOWN: UNKNOWN_EFFECT,
}
# Long-term stores the model writes (memory, and skills: procedure kept for good) -> the fields that carry
# what is written. Working-memory state and lessons are kept by Hermes, where they can be revoked.
_LONG_TERM_TEXT = {"memory": frozenset({"content"}), "skill_manage": frozenset({"content", "file_content", "new_string"})}
_WM_TRACE = re.compile(r"<working-memory|<lessons\b|\b(?:t|xp)_[0-9a-f]{12}\b|internal task state kept by Hermes"
                       r"|lessons from earlier work")
_STOPPED = {"paused": "paused", "cancelled": "cancelled"}


def _classes(effects: Iterable[str]) -> set[str]:
    return {_EFFECT_CLASSES[effect] for effect in effects}


def _fixed(cls: str) -> Callable[[dict], set[str]]:
    return lambda _args: {cls}


def _acting(cls: str, actions: frozenset) -> Callable[[dict], set[str]]:
    return lambda args: {cls} if args.get("action") in actions else set()


# tool name -> the action classes a call with these arguments would exercise
_CLASSIFIERS: dict[str, Callable[[dict], set[str]]] = {
    "terminal": lambda args: _classes(command_effects(str(args.get("command") or ""))),
    "execute_code": lambda args: _classes(code_effects(str(args.get("code") or ""))),
    "cronjob_manage": lambda args: {"other"} if args.get("action") not in ("list", "show", None) else set(),
    "browser_console": lambda args: {UNKNOWN_EFFECT} if args.get("expression") else set(),  # runs page script
    # listing or stopping subagents stays open; spawning (the default) or steering one is agent work
    "delegate_task": lambda args: set() if args.get("action") in ("list", "stop") else {"coding_agent_launch"},
    **{name: _acting(cls, actions) for name, (cls, actions) in _ACTING_ACTIONS.items()},
    **{name: _fixed(cls) for name, cls in _TOOL_CLASSES.items()},
}


def _unlisted(tool_name: str) -> set[str]:
    """A tool the gate does not know (a plugin's, a connector's, an MCP server's): its effect
    cannot be told, unless an MCP server declared it read-only."""
    if tool_name in _UNGATED:
        return set()
    if tool_name.startswith("mcp__"):
        from tools.mcp_tool_handlers import registered_tool_is_read_only

        if registered_tool_is_read_only(tool_name):
            return set()
    return {UNKNOWN_EFFECT}


def action_classes(tool_name: str, args: Any) -> tuple[str, ...]:
    """The action classes a tool call would exercise (empty: not an execution)."""
    classify = _CLASSIFIERS.get(tool_name)
    found = classify(args if isinstance(args, dict) else {}) if classify else _unlisted(tool_name)
    return tuple(cls for cls in _CLASS_ORDER if cls in found)


def _stopped_task(turn: runtime.TurnState, settings: runtime.Settings, tool_name: str,
                  classes: tuple[str, ...]) -> Optional[dict]:
    focus = runtime.focus_task(turn)
    intent = turn.intent
    if focus is not None and focus.status in _STOPPED:
        why = (f"the user {_STOPPED[focus.status]} task {focus.task_id} \"{clip(focus.title, 60)}\" "
               f"(status {focus.status.upper()})")
        rest = "Do not continue it unless the user asks again in their own words."
    elif intent is not None and intent.kind in ("cancel", "pause"):
        why = f"the user just asked to {'stop' if intent.kind == 'cancel' else 'pause'} (\"{intent.text}\")"
        rest = "Record it with task_state and answer the user."
    else:
        return None
    if set(classes) - {UNKNOWN_EFFECT}:
        return _stop(settings, f"BLOCKED: {why}, so '{tool_name}' did not run. {rest}")
    return _ask_while_stopped(settings, why, tool_name, rest)


def _stop(settings: runtime.Settings, message: str) -> dict:
    if settings.cancelled_under_yolo == "approve":
        return {"action": "approve", "message": message, "rule_key": "cognition:stopped"}
    return {"action": "block", "message": message}


def _ask_while_stopped(settings: runtime.Settings, why: str, tool_name: str, rest: str) -> dict:
    """What a call whose effect cannot be told gets while the work is stopped: the user's
    confirmation, or a block when approvals are bypassed (/yolo, approvals.mode: off), since a
    bypass would wave it through the stop."""
    from tools.approval import is_approval_bypass_active

    if settings.cancelled_under_yolo != "approve" and is_approval_bypass_active():
        return {"action": "block", "message": f"BLOCKED: {why}, and what '{tool_name}' would do cannot be told, so "
                                              f"it did not run. {rest}"}
    return {"action": "approve", "rule_key": "cognition:stopped",
            "message": f"{why}; what '{tool_name}' would do cannot be told from the call, so it needs the user's "
                       "confirmation."}


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
    found = [d for d in (_stopped_task(turn, settings, tool_name, classes), _authorized(turn, classes, tool_name),
                         _limited(turn, tool_name)) if d is not None]
    return next((d for d in found if d["action"] == "block"), found[0] if found else None)


def _written(value: Any, fields: frozenset) -> list[str]:
    """The text a long-term write would keep, nested operations included."""
    if isinstance(value, dict):
        return [text for key, item in value.items()
                for text in ([item] if key in fields and isinstance(item, str) else _written(item, fields))]
    if isinstance(value, list):
        return [text for item in value for text in _written(item, fields)]
    return []


def _copies_a_lesson(text: str, session_id: str) -> bool:
    """Whether a long-term write repeats, word for word, a lesson Hermes keeps for this project — in
    force, on trial or ended (revoked, replaced, lapsed: still in the store as history). A copy would
    outlive the lesson, or bring an ended one back. A reworded copy is not caught."""
    turn = runtime.current_turn(session_id)
    kept = runtime.store().experiences(project_key=turn.project_key if turn else runtime.project_key())
    flat = normalized(text)
    return any(len(core) >= 12 and core in flat for core in (normalized(lesson.behavior) for lesson in kept))


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> Optional[dict]:
    try:
        written = "\n".join(_written(args or {}, _LONG_TERM_TEXT.get(tool_name, frozenset())))
        if written and (_WM_TRACE.search(written) or _copies_a_lesson(written, session_id)):
            return {"action": "block", "message": "BLOCKED: working-memory state and lessons are kept by Hermes "
                                                  "(hermes cognition lessons) and must not be copied into long-term "
                                                  "memory or skills."}
        turn = runtime.current_turn(session_id)
        if turn is None:
            return None
        directive = decide(tool_name, args, turn, runtime.settings())
        if directive is not None:  # the user's cost of the turn, for the track record
            counter = "blocks" if directive["action"] == "block" else "asks"
            turn.notes[counter] = int(turn.notes.get(counter, 0)) + 1
        return directive
    except Exception as exc:
        logger.warning("cognition_wm scope gate failed (no opinion): %s", exc)
        return None
