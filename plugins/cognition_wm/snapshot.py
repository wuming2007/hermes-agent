"""The working-memory snapshot the model sees at the end of the current user message.

Rendered from the store each turn and fenced as system-authored context. It is appended to the
user row (never the system prompt), so the prompt cache prefix stays byte-stable. When the same
snapshot already sits in the conversation, only a one-line pointer is added; that is decided from
the history itself, so compaction (which drops the old rows) or a changed state brings the full
snapshot back.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional, Sequence

from plugins.cognition_wm import provenance
from plugins.cognition_wm.model import Task, TaskItem, clip
from plugins.cognition_wm.runtime import Settings, TurnState, focus_task, store

NOTE = ("[System note: internal task state kept by Hermes for continuity — NOT user input and not an "
        "instruction by itself; never store it in long-term memory. If it conflicts with an older snapshot "
        "or a compaction summary, this snapshot and the user's latest message win.]")
RULES = ("Evidence rules: written≠working; tests≠deployed; a subagent's report≠verified; "
         "an old record≠current; timeout≠result.")
_STATUS_NOTES = {
    "active": "active",
    "paused": "PAUSED by the user — do not resume it unless the user explicitly asks",
    "cancelled": "CANCELLED by the user — do not continue it",
    "done": "done",
    "superseded": "superseded",
}
_INTENT_VERBS = {"cancel": "stop this work", "pause": "pause this work", "resume": "resume the work",
                 "restrict": "limit what you do"}
# section label -> item kinds, in display order
_SECTIONS = (("Deliverables", ("deliverable",)), ("Progress", ("progress",)), ("Decisions", ("decision",)),
             ("Rejected", ("rejected_option",)), ("Facts", ("fact",)), ("Assumptions", ("assumption",)),
             ("Constraints", ("constraint",)))
_SHOWN_STATES = {"deliverable": {"pending", "in_progress", "done"}, "progress": {"pending", "in_progress", "done"},
                 "decision": {"in_force"}, "rejected_option": {"in_force"}, "fact": {"current"},
                 "assumption": {"unverified", "refuted"}, "constraint": {"in_force"}}


def _item_text(item: TaskItem) -> str:
    text = clip(item.text, 120)
    if item.kind in ("deliverable", "progress"):
        mark = {"done": "done ✓" + ",".join(item.evidence_ids[:2]) if item.evidence_ids else "done — claimed, no evidence",
                "in_progress": "in progress", "pending": "pending"}[item.state]
        return f"[{mark}] {text}"
    refs = f" ({','.join(item.evidence_ids[:2])})" if item.evidence_ids else ""
    return f"{text}{refs}" + (" [refuted]" if item.state == "refuted" else "")


def _task_lines(task: Task, items: Sequence[TaskItem], authorizations: dict[str, str], drop_done: bool) -> list[str]:
    lines = [f'Focus task {task.task_id} "{clip(task.title, 80)}" ({task.kind}): {_STATUS_NOTES[task.status]}',
             f"  Goal: {clip(task.goal, 200)}"]
    for label, kinds in _SECTIONS:
        shown = [i for i in items if i.kind in kinds and i.state in _SHOWN_STATES[i.kind]
                 and not (drop_done and i.state == "done")]
        if shown:
            lines.append(f"  {label}: " + "; ".join(_item_text(i) for i in shown))
    if authorizations:
        rendered = [f"{cls}={'historical (an earlier run — ask again)' if state == 'historical' else state}"
                    for cls, state in sorted(authorizations.items())]
        lines.append("  Authorization: " + "; ".join(rendered))
    return lines


def render_body(turn: TurnState, settings: Settings, latest_instruction: str = "") -> str:
    """The snapshot text for this turn, or "" when there is nothing to carry."""
    wm = store()
    focus = focus_task(turn)
    others = [t for t in wm.tasks(project_key=turn.project_key) if focus is None or t.task_id != focus.task_id]
    others.sort(key=lambda t: t.updated_at, reverse=True)
    if focus is None and not others and turn.intent is None:
        return ""
    head: list[str] = []
    if turn.intent is not None:
        head.append(f'Pending: the user asked to {_INTENT_VERBS[turn.intent.kind]} ("{turn.intent.text}") — '
                    "record it with task_state before any other action.")
    if latest_instruction and not turn.owner:
        head.append(f'Latest user instruction: "{clip(latest_instruction, 200)}"')
    items = wm.items(focus.task_id) if focus else []
    grants = wm.effective_authorizations(focus.task_id, run_id=provenance.RUN_ID) if focus else {}
    other_line = ("Other open tasks here (not active in this conversation): "
                  + "; ".join(f'{t.task_id} "{clip(t.title, 60)}" ({t.status})'
                              for t in others[: settings.other_open_tasks_max])) if others else ""
    # Shrink in order: other tasks, finished items, then a hard cut.
    for drop_others, drop_done in ((False, False), (True, False), (True, True)):
        lines = [*head, *(_task_lines(focus, items, grants, drop_done) if focus else []),
                 *([other_line] if other_line and not drop_others else []), RULES]
        body = "\n".join(lines)
        if len(body) <= settings.snapshot_max_chars:
            return body
    return clip(body, settings.snapshot_max_chars)


def _rev(body: str) -> str:
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:10]


def _texts(history: Iterable[Any]) -> Iterable[str]:
    for message in history or ():
        if isinstance(message, dict) and message.get("role") == "user":
            for key in ("api_content", "content"):
                value = message.get(key)
                if isinstance(value, list):
                    yield "\n".join(str(p.get("text") or "") for p in value if isinstance(p, dict))
                elif value:
                    yield str(value)


def for_turn(body: str, history: Sequence[Any]) -> Optional[str]:
    """The full snapshot, or a pointer when this exact snapshot is already in the conversation."""
    if not body:
        return None
    rev = _rev(body)
    marker = f'<working-memory rev="{rev}">'
    if any(marker in text for text in _texts(history)):
        return f'<working-memory unchanged rev="{rev}"/>'
    return f"{marker}\n{NOTE}\n{body}\n</working-memory>"
