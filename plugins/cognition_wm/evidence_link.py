"""``post_tool_call``: every tool result becomes a redacted evidence row, linked to the focus task.

Fail-open: a ledger error never affects the tool result or the turn.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from plugins.cognition_wm import runtime

logger = logging.getLogger(__name__)
_SKIPPED = frozenset({"task_state", "tool_search", "tool_describe"})
_EXCERPT = 400


def _redacted(text: str) -> str:
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(" ".join(str(text or "").split())[: _EXCERPT * 2])[:_EXCERPT]


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None, session_id: str = "",
                      tool_call_id: str = "", turn_id: str = "", status: str = "", duration_ms: Optional[int] = None,
                      **_: Any) -> None:
    if tool_name in _SKIPPED:
        return
    try:
        from agent.cognition_evidence import classify_tool_result
        from hermes_constants import hermes_home_key

        item = classify_tool_result(tool_name, args, result, tool_call_id)
        # A call without an id of its own (made inside another tool) still gets its own row.
        evidence_id = item.evidence_id if tool_call_id else f"e_{uuid.uuid4().hex[:16]}"
        turn = runtime.current_turn(session_id)
        focus = runtime.focus_task(turn) if turn else None
        runtime.store().record_evidence(
            evidence_id=evidence_id, task_id=focus.task_id if focus else None, session_id=session_id or "",
            turn_id=turn_id or "", tool_call_id=tool_call_id or None, tool_name=item.tool_name, kind=item.kind,
            status="blocked" if status == "blocked" else item.status, provenance=item.provenance,
            target=_redacted(item.target), excerpt=_redacted(item.text), duration_ms=duration_ms,
            scope={"cwd": os.environ.get("TERMINAL_CWD") or os.getcwd(), "profile": hermes_home_key()},
        )
    except Exception as exc:
        logger.warning("cognition_wm evidence record failed: %s", exc)
