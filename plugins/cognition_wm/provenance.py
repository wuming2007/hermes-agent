"""Who is speaking in a turn: the owner's own words, or something the runtime put in a user row.

Owner turns may create and cancel tasks and grant authority, so the test is conservative: the
row must pass the core's real-user check and must not open with any machine marker, and cron
runs and delegated children never speak for the owner.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from plugins.cognition_wm.model import SourceRef

# One id per process: a grant made in an earlier run (a resumed chat) is historical.
RUN_ID = uuid.uuid4().hex[:12]

# Openers the runtime writes into user-role rows (notices, notes, markers, handoffs) beyond the
# ones the core check already knows. A real message rarely opens with a bracketed tag like these.
# A mid-turn /steer row is NOT one: core delivers the user's own words in that marker, with full
# authority (agent.prompt_builder.steer_user_row).
_MACHINE_OPENER = re.compile(
    r"\A\s*\[(?:System|SYSTEM|Runtime|IMPORTANT|ASYNC DELEGATION|Background process|CONTEXT COMPACTION"
    r"|Your active task list)")
_NON_OWNER_PLATFORMS = frozenset({"cron"})


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def is_owner_turn(message: Any, *, platform: str, delegated: bool) -> bool:
    """True when ``message`` is the owner's own words in a live turn."""
    from agent.conversation_compression import _is_real_user_message

    if delegated or platform in _NON_OWNER_PLATFORMS or not isinstance(message, dict):
        return False
    return _is_real_user_message(message) and not _MACHINE_OPENER.match(_text(message.get("content")))


def turn_source(*, owner: bool, session_id: str, turn_id: str, ref: str = "", at: float | None = None) -> SourceRef:
    """The source for a state recorded in this turn: the owner's when the turn is theirs, else the model's."""
    return SourceRef(
        type="user_message" if owner else "assistant", session_id=session_id, turn_id=turn_id, ref=ref,
        at=time.time() if at is None else at, authored_by="owner" if owner else "model", run_id=RUN_ID,
    )
