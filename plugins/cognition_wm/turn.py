"""``pre_llm_call``: decide whose turn this is, notice control intents, and hand the model the
working-memory snapshot as context for the current user message.

Fail-open: any error leaves the turn without a snapshot and logs a warning.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from plugins.cognition_wm import learning, runtime, snapshot
from plugins.cognition_wm.intents import detect_intent
from plugins.cognition_wm.provenance import is_owner_turn

logger = logging.getLogger(__name__)
_LATEST_KEY = "latest_instruction:{}"


def _current_user_row(history: Any, user_message: Any) -> dict:
    """The row that opened this turn (the hook sees the history ending with it)."""
    for message in reversed(history or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return {"role": "user", "content": user_message}


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _latest_instruction(lineage_root: str, owner_text: Optional[str]) -> str:
    """Remember the owner's latest words per conversation; return them (for non-owner turns)."""
    wm, key = runtime.store(), _LATEST_KEY.format(lineage_root)
    if owner_text is not None:
        wm.set_meta(key, json.dumps({"text": owner_text[:400]}, ensure_ascii=False))
        return owner_text
    stored = wm.get_meta(key)
    return json.loads(stored)["text"] if stored else ""


def on_pre_llm_call(*, session_id: str = "", turn_id: str = "", user_message: Any = None,
                    conversation_history: Any = None, platform: str = "", parent_session_id: str = "",
                    sender_id: str = "", model: str = "", provider: str = "", **_: Any) -> Optional[dict]:
    try:
        row = _current_user_row(conversation_history, user_message)
        owner = is_owner_turn(row, platform=platform or "", delegated=bool(parent_session_id))
        text = _text(row.get("content"))
        state = runtime.TurnState(
            session_id=session_id or "", turn_id=turn_id or "", owner=owner,
            project_key=runtime.project_key(sender_id=sender_id or "", platform=platform or ""),
            lineage_root=runtime.lineage_root(session_id or ""), platform=platform or "",
            intent=detect_intent(text) if owner else None,
        )
        state.notes.update(model=model or "", provider=provider or "")
        runtime.remember_turn(state)
        runtime.upkeep()
        latest = _latest_instruction(state.lineage_root, text if owner else None)
        body = snapshot.render_body(state, runtime.settings(), latest_instruction=latest)
        context = snapshot.for_turn(body, conversation_history or [])
    except Exception as exc:  # the working memory must never break a turn
        logger.warning("cognition_wm pre_llm_call failed (no snapshot this turn): %s", exc)
        return None
    extra = learning.before_turn(state, text, conversation_history or [], model=model or "", provider=provider or "")
    result: dict = {}
    contexts = [piece for piece in (context, extra.context) if piece]
    if contexts:
        result["context"] = "\n\n".join(contexts)
    # The verifier sees no history, so it gets the whole state even when the model gets a pointer.
    verifier = "\n\n".join(piece for piece in (body if context else "", extra.verifier_body) if piece)
    if verifier:
        result["verifier_context"] = verifier
    if extra.guidance:
        result["turn_guidance"] = extra.guidance
    return result or None
