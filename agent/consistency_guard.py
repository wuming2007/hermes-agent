"""Consistency guard / verification second pass (PR3).

Post-generation guard that runs after the main reasoning loop produces a
candidate ``final_response``. Translates the per-turn
``CognitiveRoute.verification_plan`` emitted by PR1 into either:

- ``none``: no guard, candidate response passes through unchanged
- ``light``: rule-based local checks (no extra LLM call), surfaces issues
  in ``VerificationResult.notes`` but generally does not rewrite output
- ``full``: structured verifier call that may rewrite the final response

Hard contracts (do not change without bumping PR scope):

- guard NEVER mutates the system prompt — the prompt cache prefix is
  precious and post-generation by definition cannot influence it
- guard failures are non-fatal: any exception falls through to the
  candidate response so a buggy verifier never breaks a turn
- guard never persists its internal reasoning into the conversation
  ``messages`` list — only a revised final response (if any) is allowed
  to update the last assistant message

PR3 stays narrow: ``light`` is rule-based, ``full`` is one verifier call,
no chained reasoning, no multi-step repair pipelines.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple

from agent.cognitive_router import CognitiveRoute

logger = logging.getLogger(__name__)


VerificationPlan = Literal["none", "light", "full"]

_VALID_PLANS: frozenset[str] = frozenset({"none", "light", "full"})


@dataclass(frozen=True)
class VerificationResult:
    """Result of running the consistency guard on a candidate response.

    ``applied`` distinguishes "guard ran" from "guard skipped". ``changed``
    distinguishes "guard ran but kept candidate" from "guard rewrote it".
    ``notes`` is a tuple so the result can be hashed / safely shared across
    the turn-local metadata snapshot.
    """

    applied: bool
    plan: VerificationPlan
    original_response: str
    final_response: str
    changed: bool
    notes: Tuple[str, ...] = field(default_factory=tuple)


def resolve_verification_plan(
    cognition_route: Optional[CognitiveRoute],
) -> VerificationPlan:
    """Return the effective verification plan for the upcoming guard call.

    PR4 contract: this is the **single source of truth** for guard
    dispatch. The function reads ``cognition_route.verification_plan``
    only — ``cognition_route.consistency_check`` is deliberately not
    consulted here (or anywhere else in the dispatch path). See
    :class:`agent.cognitive_router.CognitiveRoute` for the full contract.

    Returns ``"none"`` when the route is missing or the plan string is
    unrecognized so callers can short-circuit without a special case.
    """
    if cognition_route is None:
        return "none"
    plan = cognition_route.verification_plan
    if plan in _VALID_PLANS:
        return plan  # type: ignore[return-value]
    return "none"


def should_run_consistency_guard(
    cognition_route: Optional[CognitiveRoute],
) -> bool:
    """Return True when ``verification_plan`` requires running the guard.

    PR4 contract: ``verification_plan`` is the only signal consulted.
    ``cognition_route.consistency_check`` is a non-execution hint
    preserved in metadata for telemetry — it does NOT influence
    dispatch. ``consistency_check=True`` while ``verification_plan="none"``
    yields ``False``; ``consistency_check=False`` while
    ``verification_plan="full"`` yields ``True``. This split keeps the
    PR3 production behavior verbatim while making the contract explicit
    so future maintainers do not silently re-couple the two fields.
    """
    return resolve_verification_plan(cognition_route) in ("light", "full")


# ---------------------------------------------------------------------------
# Light guard: rule-based local checks (no extra LLM call)
# ---------------------------------------------------------------------------

# Pairs of phrases that, when both appear in the same response, are almost
# certainly contradicting each other. Kept deliberately small and concrete —
# false positives are worse here than false negatives because the light
# guard surfaces issues without auto-repairing.
_CONTRADICTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("i've completed", "haven't started"),
    ("i have completed", "haven't started"),
    ("已完成", "尚未做"),
    ("已完成", "還沒做"),
    ("done", "not done yet"),
)


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


def _detect_contradictions(text_lower: str) -> list[str]:
    hits: list[str] = []
    for left, right in _CONTRADICTION_PAIRS:
        if left in text_lower and right in text_lower:
            hits.append(f"contradiction:{left}/{right}")
    return hits


def run_light_consistency_check(
    *,
    candidate_response: str,
    user_message: str,
) -> VerificationResult:
    """Rule-based light guard — surfaces issues without rewriting output.

    PR3 contract: this function never makes an external call and never
    rewrites the candidate response. It only attaches notes describing
    detected issues so callers (and downstream telemetry) can see what
    the guard noticed. Auto-repair is reserved for the full guard / a
    later PR.
    """
    notes: list[str] = []
    candidate = candidate_response or ""

    if _is_blank(candidate):
        notes.append("empty_response")
        return VerificationResult(
            applied=True,
            plan="light",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    text_lower = candidate.lower()
    notes.extend(_detect_contradictions(text_lower))

    return VerificationResult(
        applied=True,
        plan="light",
        original_response=candidate,
        final_response=candidate,
        changed=False,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Full guard: structured verifier call (mockable for tests)
# ---------------------------------------------------------------------------


VerifierCallable = Callable[[str], Any]
"""Verifier signature: takes a verification prompt string, returns a dict
with keys ``verdict`` (``"ok"`` or ``"revise"``), optional ``issues`` (list)
and optional ``revised_response`` (str). Tests pass a fake; production wires
in a thin wrapper around the auxiliary text client."""


def _build_verification_prompt(
    *,
    candidate_response: str,
    user_message: str,
) -> str:
    """Build a compact verifier prompt.

    Kept short on purpose: the verifier must run cheaply enough not to
    bloat ``deep`` mode latency. The prompt asks for a structured JSON
    response so callers can parse it deterministically.
    """
    return (
        "You are a strict consistency verifier. Compare the candidate "
        "assistant response against the user's request. If the candidate "
        "is acceptable, reply with JSON {\"verdict\":\"ok\"}. If it has a "
        "concrete factual / logical / completeness problem, reply with JSON "
        "{\"verdict\":\"revise\",\"issues\":[...],\"revised_response\":\"...\"}.\n\n"
        f"User request:\n{user_message}\n\n"
        f"Candidate response:\n{candidate_response}\n\n"
        "Reply with JSON only."
    )


def _coerce_verifier_payload(payload: Any) -> Optional[dict]:
    """Best-effort normalize verifier output into a dict, or return None."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        # Some auxiliary clients return raw JSON-ish strings; try to parse.
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def run_full_consistency_check(
    *,
    candidate_response: str,
    user_message: str,
    verifier: VerifierCallable,
) -> VerificationResult:
    """Full guard — runs one verifier call that may rewrite the response.

    ``verifier`` is injected so PR3 tests can drive the function without
    spinning up an aux LLM. Production callers wire in
    :func:`_default_verifier` or a similar thin wrapper. Failures of any
    kind (verifier raises, returns garbage, returns ``revise`` without a
    replacement) are non-fatal: the function falls back to the candidate
    response with a descriptive note.
    """
    candidate = candidate_response or ""
    notes: list[str] = []

    if _is_blank(candidate):
        notes.append("empty_response")
        return VerificationResult(
            applied=True,
            plan="full",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    prompt = _build_verification_prompt(
        candidate_response=candidate, user_message=user_message
    )

    try:
        raw_payload = verifier(prompt)
    except Exception as exc:
        logger.warning("consistency_guard verifier raised (non-fatal): %s", exc)
        notes.append(f"verifier_error:{type(exc).__name__}")
        return VerificationResult(
            applied=True,
            plan="full",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    payload = _coerce_verifier_payload(raw_payload)
    if payload is None:
        notes.append("verifier_parse:not_a_dict")
        return VerificationResult(
            applied=True,
            plan="full",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    verdict = str(payload.get("verdict", "")).strip().lower()
    issues = payload.get("issues") or []
    if isinstance(issues, list):
        notes.extend(str(i) for i in issues if i)

    if verdict == "ok":
        return VerificationResult(
            applied=True,
            plan="full",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    if verdict == "revise":
        revised = payload.get("revised_response")
        if isinstance(revised, str) and revised.strip():
            return VerificationResult(
                applied=True,
                plan="full",
                original_response=candidate,
                final_response=revised,
                changed=True,
                notes=tuple(notes),
            )
        notes.append("missing_revised_response")
        return VerificationResult(
            applied=True,
            plan="full",
            original_response=candidate,
            final_response=candidate,
            changed=False,
            notes=tuple(notes),
        )

    # Unknown verdict — treat as parse failure, keep candidate.
    notes.append(f"verifier_parse:unknown_verdict:{verdict or 'missing'}")
    return VerificationResult(
        applied=True,
        plan="full",
        original_response=candidate,
        final_response=candidate,
        changed=False,
        notes=tuple(notes),
    )
