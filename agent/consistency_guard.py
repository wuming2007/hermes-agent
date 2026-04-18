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

    Note: ``cognition_route.consistency_check`` is intentionally ignored
    here. PR3 treats ``verification_plan`` as the single source of truth
    for whether the guard runs; ``consistency_check`` is a separate
    orthogonal hint individual guard implementations may consult.
    """
    return resolve_verification_plan(cognition_route) in ("light", "full")
