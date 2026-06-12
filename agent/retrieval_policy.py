"""Retrieval policy mapping (PR2).

Translates the per-turn ``CognitiveRoute.retrieval_plan`` string emitted by
PR1 into an executable policy describing which memory layers the upcoming
turn should fetch. Pure: no I/O, no state, safe to import anywhere.

Returning ``None`` for unknown / missing plans is intentional — callers must
treat ``None`` as "no policy, fall through to the legacy ``prefetch_all``
path" so cognition-disabled behavior stays bit-for-bit identical to pre-PR2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from agent.cognitive_router import CognitiveRoute


RetrievalLayer = Literal["principles", "semantic", "episodic"]


@dataclass(frozen=True)
class RetrievalPolicy:
    """Executable retrieval policy derived from a CognitiveRoute.

    Frozen so callers can't mutate the plan/layers after the fact. ``layers``
    is always a tuple (immutable, ordering-preserving) — never a list — so
    downstream code can hash / compare policies cheaply.
    """

    plan: str
    layers: Tuple[RetrievalLayer, ...]


_PLAN_TO_LAYERS: dict[str, Tuple[RetrievalLayer, ...]] = {
    "principles_only": ("principles",),
    "principles_plus_semantic": ("principles", "semantic"),
    "principles_plus_semantic_plus_episodic": ("principles", "semantic", "episodic"),
}


def resolve_retrieval_policy(
    cognition_route: Optional[CognitiveRoute],
) -> Optional[RetrievalPolicy]:
    """Map a CognitiveRoute to an executable RetrievalPolicy.

    Returns ``None`` when:

    - ``cognition_route`` is ``None`` (cognition disabled / not consulted), or
    - the plan string is ``"none"`` (router explicitly said "no retrieval"; we
      still return ``None`` so the run loop falls through to legacy
      ``prefetch_all`` rather than executing an empty-layer fetch), or
    - the plan string is unknown to PR2 (forward compatibility — newer router
      strings should not break older deployments).

    A ``None`` return is the safe-legacy signal; the caller is responsible
    for choosing the legacy code path in that case.
    """
    if cognition_route is None:
        return None
    plan = cognition_route.retrieval_plan
    layers = _PLAN_TO_LAYERS.get(plan)
    if layers is None:
        return None
    return RetrievalPolicy(plan=plan, layers=layers)
