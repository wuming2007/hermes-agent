"""Deterministic memory plasticity primitives for PR17.

This module turns bounded success/correction/staleness signals into additive
promotion/decay/supersede decisions. It is intentionally pure: no provider IO,
no model calls, no destructive writes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from agent.memory_ranker import (
    MemoryCandidate,
    MemoryObjectMetadata,
    candidate_with_normalized_metadata,
    clamp_signal,
    normalize_memory_metadata,
)

_ALLOWED_ACTIONS = {"promote", "maintain", "decay", "supersede"}


@dataclass(frozen=True)
class PlasticityConfig:
    """Bounded thresholds/deltas for deterministic plasticity."""

    promotion_threshold: float = 0.45
    decay_threshold: float = 0.35
    stale_after_days: int = 90
    success_weight: float = 0.16
    verification_weight: float = 0.12
    correction_weight: float = 0.24
    explicit_decay_weight: float = 0.35
    staleness_weight: float = 0.20
    max_reinforcement_delta: float = 0.35
    max_confidence_delta: float = 0.50
    max_decay_delta: float = 0.50


@dataclass(frozen=True)
class PlasticitySignal:
    """Signals used to adapt memory ranking metadata without storage mutation."""

    success_count: int = 0
    correction_count: int = 0
    verification_count: int = 0
    days_since_verified: int = 0
    explicit_decay: float = 0.0
    superseded_by: str = ""
    confidence_delta: float = 0.0


@dataclass(frozen=True)
class PlasticityDecision:
    """Deterministic promotion/decay/supersede decision."""

    action: str = "maintain"
    promotion_delta: float = 0.0
    decay_delta: float = 0.0
    reinforcement_delta: float = 0.0
    confidence_delta: float = 0.0
    status: str = "active"
    superseded_by: str = ""
    reasons: tuple[str, ...] = ()


def _non_negative_int(value: Any, *, max_value: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    number = max(0, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


def _bounded_delta(value: Any, *, limit: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < -limit:
        return -limit
    if number > limit:
        return limit
    return number


def _as_reasons(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def normalize_plasticity_signal(value: Any) -> PlasticitySignal:
    """Normalize arbitrary signal metadata into a bounded PR17 signal."""

    if isinstance(value, PlasticitySignal):
        return value
    if not isinstance(value, Mapping):
        return PlasticitySignal()
    return PlasticitySignal(
        success_count=_non_negative_int(value.get("success_count"), max_value=50),
        correction_count=_non_negative_int(value.get("correction_count"), max_value=50),
        verification_count=_non_negative_int(value.get("verification_count"), max_value=50),
        days_since_verified=_non_negative_int(value.get("days_since_verified"), max_value=365),
        explicit_decay=clamp_signal(value.get("explicit_decay", 0.0)),
        superseded_by=str(value.get("superseded_by") or ""),
        confidence_delta=_bounded_delta(value.get("confidence_delta", 0.0), limit=0.5),
    )


def resolve_plasticity_decision(
    signal: PlasticitySignal | Mapping[str, Any] | None,
    config: PlasticityConfig | None = None,
) -> PlasticityDecision:
    """Resolve a deterministic promotion/decay/supersede action."""

    cfg = config or PlasticityConfig()
    sig = normalize_plasticity_signal(signal)
    reasons: list[str] = []
    if sig.superseded_by:
        return PlasticityDecision(
            action="supersede",
            decay_delta=min(cfg.max_decay_delta, 0.5),
            status="superseded",
            superseded_by=sig.superseded_by,
            reasons=("superseded",),
        )

    promotion = min(
        1.0,
        sig.success_count * cfg.success_weight + sig.verification_count * cfg.verification_weight,
    )
    decay = min(
        1.0,
        sig.correction_count * cfg.correction_weight
        + sig.explicit_decay * cfg.explicit_decay_weight
        + (cfg.staleness_weight if sig.days_since_verified >= cfg.stale_after_days else 0.0),
    )
    if sig.success_count:
        reasons.append("success")
    if sig.verification_count:
        reasons.append("verified")
    if sig.correction_count:
        reasons.append("correction")
    if sig.explicit_decay:
        reasons.append("explicit_decay")
    if sig.days_since_verified >= cfg.stale_after_days:
        reasons.append("stale")

    net = promotion - decay
    if net >= cfg.promotion_threshold:
        return PlasticityDecision(
            action="promote",
            promotion_delta=min(cfg.max_reinforcement_delta, net),
            decay_delta=-min(cfg.max_decay_delta, max(0.0, promotion * 0.35)),
            reinforcement_delta=min(cfg.max_reinforcement_delta, 0.1 + promotion * 0.25),
            confidence_delta=min(cfg.max_confidence_delta, 0.05 + promotion * 0.20 + sig.confidence_delta),
            status="active",
            reasons=tuple(reasons or ["promotion_signal"]),
        )
    if -net >= cfg.decay_threshold:
        return PlasticityDecision(
            action="decay",
            decay_delta=min(cfg.max_decay_delta, 0.1 + decay * 0.30),
            reinforcement_delta=-min(cfg.max_reinforcement_delta, 0.05 + decay * 0.15),
            confidence_delta=max(-cfg.max_confidence_delta, sig.confidence_delta - decay * 0.20),
            status="stale",
            reasons=tuple(reasons or ["decay_signal"]),
        )
    return PlasticityDecision(
        action="maintain",
        confidence_delta=sig.confidence_delta,
        status="active",
        reasons=tuple(reasons or ["balanced"]),
    )


def _signal_from_candidate_metadata(metadata: MemoryObjectMetadata) -> PlasticitySignal:
    notes = metadata.notes if isinstance(metadata.notes, Mapping) else None
    if not notes:
        return PlasticitySignal()
    raw = notes.get("plasticity")
    return normalize_plasticity_signal(raw)


def _has_signal(signal: PlasticitySignal) -> bool:
    return any(
        (
            signal.success_count,
            signal.correction_count,
            signal.verification_count,
            signal.days_since_verified,
            signal.explicit_decay,
            signal.superseded_by,
            signal.confidence_delta,
        )
    )


def apply_plasticity_to_candidate(
    candidate: MemoryCandidate,
    signal: PlasticitySignal | Mapping[str, Any] | None = None,
    config: PlasticityConfig | None = None,
) -> tuple[MemoryCandidate, PlasticityDecision]:
    """Return a plasticity-adjusted candidate and the decision used.

    If no explicit signal is passed, the function looks for
    `candidate.metadata.notes["plasticity"]`. The original candidate is never
    mutated or dropped.
    """

    normalized_candidate = candidate_with_normalized_metadata(candidate)
    metadata = normalize_memory_metadata(normalized_candidate.metadata)
    sig = normalize_plasticity_signal(signal) if signal is not None else _signal_from_candidate_metadata(metadata)
    decision = resolve_plasticity_decision(sig, config=config) if _has_signal(sig) else PlasticityDecision()

    new_reinforcement = clamp_signal(normalized_candidate.reinforcement + decision.reinforcement_delta)
    new_confidence = clamp_signal(normalized_candidate.confidence + decision.confidence_delta)
    new_decay = clamp_signal(normalized_candidate.decay_penalty + decision.decay_delta)
    notes = dict(metadata.notes) if isinstance(metadata.notes, Mapping) else {}
    notes["plasticity_action"] = decision.action
    notes["plasticity_reasons"] = list(decision.reasons)
    if decision.superseded_by:
        notes["plasticity_superseded_by"] = decision.superseded_by
    metadata_confidence = metadata.confidence
    if metadata_confidence is not None:
        metadata_confidence = clamp_signal(metadata_confidence + decision.confidence_delta)
    elif decision.confidence_delta:
        metadata_confidence = clamp_signal(decision.confidence_delta)
    updated_metadata = replace(
        metadata,
        confidence=metadata_confidence,
        reinforcement_count=metadata.reinforcement_count + (1 if decision.action == "promote" else 0),
        status=decision.status if decision.action in {"decay", "supersede"} else metadata.status,
        superseded_by=decision.superseded_by or metadata.superseded_by,
        notes=notes,
    )
    return (
        replace(
            normalized_candidate,
            reinforcement=new_reinforcement,
            confidence=new_confidence,
            decay_penalty=new_decay,
            metadata=updated_metadata,
        ),
        decision,
    )


def build_plasticity_metadata(decisions: Sequence[PlasticityDecision] | None) -> dict[str, Any]:
    """Build flat JSON-friendly cognition metadata for PR17."""

    items = [item for item in (decisions or []) if isinstance(item, PlasticityDecision)]
    actions = [item.action for item in items]
    return {
        "plasticity_enabled": bool(items),
        "plasticity_decision_count": len(items),
        "plasticity_actions": actions,
        "plasticity_promoted_count": actions.count("promote"),
        "plasticity_decayed_count": actions.count("decay"),
        "plasticity_superseded_count": actions.count("supersede"),
    }


def build_plasticity_context(decisions: Sequence[PlasticityDecision] | None) -> str:
    """Return compact human-readable plasticity summary."""

    items = [item for item in (decisions or []) if isinstance(item, PlasticityDecision)]
    if not items:
        return ""
    lines = ["Memory Plasticity"]
    for index, item in enumerate(items, start=1):
        reason = ",".join(item.reasons) if item.reasons else "none"
        suffix = f" superseded_by={item.superseded_by}" if item.superseded_by else ""
        lines.append(
            f"{index}. action={item.action} status={item.status} reasons={reason}{suffix}"
        )
    return "\n".join(lines)
