"""Deterministic memory ranking primitives for PR13/PR14.

This module is intentionally pure: no model calls, no tool calls, and no
provider access. Runtime code can feed structured candidates into it and get a
bounded, JSON-friendly ranking result back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


MemoryTier = str
_MEMORY_METADATA_STATUSES = {"active", "stale", "superseded", "inferred", "unverified"}


@dataclass(frozen=True)
class MemoryRankerConfig:
    """Configuration for deterministic memory ranking."""

    enabled: bool = True
    max_items: int = 8
    max_chars: int = 6000
    relevance_weight: float = 0.40
    recency_weight: float = 0.15
    importance_weight: float = 0.20
    reinforcement_weight: float = 0.10
    confidence_weight: float = 0.15
    decay_weight: float = 0.25


@dataclass(frozen=True)
class MemoryObjectMetadata:
    """Auditable metadata for a memory object (PR14)."""

    source_trace: tuple[str, ...] = ()
    compression_level: int = 0
    confidence: float | None = None
    last_verified_at: str = ""
    superseded_by: str = ""
    reinforcement_count: int = 0
    status: str = "unverified"
    notes: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MemoryCandidate:
    """A structured memory item ready for ranking."""

    text: str
    provider: str = ""
    layer: str = "semantic"
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    reinforcement: float = 0.0
    confidence: float = 0.0
    decay_penalty: float = 0.0
    source: str = ""
    metadata: Mapping[str, Any] | MemoryObjectMetadata | None = None
    object_id: str = ""


@dataclass(frozen=True)
class RankedMemory:
    """A candidate plus deterministic rank metadata."""

    candidate: MemoryCandidate
    score: float
    tier: MemoryTier
    rank: int = field(default=0)


def clamp_signal(value: Any) -> float:
    """Coerce a ranking signal into the inclusive [0.0, 1.0] range."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _safe_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_max_items(config: MemoryRankerConfig) -> int:
    return _safe_non_negative_int(config.max_items)


def _safe_max_chars(config: MemoryRankerConfig) -> int:
    return _safe_non_negative_int(config.max_chars)


def _normalize_source_trace(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None and str(item) != "")
    return (str(value),) if str(value) else ()


def normalize_memory_metadata(value: Any) -> MemoryObjectMetadata:
    """Normalize arbitrary provider metadata into PR14's auditable schema."""

    if isinstance(value, MemoryObjectMetadata):
        return value
    if not isinstance(value, Mapping):
        return MemoryObjectMetadata()

    confidence = value.get("confidence")
    normalized_confidence = None if confidence is None else clamp_signal(confidence)
    if confidence is not None:
        try:
            float(confidence)
        except (TypeError, ValueError):
            normalized_confidence = None

    status = str(value.get("status") or "unverified")
    if status not in _MEMORY_METADATA_STATUSES:
        status = "unverified"

    notes = value.get("notes")
    if not isinstance(notes, Mapping):
        notes = None

    return MemoryObjectMetadata(
        source_trace=_normalize_source_trace(value.get("source_trace")),
        compression_level=_safe_non_negative_int(value.get("compression_level", 0)),
        confidence=normalized_confidence,
        last_verified_at=str(value.get("last_verified_at") or ""),
        superseded_by=str(value.get("superseded_by") or ""),
        reinforcement_count=_safe_non_negative_int(value.get("reinforcement_count", 0)),
        status=status,
        notes=notes,
    )


def memory_metadata_to_dict(metadata: MemoryObjectMetadata | Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-friendly metadata dictionary."""

    normalized = normalize_memory_metadata(metadata)
    return {
        "source_trace": list(normalized.source_trace),
        "compression_level": normalized.compression_level,
        "confidence": normalized.confidence,
        "last_verified_at": normalized.last_verified_at,
        "superseded_by": normalized.superseded_by,
        "reinforcement_count": normalized.reinforcement_count,
        "status": normalized.status,
        "notes": dict(normalized.notes) if isinstance(normalized.notes, Mapping) else None,
    }


def candidate_with_normalized_metadata(candidate: MemoryCandidate) -> MemoryCandidate:
    """Return a copy of candidate with PR14 metadata normalized."""

    return replace(candidate, metadata=normalize_memory_metadata(candidate.metadata))


def memory_metadata_label(metadata: MemoryObjectMetadata | Mapping[str, Any] | None) -> str:
    """Build a compact, auditable label for prompt headers."""

    normalized = normalize_memory_metadata(metadata)
    parts = [f"status={normalized.status}"]
    if normalized.confidence is not None:
        parts.append(f"confidence={normalized.confidence:.2f}")
    if normalized.last_verified_at:
        parts.append(f"verified={normalized.last_verified_at}")
    if normalized.source_trace:
        parts.append(f"source_trace={'>'.join(normalized.source_trace)}")
    if normalized.superseded_by:
        parts.append(f"superseded_by={normalized.superseded_by}")
    if normalized.compression_level:
        parts.append(f"compression={normalized.compression_level}")
    if normalized.reinforcement_count:
        parts.append(f"reinforced={normalized.reinforcement_count}")
    return " ".join(parts)


def score_memory_candidate(
    candidate: MemoryCandidate,
    config: MemoryRankerConfig | None = None,
) -> float:
    """Return a deterministic weighted score for a memory candidate."""

    cfg = config or MemoryRankerConfig()
    score = (
        cfg.relevance_weight * clamp_signal(candidate.relevance)
        + cfg.recency_weight * clamp_signal(candidate.recency)
        + cfg.importance_weight * clamp_signal(candidate.importance)
        + cfg.reinforcement_weight * clamp_signal(candidate.reinforcement)
        + cfg.confidence_weight * clamp_signal(candidate.confidence)
        - cfg.decay_weight * clamp_signal(candidate.decay_penalty)
    )
    return max(0.0, min(1.0, score))


def memory_tier_for_score(score: float) -> MemoryTier:
    """Map a score into a coarse memory tier."""

    value = clamp_signal(score)
    if value >= 0.75:
        return "hot"
    if value >= 0.50:
        return "warm"
    if value >= 0.25:
        return "cold"
    return "archive"


def _tier_priority(tier: str) -> int:
    return {"hot": 0, "warm": 1, "cold": 2, "archive": 3}.get(tier, 4)


def _candidate_tiebreaker(candidate: MemoryCandidate) -> tuple[str, str, str]:
    return (candidate.provider or "", candidate.source or "", candidate.text or "")


def rank_memory_candidates(
    candidates: Sequence[MemoryCandidate],
    config: MemoryRankerConfig | None = None,
) -> list[RankedMemory]:
    """Rank memory candidates and return a bounded list.

    When disabled, the function still returns RankedMemory wrappers and applies
    `max_items`, but preserves input order. This keeps callers bounded without
    pretending a ranking decision was made.
    """

    cfg = config or MemoryRankerConfig()
    max_items = _safe_max_items(cfg)
    if max_items <= 0:
        return []

    ranked: list[RankedMemory] = []
    for candidate in candidates:
        if not isinstance(candidate, MemoryCandidate):
            continue
        score = score_memory_candidate(candidate, cfg)
        ranked.append(
            RankedMemory(
                candidate=candidate,
                score=score,
                tier=memory_tier_for_score(score),
            )
        )

    if cfg.enabled:
        ranked.sort(
            key=lambda item: (
                -item.score,
                _tier_priority(item.tier),
                *_candidate_tiebreaker(item.candidate),
            )
        )

    bounded = ranked[:max_items]
    return [
        RankedMemory(
            candidate=item.candidate,
            score=item.score,
            tier=item.tier,
            rank=index,
        )
        for index, item in enumerate(bounded, start=1)
    ]


def build_ranked_memory_context(
    ranked: Sequence[RankedMemory],
    config: MemoryRankerConfig | None = None,
) -> str:
    """Format ranked memories as bounded text for prompt injection."""

    cfg = config or MemoryRankerConfig()
    max_chars = _safe_max_chars(cfg)
    if max_chars <= 0:
        return ""

    parts: list[str] = []
    total = 0
    for item in ranked:
        if not isinstance(item, RankedMemory):
            continue
        candidate = item.candidate
        text = (candidate.text or "").strip()
        if not text:
            continue
        header_bits = [
            f"rank={item.rank}",
            f"tier={item.tier}",
            f"score={item.score:.3f}",
        ]
        if candidate.provider:
            header_bits.append(f"provider={candidate.provider}")
        if candidate.layer:
            header_bits.append(f"layer={candidate.layer}")
        if candidate.source:
            header_bits.append(f"source={candidate.source}")
        if candidate.object_id:
            header_bits.append(f"object_id={candidate.object_id}")
        metadata_label = memory_metadata_label(candidate.metadata)
        if metadata_label:
            header_bits.append(metadata_label)
        block = f"[{' '.join(header_bits)}]\n{text}"
        separator = "\n\n" if parts else ""
        available = max_chars - total - len(separator)
        if available <= 0:
            break
        if len(block) > available:
            block = block[:available].rstrip()
        if not block:
            break
        parts.append(block)
        total += len(separator) + len(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
