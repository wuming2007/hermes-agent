"""Deterministic policy / constitution memory primitives (PR15)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from agent.memory_ranker import MemoryCandidate, MemoryObjectMetadata

_ALLOWED_CATEGORIES = {
    "safety",
    "privacy",
    "external_action",
    "memory",
    "style",
    "workflow",
    "general",
}
_ALLOWED_SCOPES = {"global", "user", "project", "platform"}
_EXTERNAL_ACTION_TERMS = (
    "send",
    "email",
    "publish",
    "tweet",
    "delete",
    "寄信",
    "發送",
    "發布",
    "刪除",
)
_MEMORY_PRIVACY_TERMS = (
    "remember",
    "memory",
    "private",
    "privacy",
    "資料",
    "記憶",
    "隱私",
)


@dataclass(frozen=True)
class PolicyMemoryItem:
    """A first-class policy / constitution memory item."""

    policy_id: str
    title: str
    text: str
    category: str = "general"
    priority: int = 50
    scope: str = "global"
    source_trace: tuple[str, ...] = ()
    version: str = "1"
    enabled: bool = True
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PolicyRecallResult:
    """A recalled policy item plus deterministic citation metadata."""

    item: PolicyMemoryItem
    score: float
    matched_terms: tuple[str, ...]
    citation: str
    rank: int = field(default=0)


def _as_non_empty_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clamp_priority(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 50
    return max(0, min(100, number))


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None and str(item) != "")
    text = str(value)
    return (text,) if text else ()


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def normalize_policy_memory_item(value: Any) -> PolicyMemoryItem | None:
    """Normalize provider/user policy data into a PolicyMemoryItem."""

    if isinstance(value, PolicyMemoryItem):
        return value
    if not isinstance(value, Mapping):
        return None

    policy_id = _as_non_empty_str(value.get("policy_id"))
    title = _as_non_empty_str(value.get("title"))
    text = _as_non_empty_str(value.get("text"))
    if not policy_id or not title or not text:
        return None

    category = _as_non_empty_str(value.get("category")) or "general"
    if category not in _ALLOWED_CATEGORIES:
        category = "general"
    scope = _as_non_empty_str(value.get("scope")) or "global"
    if scope not in _ALLOWED_SCOPES:
        scope = "global"

    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = None

    return PolicyMemoryItem(
        policy_id=policy_id,
        title=title,
        text=text,
        category=category,
        priority=_clamp_priority(value.get("priority", 50)),
        scope=scope,
        source_trace=_as_str_tuple(value.get("source_trace")),
        version=_as_non_empty_str(value.get("version")) or "1",
        enabled=_as_bool(value.get("enabled", True), True),
        tags=_as_str_tuple(value.get("tags")),
        metadata=metadata,
    )


def _terms_for_item(item: PolicyMemoryItem) -> tuple[str, ...]:
    terms = set()
    for raw in (item.policy_id, item.title, item.text, item.category, *item.tags):
        text = str(raw).lower()
        for token in re.findall(r"[\w#@.-]+", text):
            if len(token) >= 2:
                terms.add(token)
    if item.category == "external_action":
        terms.update(_EXTERNAL_ACTION_TERMS)
    if item.category in {"privacy", "memory"}:
        terms.update(_MEMORY_PRIVACY_TERMS)
    return tuple(sorted(terms))


def score_policy_item(item: PolicyMemoryItem, query: str) -> tuple[float, tuple[str, ...]]:
    """Return deterministic score and matched terms for a policy item."""

    if not item.enabled:
        return 0.0, ()
    query_lower = (query or "").lower()
    matched = tuple(term for term in _terms_for_item(item) if term and term.lower() in query_lower)
    if not matched:
        return 0.0, ()
    score = min(1.0, (len(matched) * 0.15) + (item.priority / 100.0 * 0.5))
    if item.category in {"external_action", "privacy", "safety"}:
        score = min(1.0, score + 0.15)
    return score, matched


def recall_policy_memories(
    query: str,
    items: Sequence[Any],
    max_items: int = 5,
) -> list[PolicyRecallResult]:
    """Recall relevant policy memories for a query."""

    results: list[PolicyRecallResult] = []
    for raw in items:
        item = normalize_policy_memory_item(raw)
        if item is None or not item.enabled:
            continue
        score, matched = score_policy_item(item, query)
        if score <= 0:
            continue
        results.append(
            PolicyRecallResult(
                item=item,
                score=score,
                matched_terms=matched,
                citation="",
            )
        )

    results.sort(key=lambda result: (-result.score, -result.item.priority, result.item.policy_id))
    bounded = results[: max(0, int(max_items))]
    return [
        PolicyRecallResult(
            item=result.item,
            score=result.score,
            matched_terms=result.matched_terms,
            citation=f"policy:{result.item.policy_id}@{rank}",
            rank=rank,
        )
        for rank, result in enumerate(bounded, start=1)
    ]


def build_policy_memory_context(results: Sequence[PolicyRecallResult]) -> str:
    """Format recalled policies as compact prompt context."""

    blocks: list[str] = []
    for result in results:
        item = result.item
        header = [
            result.citation,
            f"category={item.category}",
            f"priority={item.priority}",
            f"scope={item.scope}",
        ]
        if item.source_trace:
            header.append(f"source_trace={'>'.join(item.source_trace)}")
        blocks.append(f"[{' '.join(header)}]\n{item.title}: {item.text}")
    return "\n\n".join(blocks)


def build_policy_recall_metadata(results: Sequence[PolicyRecallResult]) -> dict[str, Any]:
    """Build JSON-friendly policy recall metadata for cognition traces."""

    return {
        "enabled": bool(results),
        "count": len(results),
        "policy_ids": [result.item.policy_id for result in results],
        "citations": [result.citation for result in results],
        "categories": [result.item.category for result in results],
    }


def policy_item_to_candidate(item: PolicyMemoryItem, query: str = "") -> MemoryCandidate:
    """Convert a policy item into a principles-layer memory candidate."""

    score, matched = score_policy_item(item, query)
    relevance = score if score > 0 else 0.5
    metadata = MemoryObjectMetadata(
        source_trace=item.source_trace or (f"policy:{item.policy_id}",),
        confidence=1.0,
        status="active",
        notes={
            "policy_category": item.category,
            "policy_scope": item.scope,
            "policy_version": item.version,
            "matched_terms": list(matched),
        },
    )
    return MemoryCandidate(
        text=f"{item.title}: {item.text}",
        provider="policy",
        layer="principles",
        relevance=relevance,
        importance=item.priority / 100.0,
        confidence=1.0,
        source=f"policy:{item.policy_id}",
        metadata=metadata,
        object_id=item.policy_id,
    )
