"""Cognitive router: per-turn mode selection scaffold (PR1).

Given a user message, recent conversation history, and the ``cognition`` config
block, this module classifies the upcoming turn into ``fast``, ``standard``, or
``deep`` and returns a ``CognitiveRoute`` carrying retrieval/verification plan
metadata for downstream PRs to consume.

PR1 scope (intentionally small):

- Pure heuristics over the user message text.
- ``conversation_history`` and ``agent_state`` are accepted for forward
  compatibility; later PRs may use them for stickiness or workspace signals.
- Returns ``None`` when cognition is disabled, so callers can preserve existing
  behavior verbatim.
- Does not modify memory schema, system prompt, or model selection here. It
  only produces routing metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


Mode = Literal["fast", "standard", "deep"]
RetrievalPlan = Literal[
    "none",
    "principles_only",
    "principles_plus_semantic",
    "principles_plus_semantic_plus_episodic",
]
VerificationPlan = Literal["none", "light", "full"]


@dataclass
class CognitiveRoute:
    """Result of routing a single turn.

    Fields are intentionally generic and persona-neutral; private layers may
    inspect them but should not rename or repurpose them.
    """

    mode: Mode
    retrieval_plan: RetrievalPlan
    verification_plan: VerificationPlan
    allow_cheap_model: bool
    consistency_check: bool
    routing_reasons: list[str] = field(default_factory=list)


# Keyword sets per deep-mode trigger category. Mixed-language entries cover the
# most common hermes user surfaces; later PRs may make these configurable.
_HISTORICAL_KEYWORDS: tuple[str, ...] = (
    "上次", "之前", "記得", "先前", "回顧",
    "previously", "earlier", "last time",
    "why did", "root cause", "retrospective", "looking back",
)

_ARCHITECTURE_KEYWORDS: tuple[str, ...] = (
    "architecture", "design", "tradeoff", "trade-off",
    "重構", "設計", "方案", "架構",
)

_CODE_CHANGE_KEYWORDS: tuple[str, ...] = (
    "patch", "edit", "refactor", "commit", "delete", "deploy",
    "修改", "改寫", "提交",
)

_RISKY_EXTERNAL_KEYWORDS: tuple[str, ...] = (
    "email", "push", "publish", "tweet", "send",
    "寄信", "發布", "發送",
)

# (config_key, reason_label_prefix, keyword_tuple)
_DEEP_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("historical_questions", "historical", _HISTORICAL_KEYWORDS),
    ("architecture_decisions", "architecture", _ARCHITECTURE_KEYWORDS),
    ("code_changes", "code_change", _CODE_CHANGE_KEYWORDS),
    ("risky_external_actions", "risky_external", _RISKY_EXTERNAL_KEYWORDS),
)

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if not v:
            return default
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _matches_keyword(text_lower: str, keyword: str) -> bool:
    """Match keyword against lowercased text.

    For ASCII-only keywords use word-boundary matching to avoid surprises like
    ``edit`` triggering on ``credit``. For non-ASCII keywords (e.g. Chinese)
    fall back to substring matching since CJK has no word boundaries.
    """
    kw = keyword.lower()
    if kw.isascii():
        return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text_lower) is not None
    return kw in text_lower


def _hits_category(text_lower: str, keywords: Iterable[str]) -> str | None:
    for kw in keywords:
        if _matches_keyword(text_lower, kw):
            return kw
    return None


def resolve_cognitive_route(
    user_message: str,
    conversation_history: list[dict[str, Any]] | None,
    routing_config: dict[str, Any] | None,
    agent_state: dict[str, Any] | None,
) -> CognitiveRoute | None:
    """Classify the upcoming turn into fast/standard/deep.

    Returns ``None`` when cognition is disabled so callers can keep their
    existing single-mode behavior unchanged.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    # ``conversation_history`` / ``agent_state`` reserved for future PRs.
    del conversation_history, agent_state

    text = (user_message or "").strip()
    text_lower = text.lower()

    deep_triggers_cfg = cfg.get("deep_mode_triggers") or {}
    consistency_cfg = cfg.get("consistency_guard") or {}
    fast_cfg = cfg.get("fast_mode") or {}

    # 1. Deep-mode triggers (only categories explicitly enabled in config).
    deep_reasons: list[str] = []
    for cfg_key, label_prefix, keywords in _DEEP_CATEGORIES:
        if not _coerce_bool(deep_triggers_cfg.get(cfg_key), True):
            continue
        hit = _hits_category(text_lower, keywords)
        if hit:
            deep_reasons.append(f"{label_prefix}:{hit}")

    if deep_reasons:
        consistency_check = _coerce_bool(consistency_cfg.get("enabled"), True)
        return CognitiveRoute(
            mode="deep",
            retrieval_plan="principles_plus_semantic_plus_episodic",
            verification_plan="full",
            allow_cheap_model=False,
            consistency_check=consistency_check,
            routing_reasons=deep_reasons,
        )

    # 2. Empty / whitespace-only message: standard is the safe default; cheap
    # routing on an empty message is meaningless.
    if not text:
        return CognitiveRoute(
            mode="standard",
            retrieval_plan="principles_plus_semantic",
            verification_plan="light",
            allow_cheap_model=False,
            consistency_check=False,
            routing_reasons=["empty_message"],
        )

    # 3. Fast eligibility checks.
    max_chars = _coerce_int(fast_cfg.get("max_chars"), 160)
    max_words = _coerce_int(fast_cfg.get("max_words"), 28)
    allow_urls = _coerce_bool(fast_cfg.get("allow_urls"), False)
    allow_code = _coerce_bool(fast_cfg.get("allow_code_blocks"), False)

    fast_blockers: list[str] = []
    if len(text) > max_chars:
        fast_blockers.append("over_max_chars")
    if len(text.split()) > max_words:
        fast_blockers.append("over_max_words")
    if not allow_urls and _URL_RE.search(text):
        fast_blockers.append("contains_url")
    if not allow_code and ("```" in text or "`" in text):
        fast_blockers.append("contains_code_block")

    if not fast_blockers:
        return CognitiveRoute(
            mode="fast",
            retrieval_plan="principles_only",
            verification_plan="none",
            allow_cheap_model=True,
            consistency_check=False,
            routing_reasons=["short_simple"],
        )

    # 4. Standard fallback.
    return CognitiveRoute(
        mode="standard",
        retrieval_plan="principles_plus_semantic",
        verification_plan="light",
        allow_cheap_model=False,
        consistency_check=False,
        routing_reasons=fast_blockers,
    )
