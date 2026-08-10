"""Interaction stance / answer-density routing (PR10).

This layer is intentionally pure and deterministic.  It does not affect model
selection, retrieval, verification, prompts, or tool dispatch; it only produces
metadata describing how terse or expansive the final answer should be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

if False:  # pragma: no cover - type-only without runtime import cycle
    from agent.cognitive_router import CognitiveRoute

DialogueMode = Literal["query", "status", "exploration", "debate", "execution"]
AnswerDensity = Literal["brief", "standard", "expanded"]

_DIALOGUE_MODES: set[str] = {"query", "status", "exploration", "debate", "execution"}
_ANSWER_DENSITIES: set[str] = {"brief", "standard", "expanded"}


@dataclass(frozen=True)
class InteractionStance:
    """How the agent should shape the current answer.

    ``stance_reasons`` is tuple-backed to keep the dataclass immutable and easy
    to persist in JSON-friendly metadata after callers convert it to ``list``.
    """

    dialogue_mode: DialogueMode = "query"
    answer_density: AnswerDensity = "standard"
    stance_reasons: tuple[str, ...] = field(default_factory=tuple)


def _matches_keyword(text_lower: str, keyword: str) -> bool:
    kw = keyword.lower()
    if kw.isascii():
        return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text_lower) is not None
    return kw in text_lower


def _first_keyword(text_lower: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if _matches_keyword(text_lower, keyword):
            return keyword
    return None


def _coerce_dialogue_mode(value: Any, default: DialogueMode) -> DialogueMode:
    if isinstance(value, str) and value.strip().lower() in _DIALOGUE_MODES:
        return value.strip().lower()  # type: ignore[return-value]
    return default


def _coerce_answer_density(value: Any, default: AnswerDensity) -> AnswerDensity:
    if isinstance(value, str) and value.strip().lower() in _ANSWER_DENSITIES:
        return value.strip().lower()  # type: ignore[return-value]
    return default


_STATUS_KEYWORDS: tuple[str, ...] = (
    "status", "state", "progress", "currently", "where are we", "what remains",
    "目前", "狀態", "進度", "做到哪", "剩下", "還差", "現在如何", "完成了嗎",
)

_EXECUTION_KEYWORDS: tuple[str, ...] = (
    "do it", "fix", "patch", "edit", "commit", "run", "test", "create", "write",
    "organize", "delete", "deploy", "send", "publish", "implement",
    "幫我做", "修", "改", "修改", "提交", "跑測試", "建立", "寫入", "整理", "刪除", "部署", "發送", "發布", "實作",
)

_DEBATE_KEYWORDS: tuple[str, ...] = (
    "do you think", "disagree", "argue", "challenge", "evaluate", "tradeoff", "trade-off",
    "pros and cons", "should we", "is it better", "critique",
    "你覺得", "反駁", "不同意", "評估", "取捨", "利弊", "應不應該", "該不該", "批判",
)

_EXPLORATION_KEYWORDS: tuple[str, ...] = (
    "brainstorm", "explore", "ideate", "imagine", "philosophy", "vision", "why", "how would",
    "聊聊", "發想", "探索", "想像", "哲學", "願景", "為什麼", "怎麼看", "一起想",
)


def _route_mode(cognition_route: Any) -> str | None:
    mode = getattr(cognition_route, "mode", None)
    return mode if isinstance(mode, str) else None


def _fallback_for_route(cognition_route: Any) -> InteractionStance:
    mode = _route_mode(cognition_route)
    if mode == "fast":
        return InteractionStance("query", "brief", ("route:fast",))
    if mode == "deep":
        return InteractionStance("query", "standard", ("route:deep",))
    if mode == "standard":
        return InteractionStance("query", "standard", ("route:standard",))
    return InteractionStance("query", "standard", ("route:unknown",))


def _override_from_config(stance: InteractionStance, routing_config: dict[str, Any] | None) -> InteractionStance:
    cfg = routing_config or {}
    stance_cfg = cfg.get("interaction_stance") or {}
    if not isinstance(stance_cfg, dict):
        return stance

    mode_overrides = stance_cfg.get("mode_overrides") or {}
    if not isinstance(mode_overrides, dict):
        mode_overrides = {}
    density_overrides = stance_cfg.get("density_overrides") or {}
    if not isinstance(density_overrides, dict):
        density_overrides = {}

    dialogue_mode = _coerce_dialogue_mode(
        mode_overrides.get(stance.dialogue_mode),
        stance.dialogue_mode,
    )
    answer_density = _coerce_answer_density(
        density_overrides.get(dialogue_mode, density_overrides.get(stance.dialogue_mode)),
        stance.answer_density,
    )
    if dialogue_mode == stance.dialogue_mode and answer_density == stance.answer_density:
        return stance
    return InteractionStance(
        dialogue_mode=dialogue_mode,
        answer_density=answer_density,
        stance_reasons=stance.stance_reasons + ("config_override",),
    )


def resolve_interaction_stance(
    *,
    user_message: str,
    cognition_route: "CognitiveRoute | None",
    routing_config: dict[str, Any] | None,
    agent_state: dict[str, Any] | None = None,
) -> InteractionStance:
    """Resolve dialogue mode and answer density for a turn.

    Pure fail-safe helper: malformed inputs fall back to query/standard; callers
    can wrap it in try/except, but normal bad config should not raise.
    """
    del agent_state  # reserved for future stickiness/platform policies

    text = (user_message or "").strip()
    text_lower = text.lower()
    if not text:
        return _override_from_config(InteractionStance("query", "standard", ("empty_message",)), routing_config)

    hit = _first_keyword(text_lower, _STATUS_KEYWORDS)
    if hit:
        return _override_from_config(InteractionStance("status", "brief", (f"status:{hit}",)), routing_config)

    hit = _first_keyword(text_lower, _EXECUTION_KEYWORDS)
    if hit:
        return _override_from_config(InteractionStance("execution", "brief", (f"execution:{hit}",)), routing_config)

    hit = _first_keyword(text_lower, _DEBATE_KEYWORDS)
    if hit:
        return _override_from_config(InteractionStance("debate", "standard", (f"debate:{hit}",)), routing_config)

    hit = _first_keyword(text_lower, _EXPLORATION_KEYWORDS)
    if hit:
        return _override_from_config(InteractionStance("exploration", "expanded", (f"exploration:{hit}",)), routing_config)

    return _override_from_config(_fallback_for_route(cognition_route), routing_config)
