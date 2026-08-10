"""Uncertainty-aware cognition policy (PR5).

This module sits above the PR1 cognitive router.  It keeps the first version
intentionally deterministic and pure: given the user message, the current
``CognitiveRoute`` and the normalized cognition config, decide whether the turn
can be answered directly, should escalate depth, should seek tool evidence, or
should ask the human instead of hard-answering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from agent.cognitive_router import CognitiveRoute, Mode

DecisionAction = Literal["answer", "escalate_depth", "seek_tool", "seek_human"]
ConfidenceBand = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class UncertaintyDecision:
    """Decision produced by the uncertainty policy for a single turn."""

    confidence_band: ConfidenceBand
    action: DecisionAction
    escalate_depth: bool
    require_tool_evidence: bool
    seek_human: bool
    target_mode: Mode | None
    reasons: tuple[str, ...] = ()


_HISTORICAL_CUES: tuple[str, ...] = (
    "上次",
    "之前",
    "先前",
    "記得",
    "回顧",
    "last time",
    "previously",
    "earlier",
    "history",
    "historical",
)

_ARCHITECTURE_CUES: tuple[str, ...] = (
    "architecture",
    "design",
    "tradeoff",
    "trade-off",
    "架構",
    "設計",
    "方案",
)

_RISKY_EXTERNAL_CUES: tuple[str, ...] = (
    "send",
    "email",
    "publish",
    "tweet",
    "post",
    "deploy",
    "delete",
    "remove",
    "push",
    "寄信",
    "發送",
    "發布",
    "部署",
    "刪除",
    "推送",
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
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _matches_keyword(text_lower: str, keyword: str) -> bool:
    keyword_lower = keyword.lower()
    if keyword_lower.isascii():
        return re.search(rf"(?<!\w){re.escape(keyword_lower)}(?!\w)", text_lower) is not None
    return keyword_lower in text_lower


def _has_any(text_lower: str, keywords: tuple[str, ...]) -> bool:
    return any(_matches_keyword(text_lower, keyword) for keyword in keywords)


def _next_depth(mode: Mode) -> Mode | None:
    if mode == "fast":
        return "standard"
    if mode == "standard":
        return "deep"
    return None


def resolve_uncertainty_decision(
    *,
    user_message: str,
    cognition_route: CognitiveRoute | None,
    routing_config: dict[str, Any] | None,
    agent_state: dict[str, Any] | None = None,
) -> UncertaintyDecision | None:
    """Resolve uncertainty policy for the current turn.

    ``None`` means the policy is disabled or has no route to refine; callers
    should preserve legacy behavior unchanged in that case.
    """

    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    policy_cfg = cfg.get("uncertainty_policy") or {}
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    if not _coerce_bool(policy_cfg.get("enabled"), True):
        return None
    if cognition_route is None:
        return None

    # Reserved for future telemetry-driven anomaly/evidence signals.
    agent_state = agent_state or {}

    text = (user_message or "").strip()
    text_lower = text.lower()
    reasons: list[str] = []

    has_historical = _has_any(text_lower, _HISTORICAL_CUES)
    has_architecture = _has_any(text_lower, _ARCHITECTURE_CUES)
    has_risky_external = _has_any(text_lower, _RISKY_EXTERNAL_CUES)
    has_evidence_cue = bool(_URL_RE.search(text)) or "```" in text or "`" in text
    has_tool_evidence = _coerce_bool(agent_state.get("has_tool_evidence"), False)

    if has_historical:
        reasons.append("historical_cue")
    if has_architecture:
        reasons.append("architecture_cue")
    if has_risky_external:
        reasons.append("risky_external_action")
    if has_evidence_cue:
        reasons.append("evidence_cue")

    require_tool_for_risky = _coerce_bool(
        policy_cfg.get("require_tool_evidence_for_risky_actions"), True
    )
    seek_human_for_unverified = _coerce_bool(
        policy_cfg.get("seek_human_for_unverified_external_actions"), True
    )

    if has_risky_external and require_tool_for_risky and not has_tool_evidence:
        seek_human = seek_human_for_unverified
        return UncertaintyDecision(
            confidence_band="low",
            action="seek_human" if seek_human else "seek_tool",
            escalate_depth=False,
            require_tool_evidence=True,
            seek_human=seek_human,
            target_mode=None,
            reasons=tuple(reasons or ["risky_external_action"]),
        )

    escalate_fast_on_mismatch = _coerce_bool(
        policy_cfg.get("escalate_fast_on_mismatch"), True
    )
    escalate_standard_on_history_or_architecture = _coerce_bool(
        policy_cfg.get("escalate_standard_on_history_or_architecture"), True
    )

    should_escalate = False
    if cognition_route.mode == "fast" and escalate_fast_on_mismatch:
        should_escalate = has_historical or has_architecture or has_evidence_cue
    elif cognition_route.mode == "standard" and escalate_standard_on_history_or_architecture:
        should_escalate = has_historical or has_architecture

    if should_escalate:
        target = _next_depth(cognition_route.mode)
        return UncertaintyDecision(
            confidence_band="medium",
            action="escalate_depth",
            escalate_depth=target is not None,
            require_tool_evidence=False,
            seek_human=False,
            target_mode=target,
            reasons=tuple(reasons or ["route_message_mismatch"]),
        )

    if cognition_route.mode == "fast" and not reasons:
        return UncertaintyDecision(
            confidence_band="high",
            action="answer",
            escalate_depth=False,
            require_tool_evidence=False,
            seek_human=False,
            target_mode=None,
            reasons=("low_risk_fast_route",),
        )

    return UncertaintyDecision(
        confidence_band="medium",
        action="answer",
        escalate_depth=False,
        require_tool_evidence=False,
        seek_human=False,
        target_mode=None,
        reasons=tuple(reasons or ["route_accepted"]),
    )
